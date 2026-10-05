"""Resumable PPO acquisition with fixed observations and stage-specific evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .recovery_ppo import RecoveryPPO
from .run_contract import RunningEnv, STAGES, config_from, summarize
from .skill_campaign import atomic_json, sha256
from .skill_eval import wilson_interval
from .skill_train import _stats_path, _warm_start


class FrozenObservationStats(RunningMeanStd):
    """Only observation statistics are frozen; VecNormalize still learns returns."""

    def update(self, arr):
        return None

    @classmethod
    def from_stats(cls, stats):
        result = cls(shape=stats.mean.shape)
        result.mean, result.var, result.count = stats.mean.copy(), stats.var.copy(), stats.count
        return result


class RunningPPO(RecoveryPPO):
    def train(self):
        # Hooks observe gradients before SB3 clips the combined norm. No optimizer
        # behavior is changed and hooks/wrappers are removed before serialization.
        pending, actor_norms, critic_norms, hooks = {}, [], [], []
        actor_ids = {id(p) for p in actor_parameters(self)}
        for p in self.policy.parameters():
            if p.requires_grad:
                hooks.append(p.register_hook(lambda g, key=id(p): pending.__setitem__(key, float(g.square().sum()))))
        original_step = self.policy.optimizer.step

        def step(*args, **kwargs):
            actor_norms.append(sum(v for k, v in pending.items() if k in actor_ids) ** 0.5)
            critic_norms.append(sum(v for k, v in pending.items() if k not in actor_ids) ** 0.5)
            pending.clear()
            return original_step(*args, **kwargs)

        self.policy.optimizer.step = step
        diagnostics = self._diagnostics_path
        self._diagnostics_path = None
        try:
            super().train()
        finally:
            self.policy.optimizer.step = original_step
            for hook in hooks:
                hook.remove()
            self._diagnostics_path = diagnostics
        self.logger.record("train/actor_gradient_norm", float(np.mean(actor_norms)) if actor_norms else 0.0)
        self.logger.record("train/critic_gradient_norm", float(np.mean(critic_norms)) if critic_norms else 0.0)
        self.logger.record("train/actor_enabled", float(self.policy.action_net.weight.requires_grad))
        values = {k.removeprefix("train/"): float(v) for k, v in self.logger.name_to_value.items()
                  if k.startswith("train/") and np.isscalar(v)}
        if any(not np.isfinite(v) for v in values.values()):
            raise FloatingPointError("Nonfinite PPO diagnostic; retain previous valid bundle")
        if diagnostics:
            with Path(diagnostics).open("a") as f:
                f.write(json.dumps({"training_steps": self.num_timesteps, **values}, allow_nan=False) + "\n")


def actor_parameters(model):
    return [*model.policy.mlp_extractor.policy_net.parameters(), *model.policy.action_net.parameters(),
            model.policy.log_std]


def reseed_after_loading(model, seed):
    """Restore the experiment RNG after SB3 checkpoint loading sets a global seed."""
    model.set_random_seed(seed + model.num_timesteps)
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes() + np.random.get_state()[1].tobytes()).hexdigest()


def evaluate_stage(model, normalizer, stage, seed_start, count=20, video=None, progress=None):
    env = RunningEnv(model.run_contract, stage, render_mode="rgb_array" if video else None)
    writer = None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=10, codec="libx264")
    episodes = []
    try:
        for seed in range(seed_start, seed_start + count):
            obs, _ = env.reset(seed=seed)
            for step in range(600):
                if progress and step % 50 == 0:
                    progress(seed - seed_start, step)
                action, _ = model.predict(normalizer.normalize_obs(obs[None]), deterministic=True)
                obs, _, terminated, truncated, info = env.step(action[0])
                if writer and step % 5 == 4:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            episodes.append({"seed": seed, **info})
    finally:
        env.close()
        if writer:
            writer.close()
    report = {"stage": stage, "stage_name": STAGES[stage], "seed_start": seed_start,
              "summary": summarize(episodes), "episodes": episodes, "training_steps": model.num_timesteps}
    report["wilson_95"] = wilson_interval(sum(e["success"] for e in episodes), count)
    return report


def save_bundle(model, normalizer, output: Path, label: str) -> Path:
    bundle = output / "checkpoints" / f"{model.num_timesteps:09d}_{label}"
    bundle.mkdir(parents=True, exist_ok=False)
    policy = bundle / "policy.zip"
    model.save(policy)
    normalizer.save(_stats_path(policy))
    atomic_json(bundle / "contract.json", model.run_contract)
    atomic_json(bundle / "state.json", model.run_state)
    atomic_json(bundle / "manifest.json", {p.name: sha256(p) for p in bundle.iterdir() if p.is_file()})
    atomic_json(output / "latest.json", {"checkpoint": str(policy.resolve()), "steps": model.num_timesteps})
    return policy


class RunningMonitor(BaseCallback):
    def __init__(self, output, config, deadline, dev_seed):
        super().__init__()
        self.output, self.config, self.deadline, self.dev_seed = output, config, deadline, dev_seed
        self.stop_reason, self.stop_requested = "step_budget", False
        self.previous_heartbeat = 0.0
        self.high_kl = 0
        self.last_update = -1

    def _on_rollout_start(self):
        state, c = self.model.run_state, self.config
        enabled = self.model.num_timesteps >= c["warmup_rollouts"] * c["workers"] * c["rollout_steps"]
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)
        values = self.model.logger.name_to_value
        if self.model._n_updates != self.last_update:
            self.last_update = self.model._n_updates
            kl = values.get("train/ppo_kl", 0.)
            self.high_kl = self.high_kl + 1 if kl > 0.05 else 0
            if not np.isfinite(kl) or self.high_kl >= 3:
                self.stop_reason, self.stop_requested = "repeated_high_or_nonfinite_kl", True
        if self.model.num_timesteps - state["last_evaluation"] >= c["evaluation_interval"]:
            result = evaluate_stage(self.model, self.training_env, state["stage"], self.dev_seed,
                                    c["development_episodes"], progress=self.phase_progress("evaluating"))
            state["last_evaluation"] = self.model.num_timesteps
            state["latest_evaluation"] = {k: v for k, v in result.items() if k != "episodes"}
            state["pass_streak"] = state["pass_streak"] + 1 if result["summary"]["pass_rate"] >= .8 else 0
            atomic_json(self.output / "evaluations" / f"{self.model.num_timesteps}.json", result)
            print(f"[probe] steps={self.model.num_timesteps} stage={result['stage_name']} "
                  f"metrics={result['summary']} streak={state['pass_streak']}", flush=True)
            if state["pass_streak"] >= 2:
                self.videos(result, "stage_pass")
                if state["stage"] == 7:
                    self.stop_reason, self.stop_requested = "ready_for_qualification", True
                else:
                    state["stage"] += 1
                    state["pass_streak"] = 0
                    self.training_env.env_method("set_run_stage", state["stage"])
                    print(f"[curriculum] next resets use {STAGES[state['stage']]}", flush=True)
            save_bundle(self.model, self.training_env, self.output, "probe")

    def videos(self, result, label):
        for success in (True, False):
            case = next((e for e in result["episodes"] if e["success"] == success), None)
            if case:
                filename = f"{self.model.num_timesteps}_{label}_{'success' if success else 'failure'}.mp4"
                evaluate_stage(self.model, self.training_env, result["stage"], case["seed"], 1,
                               self.output / "videos" / filename, progress=self.phase_progress("rendering"))

    def phase_progress(self, phase):
        def progress(episode, step):
            atomic_json(self.output / "status.json", {"state": "running", "phase": phase,
                "pid": os.getpid(), "heartbeat": time.time(), "steps": self.model.num_timesteps,
                "updates": self.model._n_updates, "stage": STAGES[self.model.run_state["stage"]],
                "episode": episode, "episode_step": step,
                "remaining_hours": max(0, self.deadline - time.time()) / 3600,
                "evaluation": self.model.run_state.get("latest_evaluation")})
        return progress

    def _on_step(self):
        now = time.time()
        if now >= self.deadline:
            self.stop_reason, self.stop_requested = "time_budget", True
        for info in self.locals.get("infos", []):
            if "episode" in info:
                with (self.output / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps({"steps": self.model.num_timesteps, **info}, default=lambda x: x.tolist()) + "\n")
        if now - self.previous_heartbeat >= 10:
            self.previous_heartbeat = now
            state = self.model.run_state
            atomic_json(self.output / "status.json", {"state": "running", "pid": os.getpid(),
                "heartbeat": now, "steps": self.model.num_timesteps, "updates": self.model._n_updates,
                "stage": STAGES[state["stage"]], "remaining_hours": max(0, self.deadline - now) / 3600,
                "actor_enabled": self.model.policy.action_net.weight.requires_grad,
                "evaluation": state.get("latest_evaluation"), "stop_reason": self.stop_reason})
        return not self.stop_requested


def train(args):
    config = config_from(args.config)
    if args.steps < 0 or not np.isfinite(args.hours) or args.hours <= 0:
        raise ValueError("Training budgets must be finite and nonnegative (hours positive)")
    torch.set_num_threads(1)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", config)
    source = RunningPPO.load(args.checkpoint, device="cpu")
    resume = getattr(source, "run_contract", None) is not None
    if resume and source.run_contract != config:
        raise ValueError("Resume configuration differs from checkpoint contract")
    factories = [lambda: Monitor(RunningEnv(config)) for _ in range(config["workers"])]
    base = SubprocVecEnv(factories, start_method="forkserver") if config["workers"] > 1 else DummyVecEnv(factories)
    normalizer = VecNormalize.load(str(_stats_path(args.checkpoint)), base)
    normalizer.obs_rms = FrozenObservationStats.from_stats(normalizer.obs_rms)
    normalizer.training, normalizer.norm_reward, normalizer.gamma = True, True, config["gamma"]
    if resume:
        model = source
        model.set_env(normalizer)
        # Resume workers with distinct reset streams; optimizer and curriculum persist.
        normalizer.seed(args.seed + model.num_timesteps)
        normalizer.env_method("set_run_stage", model.run_state["stage"])
    else:
        model = RunningPPO("MlpPolicy", normalizer, device="cpu", seed=args.seed, verbose=1,
            n_steps=config["rollout_steps"], batch_size=config["batch_size"], n_epochs=config["epochs"],
            learning_rate=config["learning_rate"], gamma=config["gamma"], clip_range=config["clip_range"],
            target_kl=config["target_kl"], max_grad_norm=config["max_grad_norm"],
            policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                           "activation_fn": torch.nn.ELU})
        _warm_start(model, normalizer, args.checkpoint)
        normalizer.ret_rms = RunningMeanStd(shape=())
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
            model.policy.log_std.fill_(np.log(config["initial_std"]))
        model.run_contract = config
        model.run_state = {"stage": 0, "pass_streak": 0, "last_evaluation": 0}
    # PPO.load inside _warm_start restores the source seed globally. Reseed only
    # after every checkpoint load, otherwise nominal trials silently coincide.
    rng_hash = reseed_after_loading(model, args.seed)
    del source
    model._diagnostics_path = args.output / "updates.jsonl"
    model.environment_version = 12
    snapshot = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    atomic_json(args.output / "provenance.json", {"source": str(args.checkpoint.resolve()), "source_hash": sha256(args.checkpoint),
                "source_hashes": snapshot, "seed": args.seed, "resumed": resume, "initial_rng_sha256": rng_hash})
    monitor = RunningMonitor(args.output, config, time.time() + args.hours * 3600, args.dev_seed)
    def stop(*_):
        monitor.stop_requested, monitor.stop_reason = True, "interrupted"
    previous_handlers = {sig: signal.signal(sig, stop)
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    start = model.num_timesteps
    try:
        monitor.model = model
        initial = evaluate_stage(model, normalizer, model.run_state["stage"], args.dev_seed, config["development_episodes"],
                                 progress=monitor.phase_progress("initial_evaluation"))
        atomic_json(args.output / "initial.json", initial)
        save_bundle(model, normalizer, args.output, "initial")
        rollout = config["workers"] * config["rollout_steps"]
        available = max(0, (config["maximum_steps"] - model.num_timesteps) // rollout) * rollout
        learning_steps = min(args.steps, available)
        if learning_steps:
            model.learn(total_timesteps=learning_steps, reset_num_timesteps=not resume, callback=monitor)
        for p in actor_parameters(model):
            p.requires_grad_(True)
        final = evaluate_stage(model, normalizer, model.run_state["stage"], args.dev_seed, config["development_episodes"],
                               progress=monitor.phase_progress("final_evaluation"))
        checkpoint = save_bundle(model, normalizer, args.output, "final")
        report = {"initial": initial, "final": final, "stop_reason": monitor.stop_reason,
                  "steps": model.num_timesteps, "additional_steps": model.num_timesteps - start,
                  "checkpoint": str(checkpoint.resolve()), "state": copy.deepcopy(model.run_state)}
        atomic_json(args.output / "report.json", report)
        monitor.videos(final, "final")
        atomic_json(args.output / "status.json", {"state": "completed", "steps": model.num_timesteps,
                    "stop_reason": monitor.stop_reason, "stage": STAGES[model.run_state["stage"]], "heartbeat": time.time()})
    except Exception as error:
        atomic_json(args.output / "status.json", {"state": "failed", "error": repr(error), "heartbeat": time.time(),
                    "last_valid_bundle": json.loads((args.output / "latest.json").read_text())})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        normalizer.close()
