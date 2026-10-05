"""Bounded whole-body imitation/PPO trials with backward recovery resets."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .recovery_ppo import RecoveryPPO
from .skill_env import ENVIRONMENT_VERSION, SkillEnv
from .skill_eval import evaluate
from .skill_scenarios import prepare_recovery
from .skill_train import _stats_path, _warm_start


class TrajectoryEnv(SkillEnv):
    def __init__(self, skill: str, dataset: Path | None):
        super().__init__("run_train" if skill == "run" else skill,
                         recovery_sensor_memory=skill == "recover", recovery_phase_features=skill == "recover")
        self.bank = None
        self.backward_fraction = 0.75
        self.episode_is_rehearsal = False
        if dataset and skill == "recover":
            with np.load(dataset, allow_pickle=False) as saved:
                self.bank = {key: saved[key].copy() for key in saved.files}

    def set_backward_fraction(self, fraction: float) -> None:
        if fraction not in (0.75, 0.5, 0.25, 0.0):
            raise ValueError("Invalid backward curriculum fraction")
        self.backward_fraction = fraction

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["rehearsal"] = self.episode_is_rehearsal
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.training_skill != "recover":
            return observation, info
        # Retain mild balance without mislabelling it as a floor-recovery sample.
        self.episode_is_rehearsal = self.np_random.random() < 0.3
        if self.episode_is_rehearsal:
            self.recovery_level = 9
            observation, info = super().reset(options=options)
            self.recovery_level = None
            return observation, info
        if not self.backward_fraction:
            scenario = str(self.np_random.choice(("standard", "settled_floor", "dynamic_fall")))
            if scenario == "standard":
                return observation, info
            pose = self.recovery_pose
            if scenario == "dynamic_fall":
                observation, info = super().reset(options={"recovery_pose": pose, "recovery_tilt_deg": 0})
            return prepare_recovery(self, scenario, pose)
        if self.bank is None:
            return observation, info
        starts = self.bank["episode_starts"]
        episode = int(self.np_random.integers(len(starts)))
        start = int(starts[episode])
        end = int(starts[episode + 1]) if episode + 1 < len(starts) else len(self.bank["actions"])
        # Leave at least the two-second standing hold to be executed by the learner.
        last = max(start, end - 110)
        frame = start + int((last - start) * self.np_random.uniform(self.backward_fraction,
                                                                  min(1.0, self.backward_fraction + 0.15)))
        for name in ("qpos", "qvel"):
            getattr(self.data, name)[:] = self.bank[name][frame]
        self.data.ctrl[:] = self.bank["torque"][frame]
        self.target = self.bank["target"][frame].copy()
        self.action_buffer.clear()
        self.action_buffer.extend(self.bank["action_buffer"][frame].copy())
        self.sensor_buffer.clear()
        self.sensor_buffer.extend(self.bank["sensor_buffer"][frame].copy())
        self.last_action = self.bank["last_action"][frame].copy()
        self.previous_action = self.bank["previous_action"][frame].copy()
        self.step_count = int(self.bank["step"][frame])
        self.phase = float(self.bank["phase"][frame])
        self.recovery_initial_gravity[:] = self.bank["initial_gravity"][frame]
        mujoco.mj_forward(self.model, self.data)
        self.last_contacts, _, _ = self._contacts()
        return self._observe(), info


class TrialMonitor(BaseCallback):
    def __init__(self, skill: str, output: Path, deadline: float, dev_seed: int,
                 warmup_steps: int, initial: dict, interval: int = 25000):
        super().__init__()
        self.skill, self.output, self.deadline = skill, output, deadline
        self.dev_seed, self.warmup_steps, self.initial = dev_seed, warmup_steps, initial
        self.interval, self.last_probe = interval, 0
        self.recent = deque(maxlen=100)
        self.backward = 0.75
        self.start = 0
        self.actor = []
        self.best = initial["pass_rate"]
        self.last_progress = 0
        self.retention_failures = 0
        self.stop_reason = "step_budget"
        self.high_kl_count = 0
        self.last_update = -1

    def _on_training_start(self):
        self.start = self.num_timesteps
        self.last_progress = self.start
        self.last_probe = self.start
        self.actor = list(self.model.policy.mlp_extractor.policy_net.parameters()) + list(self.model.policy.action_net.parameters())
        self.actor += [self.model.policy.log_std]
        if self.warmup_steps:
            for parameter in self.actor:
                parameter.requires_grad_(False)
        if self.skill == "run":
            self.model.get_env().venv.env_method("set_run_training_speed", 0.3)

    def _on_step(self):
        elapsed = self.num_timesteps - self.start
        if time.monotonic() >= self.deadline:
            self.stop_reason = "time_budget"
            return False
        if elapsed >= self.warmup_steps and not self.actor[0].requires_grad:
            for parameter in self.actor:
                parameter.requires_grad_(True)
            print(f"[trial] actor enabled at {elapsed} additional steps", flush=True)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                if not info.get("rehearsal"):
                    self.recent.append(bool(info.get("success")))
                with (self.output / "training_episodes.jsonl").open("a") as handle:
                    handle.write(json.dumps({"additional_steps": elapsed, **{key: info.get(key) for key in (
                        "skill", "success", "termination_reason", "rehearsal", "episode", "recovery_pose",
                        "recovery_tilt_deg", "recovery_reward_totals")}}) + "\n")
        if self.model._n_updates != self.last_update:
            self.last_update = self.model._n_updates
            kl = self.model.logger.name_to_value.get("train/ppo_kl", 0.0)
            self.high_kl_count = self.high_kl_count + 1 if kl > 0.05 else 0
            if not np.isfinite(kl) or self.high_kl_count >= 3:
                self.stop_reason = "repeated_high_or_nonfinite_kl"
                return False
        if self.num_timesteps - self.last_probe < self.interval:
            return True
        self.last_probe = self.num_timesteps
        normalizer = self.model.get_env()
        if self.skill == "run":
            normalizer.venv.env_method("set_run_training_speed", min(0.6, 0.3 + 0.3 * elapsed / 150000))
        result = evaluate(self.model, normalizer, self.skill, self.dev_seed, 50)
        retention = evaluate(self.model, normalizer, "recover", self.dev_seed + 100, 50, 9) if self.skill == "recover" else (
            evaluate(self.model, normalizer, "nav", self.dev_seed + 100, 50) if self.skill == "run" else None)
        retained = retention is None or retention["criterion_met"]
        if self.skill == "recover" and len(self.recent) == 100 and np.mean(self.recent) >= 0.8 and retained and self.backward:
            self.backward = max(0.0, self.backward - 0.25)
            normalizer.venv.env_method("set_backward_fraction", self.backward)
            self.recent.clear()
            self.last_progress = self.num_timesteps
        if retained and result["pass_rate"] > self.best:
            self.best = result["pass_rate"]
            self.last_progress = self.num_timesteps
            self.model.save(self.output / "best")
            normalizer.save(self.output / "best_vecnormalize.pkl")
        self.retention_failures = 0 if retained else self.retention_failures + 1
        record = {"additional_steps": elapsed, "development": result, "retention": retention,
                  "backward_fraction": self.backward, "actor_enabled": self.actor[0].requires_grad}
        with (self.output / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.model.save(self.output / "latest")
        normalizer.save(self.output / "latest_vecnormalize.pkl")
        print(f"[trial] steps={elapsed} success={result['pass_rate']:.1%} retention={retained} "
              f"backward_fraction={self.backward}", flush=True)
        if self.retention_failures >= 3:
            self.stop_reason = "retention_failed_three_times"
            return False
        if self.num_timesteps - self.last_progress >= 250000:
            self.stop_reason = "no_progress_250000_steps"
            return False
        values = self.model.logger.name_to_value
        if any(not np.isfinite(value) for name, value in values.items()
               if name.startswith("train/") and np.isscalar(value)):
            self.stop_reason = "nonfinite_training_metric"
            return False
        return True


def demonstration_data(dataset: Path, replay: Path | None = None) -> dict:
    with np.load(dataset, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("observations", "actions", "episode_starts")}
    result["primary_frames"] = len(result["observations"])
    result["policy_means"] = result["actions"].copy()
    if replay:
        with np.load(replay, allow_pickle=False) as data:
            result["episode_starts"] = np.r_[result["episode_starts"], data["episode_starts"] + result["primary_frames"]]
            result["policy_means"] = np.concatenate((result["policy_means"],
                                                    data["policy_means"] if "policy_means" in data else data["actions"]))
            for key in ("observations", "actions"):
                result[key] = np.concatenate((result[key], data[key]))
    return result


def fit(model, normalizer, dataset: Path, updates: int, seed: int, replay: Path | None = None) -> None:
    data = demonstration_data(dataset, replay)
    raw, targets = data["observations"], data["policy_means"]
    normalizer.obs_rms.count = min(normalizer.obs_rms.count, 5000)
    normalizer.obs_rms.update(raw)
    observations = torch.as_tensor(normalizer.normalize_obs(raw), dtype=torch.float32)
    observations[:, 53:76] = 0
    first = model.policy.mlp_extractor.policy_net[0]
    with torch.no_grad():
        first.weight[:, 53:76] = 0
    targets = torch.as_tensor(targets, dtype=torch.float32)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    rng = np.random.default_rng(seed)
    for update in range(updates):
        indices = (np.r_[rng.integers(data["primary_frames"], size=358),
                         rng.integers(data["primary_frames"], len(observations), size=154)]
                   if replay else rng.integers(len(observations), size=512))
        predicted = model.policy.get_distribution(observations[indices]).distribution.mean
        loss = torch.nn.functional.mse_loss(predicted, targets[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if (update + 1) % 500 == 0:
            print(f"[fit] update={update + 1} loss={float(loss):.6g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skill", choices=("nav", "squat", "recover", "run", "jump"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--fit-updates", type=int, default=0)
    parser.add_argument("--steps", type=int, default=196608)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--dev-seed", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--config", type=Path, help="Versioned running curriculum configuration")
    args = parser.parse_args()
    if args.config:
        if args.skill != "run" or args.dataset or args.fit_updates or args.replay:
            parser.error("Running curriculum configuration requires --skill run without imitation data")
        from .run_training import train
        train(args)
        return
    if args.output.exists() or args.steps < 0 or args.hours <= 0 or args.fit_updates < 0:
        parser.error("Output must be new; budgets must be valid")
    if args.fit_updates and not args.dataset:
        parser.error("Fitting requires demonstrations")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(environment_version=ENVIRONMENT_VERSION, training_contract="whole_body_backward_v1",
                         recovery_version=12 if args.skill == "recover" else None,
                         recovery_sensor_memory=args.skill == "recover", recovery_phase_features=args.skill == "recover",
                         normalization_frozen=True, optimizer="fresh_Adam", critic="reset_before_trial",
                         workers=8, rollout_steps=768, learning_rate=1e-5, clip_range=0.1)
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    # Each subprocess constructs its own simulator; no live environments cross forks.
    base = SubprocVecEnv([lambda: Monitor(TrajectoryEnv(args.skill, args.dataset)) for _ in range(8)],
                         start_method="forkserver")
    normalizer = VecNormalize.load(str(_stats_path(args.checkpoint)), base)
    normalizer.training = False
    normalizer.norm_reward = args.skill == "recover"
    model = RecoveryPPO("MlpPolicy", normalizer, device="cpu", seed=args.seed, verbose=1,
                        n_steps=768, batch_size=512, n_epochs=4, learning_rate=1e-5,
                        gamma=0.995, clip_range=0.1, target_kl=0.015, max_grad_norm=1.0,
                        policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                                       "activation_fn": torch.nn.ELU, "log_std_init": -2.5257})
    try:
        _warm_start(model, normalizer, args.checkpoint)
        model.recovery_sensor_memory = args.skill == "recover"
        model.recovery_phase_features = args.skill == "recover"
        model.environment_version = ENVIRONMENT_VERSION
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
            model.policy.log_std.fill_(np.log(0.08))
        if args.fit_updates:
            fit(model, normalizer, args.dataset, args.fit_updates, args.seed, args.replay)
        model.save(args.output / "initialized")
        normalizer.save(args.output / "initialized_vecnormalize.pkl")
        initial = evaluate(model, normalizer, args.skill, args.dev_seed, 50)
        def retention():
            if args.skill == "recover":
                return evaluate(model, normalizer, "recover", args.dev_seed + 100, 50, 9)
            if args.skill == "run":
                return evaluate(model, normalizer, "nav", args.dev_seed + 100, 50)
            return None
        initial_retention = retention()
        (args.output / "initial_evaluation.json").write_text(json.dumps(initial, indent=2) + "\n")
        if args.dataset:
            data = demonstration_data(args.dataset, args.replay)
            model.set_demonstrations(data["observations"], data["actions"], normalizer, weight=10, updates=2,
                                     seed=args.seed, policy_means=data["policy_means"],
                                     episode_starts=data["episode_starts"])
        model._diagnostics_path = args.output / "updates.jsonl"
        callback = TrialMonitor(args.skill, args.output, time.monotonic() + args.hours * 3600,
                                args.dev_seed, args.warmup_steps, initial)
        if args.steps:
            model.learn(args.steps, callback=callback)
        # Frozen warmup parameters must not leak into future resumed trials.
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
        model.save(args.output / "final")
        normalizer.save(args.output / "final_vecnormalize.pkl")
        final = evaluate(model, normalizer, args.skill, args.dev_seed, 50)
        result = {"skill": args.skill, "initial": initial, "final": final,
                  "initial_retention": initial_retention, "final_retention": retention(),
                  "stop_reason": callback.stop_reason, "environment_version": ENVIRONMENT_VERSION,
                  "additional_steps": model.num_timesteps, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                                           for k, v in vars(args).items()}}
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
