"""Bounded v2 PPO trials with nominal retention, rollback, and complete stop reports."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import signal
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .command_walk import DEFAULTS, FAMILIES
from .command_walk_eval import compact, rank
from .command_walk_robust import CONTRACT, PROFILES, RobustCommandEnv
from .command_walk_robust_eval import evaluate
from .command_walk_train import save_bundle
from .run_training import FrozenObservationStats, RunningPPO, actor_parameters, reseed_after_loading
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path, _warm_start


def new_budget(steps, rollout):
    if steps < 0 or rollout < 1:
        raise ValueError("Invalid additional-step budget")
    return (steps // rollout) * rollout


def retention_passed(reference, report, tolerance=0.10):
    return all(
        report["families"][k]["pass_rate"] + tolerance + 1e-9 >= reference["families"][k]["pass_rate"] for k in FAMILIES
    )


def create_model(checkpoint, variant="aligned", seed=44, workers=8, level=0, rollout_steps=768):
    source = RunningPPO.load(checkpoint, device="cpu")
    contract = getattr(source, "command_walk_contract", {})
    if contract.get("contract") not in ("asimov_command_walk_v1", CONTRACT):
        raise ValueError("Use a v1/v2 command walking checkpoint")
    resume = contract["contract"] == CONTRACT
    if resume and contract["variant"] != variant:
        raise ValueError("Cannot change variant on resume")
    config = {
        **DEFAULTS,
        "contract": CONTRACT,
        "variant": variant,
        "workers": workers,
        "rollout_steps": rollout_steps,
        "batch_size": min(512, workers * rollout_steps),
    }
    if resume and (contract["workers"] != workers or contract["rollout_steps"] != rollout_steps):
        raise ValueError("Resume must preserve rollout shape")
    factories = [lambda: Monitor(RobustCommandEnv(level=level, variant=variant, training=True)) for _ in range(workers)]
    base = SubprocVecEnv(factories, start_method="forkserver") if workers > 1 else DummyVecEnv(factories)
    try:
        if resume:
            model = source
            norm = VecNormalize.load(str(_stats_path(checkpoint)), base)
            model.set_env(norm)
            base.env_method("set_level", model.command_walk_state["level"])
        else:
            norm = VecNormalize(base, gamma=config["gamma"])
            model = RunningPPO(
                "MlpPolicy",
                norm,
                device="cpu",
                seed=seed,
                verbose=1,
                n_steps=rollout_steps,
                batch_size=config["batch_size"],
                n_epochs=4,
                learning_rate=3e-5,
                gamma=0.995,
                clip_range=0.1,
                target_kl=0.015,
                max_grad_norm=1.0,
                policy_kwargs={
                    "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                    "activation_fn": torch.nn.ELU,
                },
            )
            _warm_start(model, norm, checkpoint)
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.zero_()
            model.num_timesteps = source.num_timesteps
            model.command_walk_contract = config
            model.command_walk_state = dict(
                level=level,
                pass_streak=0,
                regression_streak=0,
                warmup_until=model.num_timesteps + 2 * workers * rollout_steps,
                best_by_level={},
                last_evaluation=model.num_timesteps,
                last_full=model.num_timesteps,
                last_review=model.num_timesteps,
                rollback_count=0,
            )
        norm.obs_rms = FrozenObservationStats.from_stats(norm.obs_rms)
        norm.training, norm.norm_reward = True, True
        rng = reseed_after_loading(model, seed)
        return model, norm, resume, rng
    except Exception:
        base.close()
        raise


class RobustMonitor(BaseCallback):
    def __init__(self, output, start, deadline, pilot=False, quick_count=10, full_count=50):
        super().__init__()
        self.output, self.start, self.deadline = output, start, deadline
        self.pilot, self.quick_count, self.full_count = pilot, quick_count, full_count
        self.stop_reason, self.stop_requested, self.cancelled = "step_budget", False, False
        self.latest, self.last_heartbeat = None, 0.0
        self.last_update, self.bad_updates = -1, 0

    def stop(self, reason):
        self.stop_reason, self.stop_requested = reason, True
        if reason in ("interrupted", "time_budget"):
            self.cancelled = True

    def progress(self, family="", index=0, step=0):
        if time.time() >= self.deadline:
            self.stop("time_budget")
        if self.cancelled:
            raise TimeoutError(self.stop_reason)
        if time.time() - self.last_heartbeat >= 10:
            self.heartbeat("evaluating", family=family, episode=index, episode_step=step)
            print(f"[evaluation] {family} case={index+1} step={step}", flush=True)

    def heartbeat(self, phase="training", **extra):
        self.last_heartbeat = time.time()
        state = self.model.command_walk_state
        atomic_json(
            self.output / "status.json",
            dict(
                state="running",
                phase=phase,
                pid=os.getpid(),
                steps=self.model.num_timesteps,
                additional_steps=self.model.num_timesteps - self.start,
                level=state["level"],
                profile=PROFILES[state["level"]].name,
                heartbeat=time.time(),
                rollback_count=state["rollback_count"],
                evaluation=self.latest,
                **extra,
            ),
        )

    def probe(self, label, full=False):
        state = self.model.command_walk_state
        count = self.full_count if full else self.quick_count
        level = state["level"]
        nominal = evaluate(self.model, self.training_env, 0, 1001, count, progress=self.progress)
        current = (
            nominal
            if level == 0
            else evaluate(self.model, self.training_env, level, 1001, count, progress=self.progress)
        )
        result = dict(nominal=nominal, current=current, full=full, count=count)
        self.latest = {k: compact(v) if k in ("nominal", "current") else v for k, v in result.items()}
        state["last_evaluation"] = self.model.num_timesteps
        state.setdefault("review_reference", compact(current))
        if full:
            state["last_full"] = self.model.num_timesteps
            state.setdefault("retention_reference", compact(nominal))
            retained = retention_passed(state["retention_reference"], nominal)
            result["retained"] = self.latest["retained"] = retained
            state["regression_streak"] = 0 if retained else state["regression_streak"] + 1
            eligible = retained and nominal["criterion_met"]
            key = str(level)
            old = state["best_by_level"].get(key)
            if retained and (old is None or rank(current) > tuple(old["rank"])):
                path = self.output / "checkpoints" / f"{self.model.num_timesteps:09d}_{label}" / "policy.zip"
                state["best_by_level"][key] = dict(
                    checkpoint=str(path.resolve()), rank=list(rank(current)), report=self.latest
                )
            if retained:
                state["safe_checkpoint"] = state["best_by_level"][key]["checkpoint"]
            state["pass_streak"] = state["pass_streak"] + 1 if eligible and current["criterion_met"] else 0
            if label not in ("initial", "final"):
                if state["regression_streak"] >= 2:
                    state["rollback_pending"] = True
                elif not self.pilot and self.full_count >= 50 and state["pass_streak"] >= 2:
                    if level == len(PROFILES) - 1:
                        self.stop("ready_for_qualification")
                    else:
                        state["level"] += 1
                        state["pass_streak"] = 0
                        state["last_review"] = self.model.num_timesteps
                        state.pop("review_reference", None)
                        self.training_env.env_method("set_level", state["level"])
                        state["needs_entry_probe"] = True
                        print(f'[curriculum] {PROFILES[state["level"]].name}', flush=True)
        atomic_json(self.output / "evaluations" / f"{self.model.num_timesteps}_{label}.json", result)
        # Store updated curriculum/retention state with each immutable policy/statistics pair.
        checkpoint = save_bundle(self.model, self.training_env, self.output, label)
        atomic_json(self.output / "best.json", state["best_by_level"])
        print(
            f"[probe] steps={self.model.num_timesteps} full={full} level={level} "
            f'nominal={ {k:round(v["pass_rate"],2) for k,v in nominal["families"].items()} } '
            f'current={ {k:round(v["pass_rate"],2) for k,v in current["families"].items()} }',
            flush=True,
        )
        return result, checkpoint

    def rollback(self):
        state = self.model.command_walk_state
        path = Path(state["safe_checkpoint"])
        saved = RunningPPO.load(path, device="cpu")
        self.model.policy.load_state_dict(saved.policy.state_dict())
        self.model.policy.optimizer.load_state_dict(saved.policy.optimizer.state_dict())
        with _stats_path(path).open("rb") as f:
            old_norm = pickle.load(f)
        self.training_env.ret_rms = copy.deepcopy(old_norm.ret_rms)
        state["rollback_count"] += 1
        state["level"] = max(0, state["level"] - 1)
        state["pass_streak"] = state["regression_streak"] = 0
        state["rollback_pending"] = False
        state["needs_entry_probe"] = True
        state["last_review"] = self.model.num_timesteps
        state.pop("review_reference", None)
        self.training_env.env_method("set_level", state["level"])
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.training_env.num_envs, dtype=bool)
        lr = max(3e-6, 3e-5 * 0.5 ** state["rollback_count"])
        self.model.learning_rate = lr
        self.model.lr_schedule = lambda _: lr
        print(f'[rollback] source={path} level={state["level"]} lr={lr}', flush=True)
        atomic_json(
            self.output / "rollbacks" / f"{self.model.num_timesteps}.json",
            dict(source=str(path), level=state["level"], lr=lr),
        )
        if state["rollback_count"] >= 3:
            self.stop("repeated_regression")

    def _on_rollout_start(self):
        state = self.model.command_walk_state
        if self.model._n_updates != self.last_update:
            self.last_update = self.model._n_updates
            kl = self.model.logger.name_to_value.get("train/ppo_kl", 0.0)
            self.bad_updates = self.bad_updates + 1 if kl > 0.05 or not np.isfinite(kl) else 0
            if self.bad_updates >= 3:
                self.stop("unstable_updates")
        if self.stop_requested:
            return
        if state.pop("rollback_pending", False):
            self.rollback()
        if state.pop("needs_entry_probe", False):
            self.probe("entry", full=False)
        if self.model.num_timesteps - state["last_evaluation"] >= 100_000:
            result, _ = self.probe("quick", full=False)
            due = self.model.num_timesteps - state["last_full"] >= 250_000
            promising = result["nominal"]["criterion_met"] and result["current"]["criterion_met"]
            if due or promising:
                result, _ = self.probe("full", full=True)
            if self.model.num_timesteps - state["last_review"] >= 1_000_000:
                before = state.get("review_reference")
                after = compact(result["current"])
                improved = before is None or (
                    after["minimum_pass_rate"] >= before["minimum_pass_rate"] + 0.10
                    or (before["mean_violation"] > 1e-6 and after["mean_violation"] < 0.9 * before["mean_violation"])
                )
                atomic_json(
                    self.output / "reviews" / f"{self.model.num_timesteps}.json",
                    dict(before=before, after=after, improved=improved),
                )
                if not improved:
                    self.stop("behavioral_plateau")
                state["review_reference"], state["last_review"] = after, self.model.num_timesteps
        if state.pop("needs_entry_probe", False):
            self.probe("entry", full=False)
        enabled = self.model.num_timesteps >= state["warmup_until"]
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)

    def _on_step(self):
        if time.time() >= self.deadline:
            self.stop("time_budget")
        if time.time() - self.last_heartbeat >= 10:
            self.heartbeat()
        for info in self.locals.get("infos", []):
            if "episode" in info:
                row = {
                    k: info[k]
                    for k in ["family", "level", "success", "failures", "violation", "episode", "reward_totals"]
                }
                with (self.output / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(dict(steps=self.model.num_timesteps, **row)) + "\n")
        return not self.stop_requested


def train(args):
    torch.set_num_threads(1)
    if args.steps < 0 or args.hours <= 0 or not np.isfinite(args.hours) or args.workers not in range(1, 9):
        raise ValueError("Invalid run budget")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "status.json", dict(state="starting", pid=os.getpid(), heartbeat=time.time()))
    model, norm, resume, rng = create_model(
        args.checkpoint, args.variant, args.seed, args.workers, args.level, args.rollout_steps
    )
    start = model.num_timesteps
    model._diagnostics_path = args.output / "updates.jsonl"
    monitor = RobustMonitor(
        args.output, start, time.time() + args.hours * 3600, args.pilot, args.quick_count, args.full_count
    )
    monitor.init_callback(model)
    atomic_json(
        args.output / "provenance.json",
        dict(
            source=str(args.checkpoint.resolve()),
            source_sha256=sha256(args.checkpoint),
            normalization_sha256=sha256(_stats_path(args.checkpoint)),
            seed=args.seed,
            rng_hash=rng,
            resumed=resume,
            additional_budget=args.steps,
            contract=model.command_walk_contract,
            source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        ),
    )
    handlers = {
        sig: signal.signal(sig, lambda *_: monitor.stop("interrupted")) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    initial = final = checkpoint = None
    try:
        initial, _ = monitor.probe("initial", full=True)
        count = new_budget(args.steps, args.workers * args.rollout_steps)
        if count:
            model.learn(total_timesteps=count, reset_num_timesteps=False, callback=monitor)
        for p in actor_parameters(model):
            p.requires_grad_(True)
        final, checkpoint = monitor.probe("final", full=True)
    except TimeoutError:
        pass
    except Exception as error:
        monitor.stop("error")
        atomic_json(args.output / "error.json", dict(error=repr(error)))
        raise
    finally:
        try:
            for p in actor_parameters(model):
                p.requires_grad_(True)
            checkpoint = save_bundle(model, norm, args.output, "resume")
            report = dict(
                stop_reason=monitor.stop_reason,
                steps=model.num_timesteps,
                additional_steps=model.num_timesteps - start,
                checkpoint=str(checkpoint.resolve()),
                initial=initial,
                final=final,
                best_by_level=model.command_walk_state["best_by_level"],
            )
            atomic_json(args.output / "report.json", report)
            atomic_json(
                args.output / "status.json",
                dict(
                    state=(
                        "failed" if monitor.stop_reason == "error" else "stopped" if monitor.cancelled else "completed"
                    ),
                    stop_reason=monitor.stop_reason,
                    steps=model.num_timesteps,
                    additional_steps=model.num_timesteps - start,
                    level=model.command_walk_state["level"],
                    heartbeat=time.time(),
                    checkpoint=str(checkpoint.resolve()),
                ),
            )
        finally:
            norm.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["control", "aligned"], default="aligned")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--rollout-steps", type=int, default=768)
    p.add_argument("--quick-count", type=int, default=10)
    p.add_argument("--full-count", type=int, default=50)
    train(p.parse_args())


if __name__ == "__main__":
    main()
