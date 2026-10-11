"""V4 PPO jobs with explicit transfer/resume, actor budgets, and nominal skill gates."""

import argparse
import copy
import json
import os
import pickle
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_eval import compact
from .command_walk_robust_train import RobustMonitor, create_model, new_budget
from .command_walk_train import save_bundle
from .command_walk_v4 import INITIALIZATIONS, TARGETS, MotionEnv, Spec, panel, passes, retains, specialist_passes
from .run_training import RunningPPO, actor_parameters
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path


class DiagnosticPPO(RunningPPO):
    def train(self):
        b = self.rollout_buffer
        self.logger.record("train/advantage_mean", float(np.mean(b.advantages)))
        self.logger.record("train/advantage_std", float(np.std(b.advantages)))
        self.logger.record("train/value_target_rmse", float(np.sqrt(np.mean((b.returns - b.values) ** 2))))
        self.logger.record("train/return_std", float(np.std(b.returns)))
        self.logger.record("train/reward_scale", float(np.sqrt(self.get_env().ret_rms.var + self.get_env().epsilon)))
        step = self.policy.optimizer.step

        def checked_step(*args, **kwargs):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):
                raise FloatingPointError("Nonfinite gradient; refusing update")
            return step(*args, **kwargs)

        self.policy.optimizer.step = checked_step
        try:
            super().train()
        finally:
            self.policy.optimizer.step = step


def build(checkpoint, spec, seed, resume=False):
    preserve, warmup = INITIALIZATIONS[spec.initialization]
    model, norm, _, rng = create_model(
        checkpoint,
        variant="control",
        seed=seed,
        workers=spec.workers,
        rollout_steps=spec.rollout_steps,
        model_class=DiagnosticPPO,
        transfer=not resume,
        preserve_critic=preserve,
        warmup_rollouts=warmup,
        env_factory=lambda: MotionEnv(spec, training=True, gentle=spec.task in ("arcs", "pivots", "braking")),
    )
    if resume:
        if model.command_walk_contract.get("v4") != asdict(spec):
            norm.close()
            raise ValueError("Resume specification differs from checkpoint")
        state = model.command_walk_state
        norm.env_method("set_gentle", state["gentle"])
    else:
        model.command_walk_contract["v4"] = asdict(spec)
        model.command_walk_state.update(
            training_start=model.num_timesteps,
            actor_updates=0,
            last_saved=model.num_timesteps,
            gentle=spec.task in ("arcs", "pivots", "braking"),
            candidates=[],
            calibration_saved=False,
        )
    return model, norm, rng


class Monitor(RobustMonitor):
    def __init__(self, output, start, deadline, spec, cases=50, diagnostic=False):
        super().__init__(output, start, deadline, pilot=True, full_count=cases)
        self.spec, self.diagnostic = spec, diagnostic
        self.initial = None

    def heartbeat(self, phase=None, **extra):
        state = self.model.command_walk_state
        phase = phase or ("calibrating" if self.model.num_timesteps < state["warmup_until"] else "training")
        super().heartbeat(
            phase,
            actor_steps=max(0, self.model.num_timesteps - state["warmup_until"]),
            calibration_steps=min(self.model.num_timesteps, state["warmup_until"]) - state["training_start"],
            task=self.spec.task,
            initialization=self.spec.initialization,
            **extra,
        )

    def probe(self, label, full=False):
        state = self.model.command_walk_state
        count = self.full_count if full else 10
        cases = tuple(dict.fromkeys((*FAMILIES, *TARGETS[self.spec.task])))
        report = panel(self.model, self.training_env, cases, count, progress=self.progress)
        reference = state.setdefault("v4_reference", compact(report)) if full else state.get("v4_reference")
        keep_cases = ("forward", "stop") if self.spec.task in ("arcs", "pivots", "braking") else FAMILIES
        retained = reference is not None and retains(reference, report, cases=keep_cases)
        if self.spec.task in ("arcs", "pivots", "braking"):
            retained = passes(report, keep_cases, 0.92, count)
        self.latest = dict(nominal=compact(report), current=compact(report), full=full, count=count, retained=retained)
        state["last_evaluation"] = self.model.num_timesteps
        if full:
            state["last_full"] = self.model.num_timesteps
            if retained:
                targets = TARGETS[self.spec.task]
                rank = (
                    min(report["families"][k]["pass_rate"] for k in targets),
                    np.mean([report["families"][k]["pass_rate"] for k in targets]),
                )
                if "best_rank" not in state or rank > tuple(state["best_rank"]):
                    state["best_rank"] = rank
                    state["safe_checkpoint"] = str(
                        (
                            self.output / "checkpoints" / f"{self.model.num_timesteps:09d}_{label}" / "policy.zip"
                        ).resolve()
                    )
            if not self.diagnostic and label not in ("initial", "final") and not retained:
                # Confirm on another full panel at the same checkpoint before rollback.
                confirmation = panel(
                    self.model, self.training_env, cases, count, seed_start=5001, progress=self.progress
                )
                confirmed = (
                    not passes(confirmation, keep_cases, 0.92, count)
                    if self.spec.task in ("arcs", "pivots", "braking")
                    else not retains(reference, confirmation, cases=keep_cases)
                )
                if confirmed:
                    state["rollback_pending"] = True
                atomic_json(self.output / "evaluations" / f"{self.model.num_timesteps}_confirmation.json", confirmation)
            if state["gentle"]:
                gentle = panel(
                    self.model, self.training_env, TARGETS[self.spec.task], count, gentle=True, progress=self.progress
                )
                if passes(gentle, TARGETS[self.spec.task], 0.90, 50):
                    state["gentle"] = False
                    self.training_env.env_method("set_gentle", False)
                atomic_json(self.output / "evaluations" / f"{self.model.num_timesteps}_{label}_gentle.json", gentle)
        checkpoint = save_bundle(self.model, self.training_env, self.output, label)
        record = dict(
            report=report,
            full=full,
            retained=retained,
            checkpoint=str(checkpoint.resolve()),
            actor_steps=max(0, self.model.num_timesteps - state["warmup_until"]),
        )
        atomic_json(self.output / "evaluations" / f"{self.model.num_timesteps}_{label}.json", record)
        print(
            f'[probe] {label} actor_steps={record["actor_steps"]} retained={retained} '
            f'{ {k: v["pass_rate"] for k,v in report["families"].items()} }',
            flush=True,
        )
        return report, checkpoint

    def rollback(self):
        state = self.model.command_walk_state
        if "safe_checkpoint" not in state:
            self.stop("no_safe_checkpoint")
            return
        checkpoint = Path(state["safe_checkpoint"])
        saved = DiagnosticPPO.load(checkpoint, device="cpu")
        self.model.policy.load_state_dict(saved.policy.state_dict())
        self.model.policy.optimizer.load_state_dict(saved.policy.optimizer.state_dict())
        with _stats_path(checkpoint).open("rb") as f:
            self.training_env.ret_rms = copy.deepcopy(pickle.load(f).ret_rms)
        state["rollback_count"] += 1
        state["rollback_pending"] = False
        lr = 3e-5 * 0.5 ** state["rollback_count"]
        self.model.learning_rate = lr
        self.model.lr_schedule = lambda _: lr
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.training_env.num_envs, dtype=bool)
        atomic_json(self.output / "rollbacks" / f"{self.model.num_timesteps}.json", dict(source=str(checkpoint), lr=lr))
        if state["rollback_count"] >= 2:
            self.stop("repeated_regression")

    def _on_rollout_start(self):
        state = self.model.command_walk_state
        if state.pop("rollback_pending", False):
            self.rollback()
        enabled = self.model.num_timesteps >= state["warmup_until"]
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)
        if enabled and not state["calibration_saved"]:
            state["calibration_saved"] = True
            state["last_evaluation"] = self.model.num_timesteps
            save_bundle(self.model, self.training_env, self.output, "calibrated")
        actor_steps = max(0, self.model.num_timesteps - state["warmup_until"])
        if enabled and self.model.num_timesteps - state["last_saved"] >= 25_000:
            state["last_saved"] = self.model.num_timesteps
            save_bundle(self.model, self.training_env, self.output, "periodic")
        if enabled and actor_steps and self.model.num_timesteps - state["last_evaluation"] >= 50_000:
            # Calibration time is not counted toward the probe interval.
            report, _ = self.probe("quick", full=False)
            due = self.model.num_timesteps - state["last_full"] >= 150_000
            promising = passes(report, TARGETS[self.spec.task], 0.90, 10)
            if not self.diagnostic and (due or promising):
                self.probe("full", full=True)
        kl = self.model.logger.name_to_value.get("train/ppo_kl", 0.0)
        if not np.isfinite(kl):
            self.stop("nonfinite_update")


def train(args):
    torch.set_num_threads(1)
    spec = Spec(args.initialization, args.task, args.heading_penalty, args.workers, args.rollout_steps)
    args.output.mkdir(parents=True, exist_ok=False)
    model = norm = monitor = None
    handlers = {}
    try:
        model, norm, rng = build(args.checkpoint, spec, args.seed, args.resume)
        state = model.command_walk_state
        monitor = Monitor(args.output, model.num_timesteps, args.deadline, spec, args.cases, args.diagnostic)
        monitor.init_callback(model)
        model._diagnostics_path = args.output / "updates.jsonl"
        atomic_json(
            args.output / "provenance.json",
            dict(
                spec=asdict(spec),
                source=str(args.checkpoint.resolve()),
                source_sha256=sha256(args.checkpoint),
                normalization_sha256=sha256(_stats_path(args.checkpoint)),
                seed=args.seed,
                rng_hash=rng,
                resume=args.resume,
                deadline=args.deadline,
            ),
        )
        handlers = {
            sig: signal.signal(sig, lambda *_: monitor.stop("interrupted")) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        initial, _ = monitor.probe("initial", full=True)
        actor_budget = new_budget(args.steps, spec.workers * spec.rollout_steps)
        target = state["warmup_until"] + actor_budget
        model.learn(total_timesteps=max(0, target - model.num_timesteps), reset_num_timesteps=False, callback=monitor)
        for p in actor_parameters(model):
            p.requires_grad_(True)
        final, _ = monitor.probe("final", full=True)
        atomic_json(
            args.output / "report.json",
            dict(initial=initial, final=final, stop_reason=monitor.stop_reason, candidates=state["candidates"]),
        )
    except TimeoutError:
        pass
    except Exception as error:
        atomic_json(args.output / "error.json", dict(error=repr(error)))
        if monitor:
            monitor.stop("error")
        raise
    finally:
        if model is not None:
            for p in actor_parameters(model):
                p.requires_grad_(True)
            path = save_bundle(model, norm, args.output, "resume")
            reason = monitor.stop_reason if monitor else "error"
            atomic_json(
                args.output / "status.json",
                dict(
                    state="completed" if reason == "step_budget" else "stopped",
                    stop_reason=reason,
                    checkpoint=str(path.resolve()),
                    heartbeat=time.time(),
                    steps=model.num_timesteps,
                    actor_steps=max(0, model.num_timesteps - model.command_walk_state["warmup_until"]),
                    calibration_steps=min(model.num_timesteps, model.command_walk_state["warmup_until"])
                    - model.command_walk_state["training_start"],
                ),
            )
        else:
            atomic_json(
                args.output / "status.json",
                dict(state="failed", stop_reason="initialization_error", heartbeat=time.time()),
            )
        if norm:
            norm.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--initialization", choices=INITIALIZATIONS, default="preserved")
    p.add_argument("--task", choices=TARGETS, default="all")
    p.add_argument("--heading-penalty", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=768)
    p.add_argument("--cases", type=int, default=50)
    args = p.parse_args()
    if args.steps < 0 or args.cases < 1 or not np.isfinite(args.deadline):
        p.error("Invalid budget")
    train(args)
