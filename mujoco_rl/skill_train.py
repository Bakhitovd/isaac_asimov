"""Train the five MuJoCo skills in gated, resumable Loudbox stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .skill_env import JOINT_NAMES, RECOVERY_TILT_RANGES, SKILLS, SkillEnv
from .skill_eval import evaluate
from .recovery_ppo import RecoveryPPO


ROOT = Path(__file__).resolve().parents[1]
WALK_CHECKPOINT = ROOT / "mujoco_rl" / "checkpoints" / "full_body_v1" / "policy.zip"
STAGE_ORDER = ("nav", "squat", "recover", "run", "jump")


def _factory(skill: str, seed: int):
    def create():
        env = Monitor(SkillEnv(skill, randomize=True,
                               curriculum_level=0.0 if skill == "nav" else 1.0,
                               recovery_level=0 if skill == "recover" else None,
                               recovery_rehearsal=skill == "recover"))
        env.reset(seed=seed)
        return env
    return create


def vector_env(skill: str, workers: int, seed: int):
    factories = [_factory(skill, seed + index) for index in range(workers)]
    return SubprocVecEnv(factories, start_method="forkserver") if workers > 1 else DummyVecEnv(factories)


def _stats_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")


def _recovery_fresh_seed_start(run_dir: Path) -> int:
    """Separate confirmation seeds across runs, reproducibly from their paths."""
    digest = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()
    return 1_000_000 + int(digest[:8], 16) % 1_000_000_000


def _recovery_level_from_checkpoint(checkpoint: Path) -> int:
    state_path = checkpoint.with_name(checkpoint.stem + "_training.json")
    if state_path.is_file():
        return int(json.loads(state_path.read_text(encoding="utf-8"))["recovery_level"])
    if not checkpoint.stem.startswith("checkpoint_"):
        return 0
    checkpoint_step = int(checkpoint.stem.removeprefix("checkpoint_"))
    evaluations = checkpoint.parent / "evaluations.jsonl"
    if evaluations.is_file():
        for line in reversed(evaluations.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            if record["training_steps"] == checkpoint_step:
                return int(record.get("next_recovery_level", 0))
    return 0


def _warm_start(model: PPO, normalizer: VecNormalize, checkpoint: Path,
                neutral_action_head: bool = False) -> None:
    source = PPO.load(str(checkpoint), device="cpu")
    old_state = source.policy.state_dict()
    new_state = model.policy.state_dict()
    for key, target in new_state.items():
        value = old_state.get(key)
        if value is None:
            continue
        if value.shape == target.shape:
            new_state[key] = value.clone()
        elif key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight") and value.shape[0] == target.shape[0] and value.shape[1] < target.shape[1]:
            padded = torch.zeros_like(target)
            padded[:, :value.shape[1]] = value
            new_state[key] = padded
        else:
            raise ValueError(f"Cannot transfer parameter {key}: {tuple(value.shape)} -> {tuple(target.shape)}")
    model.policy.load_state_dict(new_state)
    if neutral_action_head:
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
            model.policy.log_std.fill_(-3.0)
    stats = _stats_path(checkpoint)
    if not stats.is_file():
        raise FileNotFoundError(f"Missing observation statistics: {stats}")
    with stats.open("rb") as handle:
        old_norm = pickle.load(handle)
    old_mean = old_norm.obs_rms.mean
    old_var = old_norm.obs_rms.var
    if len(old_mean) > len(normalizer.obs_rms.mean):
        raise ValueError("Source observation vector is larger than the new environment")
    normalizer.obs_rms.mean[:len(old_mean)] = old_mean
    normalizer.obs_rms.var[:len(old_var)] = old_var
    normalizer.obs_rms.count = 1_000.0 if neutral_action_head else old_norm.obs_rms.count
    print(f"[warm-start] transferred {len(old_mean)} observation channels from {checkpoint}", flush=True)


def _normalize_recovery_rewards(model: PPO, normalizer: VecNormalize) -> bool:
    """Migrate unnormalized recovery value targets without changing the actor."""
    migrated = not normalizer.norm_reward
    normalizer.norm_reward = True
    if migrated:
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
        for parameter in model.policy.value_net.parameters():
            model.policy.optimizer.state.pop(parameter, None)
        print("[recover] normalized reward targets; reset critic output and its optimizer state", flush=True)
    return migrated


def _recovery_focus_ranges(probe: dict) -> dict[str, tuple[float, float]]:
    """Practice around the observed success boundary in weak directions."""
    nominal_lo, nominal_hi = RECOVERY_TILT_RANGES[probe["recovery_level"]]
    ranges = {}
    for pose, rate in probe.get("group_rates", {}).items():
        if rate >= 0.75:
            continue
        episodes = [item for item in probe["episodes"] if item.get("recovery_pose") == pose]
        passed = [item["recovery_tilt_deg"] for item in episodes if item["passed"]]
        failed = [item["recovery_tilt_deg"] for item in episodes if not item["passed"]]
        if passed and failed:
            if min(failed) < max(passed) and max(failed) - min(failed) > 2.0:
                # Mixed failures are not a single tilt boundary. Keep both
                # tails instead of concentrating on a successful middle band.
                ranges[pose] = (max(nominal_lo, min(failed) - 0.75), min(nominal_hi, max(failed) + 0.75))
            else:
                boundary = 0.5 * (max(passed) + min(failed))
                ranges[pose] = (max(nominal_lo, boundary - 0.75), min(nominal_hi, boundary + 0.75))
    return ranges


class GateCallback(BaseCallback):
    def __init__(self, skill: str, run_dir: Path, deadline: float, interval: int,
                 recovery_level: int = 0, recovery_critic_warmup_steps: int = 0,
                 recovery_stress_seed_start: int | None = None,
                 recovery_stall_steps: int = 1_000_000):
        super().__init__()
        self.skill = skill
        self.run_dir = run_dir
        self.deadline = deadline
        self.interval = interval
        self.last_eval = 0
        self.best_score = -float("inf")
        self.accepted = False
        self.holdout_attempt = 0
        self.curriculum_holdout_attempt = 0
        self.curriculum_holdout_seed_start = _recovery_fresh_seed_start(run_dir)
        self.recovery_level = recovery_level
        self.recovery_start_step = 0
        self.recovery_lr_reduced = False
        self.high_kl_windows = 0
        self.kl_samples: list[float] = []
        self.abort_reason: str | None = None
        self.probe_pass_streak = 0
        self.best_probe_rate = 0.0
        self.best_recovery_quality = (-1, -1.0, -1.0)
        self.regression_streak = 0
        self.retention_failure_streak = 0
        self.last_level_progress_step = 0
        self.last_probe_step = 0
        self.last_probe_record: dict = {}
        self.recovery_critic_warmup_steps = recovery_critic_warmup_steps
        self.actor_warmup_active = False
        self.actor_lr_schedule = None
        self.recovery_stress_seed_start = recovery_stress_seed_start
        self.recovery_stall_steps = recovery_stall_steps

    def _on_training_start(self) -> None:
        self.last_eval = self.model.num_timesteps
        self.recovery_start_step = self.model.num_timesteps
        self.last_level_progress_step = self.model.num_timesteps
        self.last_probe_step = self.model.num_timesteps
        previous = self.run_dir / "curriculum.jsonl"
        if previous.exists():
            with previous.open(encoding="utf-8") as handle:
                self.curriculum_holdout_attempt = sum("curriculum_fresh_evaluation" in json.loads(line)
                                                       for line in handle if line.strip())

    def _on_rollout_end(self) -> None:
        if self.skill == "recover":
            values = self.model.logger.name_to_value
            kls = [values[key] for key in ("train/approx_kl", "train/ppo_kl", "train/post_imitation_kl")
                   if key in values]
            if kls:
                self.kl_samples.append(max(float(kl) for kl in kls))

    def _on_rollout_start(self) -> None:
        if self.skill != "recover" or not self.recovery_critic_warmup_steps:
            return
        warming = self.num_timesteps - self.recovery_start_step < self.recovery_critic_warmup_steps
        parameters = (list(self.model.policy.mlp_extractor.policy_net.parameters())
                      + list(self.model.policy.action_net.parameters()) + [self.model.policy.log_std])
        for parameter in parameters:
            parameter.requires_grad_(not warming)
        if warming != self.actor_warmup_active:
            if warming:
                self.actor_lr_schedule = self.model.lr_schedule
                self.model.lr_schedule = get_schedule_fn(1e-4)
            elif self.actor_lr_schedule is not None:
                self.model.lr_schedule = self.actor_lr_schedule
            for group in self.model.policy.optimizer.param_groups:
                group["lr"] = float(self.model.lr_schedule(1.0))
            print(f"[recover] critic warm-up {'started; actor frozen' if warming else 'complete; actor updates enabled'}", flush=True)
        self.actor_warmup_active = warming

    def _probe_recovery(self, normalizer: VecNormalize) -> None:
        record = {"training_steps": self.num_timesteps}
        probe = evaluate(self.model, normalizer, self.skill, seed_start=20_000, seeds=48,
                         recovery_level=self.recovery_level)
        record["recovery_probe"] = probe
        record["recovery_level"] = self.recovery_level
        validation = evaluate(self.model, normalizer, self.skill, seed_start=30_000, seeds=48,
                              recovery_level=self.recovery_level)
        record["recovery_current_validation"] = validation
        panels = [probe, validation]
        if self.recovery_stress_seed_start is not None:
            stress = evaluate(self.model, normalizer, self.skill, seed_start=self.recovery_stress_seed_start,
                              seeds=50, recovery_level=self.recovery_level)
            record["recovery_stress_validation"] = stress
            panels.append(stress)
        pending = getattr(self.model, "recovery_failed_fresh_panel", None)
        if pending is not None and pending["recovery_level"] == self.recovery_level:
            retry = evaluate(self.model, normalizer, self.skill, seed_start=pending["seed_start"],
                             seeds=50, recovery_level=self.recovery_level)
            record["recovery_retry_validation"] = retry
            panels.append(retry)
        passed_panels = all(panel["criterion_met"] for panel in panels)
        retained = True
        if self.recovery_level > 0:
            retention = evaluate(self.model, normalizer, self.skill, seed_start=30_000, seeds=48,
                                 recovery_level=self.recovery_level - 1)
            record["recovery_retention"] = retention
            retained = bool(retention["criterion_met"])
            if (self.recovery_stress_seed_start is not None
                    and self.recovery_level - 1 in getattr(self.model, "recovery_stress_validated_levels", [])):
                stress_retention = evaluate(self.model, normalizer, self.skill,
                                            seed_start=self.recovery_stress_seed_start, seeds=50,
                                            recovery_level=self.recovery_level - 1)
                record["recovery_stress_retention"] = stress_retention
                retained = retained and bool(stress_retention["criterion_met"])
            retry_baseline = getattr(self.model, "recovery_validated_retry_panels", {}).get(
                str(self.recovery_level - 1))
            if retry_baseline is not None:
                retry_retention = evaluate(self.model, normalizer, self.skill,
                                           seed_start=retry_baseline["seed_start"], seeds=50,
                                           recovery_level=self.recovery_level - 1)
                record["recovery_retry_retention"] = retry_retention
                retained = retained and bool(retry_retention["criterion_met"])
        if passed_panels and retained and self.recovery_stress_seed_start is not None:
            validated = set(getattr(self.model, "recovery_stress_validated_levels", []))
            self.model.recovery_stress_validated_levels = sorted(validated | {self.recovery_level})
        self.retention_failure_streak = 0 if retained else self.retention_failure_streak + 1
        self.probe_pass_streak = (self.probe_pass_streak + 1
                                  if passed_panels and retained else 0)
        advance = self.probe_pass_streak >= 2 and self.recovery_level < len(RECOVERY_TILT_RANGES) - 1
        if advance:
            fresh = evaluate(self.model, normalizer, self.skill,
                             seed_start=self.curriculum_holdout_seed_start + 50 * self.curriculum_holdout_attempt,
                             seeds=50, recovery_level=self.recovery_level)
            self.curriculum_holdout_attempt += 1
            record["curriculum_fresh_evaluation"] = fresh
            advance = bool(fresh["criterion_met"])
            if not advance:
                self.probe_pass_streak = 0
                self.model.recovery_failed_fresh_panel = {"seed_start": fresh["seed_start"],
                                                          "recovery_level": self.recovery_level}
                panels.append(fresh)
            else:
                if pending is not None and pending["recovery_level"] == self.recovery_level:
                    baselines = dict(getattr(self.model, "recovery_validated_retry_panels", {}))
                    baselines[str(self.recovery_level)] = pending
                    self.model.recovery_validated_retry_panels = baselines
                self.model.recovery_failed_fresh_panel = None
        weakest_rates = {pose: min(panel.get("group_rates", {}).get(pose, 1.0) for panel in panels)
                         for pose in ("front", "back", "left", "right")}
        weights = [0.10 + 1.0 - rate for rate in weakest_rates.values()]
        weights = [value / sum(weights) for value in weights]
        record["next_training_pose_weights"] = weights
        normalizer.venv.env_method("set_recovery_pose_weights", weights)
        focus_ranges = _recovery_focus_ranges({**probe, "group_rates": weakest_rates,
                                              "episodes": [episode for panel in panels for episode in panel["episodes"]]})
        record["next_training_focus_ranges"] = focus_ranges
        normalizer.venv.env_method("set_recovery_focus_ranges", focus_ranges)
        next_level = self.recovery_level + int(advance)
        record["next_recovery_level"] = next_level
        self.regression_streak = (self.regression_streak + 1
                                  if self.best_probe_rate - probe["pass_rate"] >= 0.20 else 0)
        self.best_probe_rate = max(self.best_probe_rate, probe["pass_rate"])
        quality = (int(all(panel["criterion_met"] for panel in panels) and retained), min(weakest_rates.values()),
                   min(panel["pass_rate"] for panel in panels))
        if quality > self.best_recovery_quality:
            self.best_recovery_quality = quality
            self.model.save(str(self.run_dir / f"curriculum_best_{self.recovery_level}"))
            normalizer.save(str(self.run_dir / f"curriculum_best_{self.recovery_level}_vecnormalize.pkl"))
            (self.run_dir / f"curriculum_best_{self.recovery_level}_training.json").write_text(
                json.dumps({"recovery_level": self.recovery_level}) + "\n", encoding="utf-8")
        if self.regression_streak >= 2:
            self.abort_reason = "recovery probe regressed by at least 20 points twice; restore curriculum best"
        if self.retention_failure_streak >= 3:
            self.abort_reason = self.abort_reason or "previous recovery level failed retention three times; restore curriculum best"
        if advance:
            self.last_level_progress_step = self.num_timesteps
        elif self.num_timesteps - self.last_level_progress_step >= self.recovery_stall_steps:
            self.abort_reason = f"recovery curriculum stalled for {self.recovery_stall_steps:,} steps"
        high_kl_fraction = (sum(kl > 0.03 for kl in self.kl_samples) / len(self.kl_samples)
                            if self.kl_samples else 0.0)
        record["high_kl_fraction"] = high_kl_fraction
        self.kl_samples.clear()
        if high_kl_fraction > 0.20:
            if not self.recovery_lr_reduced:
                self.recovery_lr_reduced = True
                reduced_lr = min(1e-5, float(self.model.lr_schedule(1.0)) / 3.0)
                self.model.learning_rate = reduced_lr
                self.model.lr_schedule = get_schedule_fn(reduced_lr)
                for group in self.model.policy.optimizer.param_groups:
                    group["lr"] = reduced_lr
                print(f"[recover] high KL; reducing learning rate to {reduced_lr:.2g}", flush=True)
            else:
                self.high_kl_windows += 1
                if self.high_kl_windows >= 2 and self.num_timesteps - self.recovery_start_step >= 500_000:
                    self.abort_reason = "KL remained high after learning-rate reduction"
        elif self.recovery_lr_reduced:
            self.high_kl_windows = 0
        if self.num_timesteps - self.recovery_start_step >= 2_000_000 and next_level == 0:
            self.abort_reason = "recovery curriculum did not advance by 2 million steps"
        record["learning_rate"] = float(self.model.lr_schedule(1.0))
        if self.abort_reason:
            record["abort_reason"] = self.abort_reason
        self.last_probe_step = self.num_timesteps
        self.last_probe_record = record
        with (self.run_dir / "curriculum.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if next_level != self.recovery_level:
            self.recovery_level = next_level
            self.probe_pass_streak = 0
            self.best_probe_rate = 0.0
            self.best_recovery_quality = (-1, -1.0, -1.0)
            self.regression_streak = 0
            self.retention_failure_streak = 0
            normalizer.venv.env_method("set_recovery_level", next_level)
        retained_rate = record.get("recovery_retention", {}).get("pass_rate", 1.0)
        direction_rates = ", ".join(f"{pose}:{rate:.0%}" for pose, rate in probe.get("group_rates", {}).items())
        validation_rates = ", ".join(f"{pose}:{rate:.0%}" for pose, rate in validation.get("group_rates", {}).items())
        fresh_note = (f", fresh={record['curriculum_fresh_evaluation']['pass_rate']:.1%}"
                      if "curriculum_fresh_evaluation" in record else "")
        stress_note = (f", stress={stress['pass_rate']:.1%}, stress_passed={stress['criterion_met']}, "
                       f"stress_directions={stress.get('group_rates', {})}"
                       if self.recovery_stress_seed_start is not None else "")
        retry_note = (f", retry={retry['pass_rate']:.1%}, retry_passed={retry['criterion_met']}, "
                      f"retry_directions={retry.get('group_rates', {})}"
                      if "recovery_retry_validation" in record else "")
        retry_retention_note = (f", retry_retention={retry_retention['pass_rate']:.1%}, "
                                f"retry_retention_passed={retry_retention['criterion_met']}"
                                if "recovery_retry_retention" in record else "")
        print(f"[probe] recover {self.num_timesteps:,} steps: level={record['recovery_level']}, "
              f"success={probe['pass_rate']:.1%}, retention={retained_rate:.1%}, "
              f"retention_passed={retained}, next_level={next_level}, directions=[{direction_rates}], "
              f"validation={validation['pass_rate']:.1%}, validation_passed={validation['criterion_met']}, "
              f"validation_directions=[{validation_rates}]{stress_note}{retry_note}"
              f"{retry_retention_note}{fresh_note}", flush=True)

    def _on_step(self) -> bool:
        if self.skill == "recover":
            episodes = [{"training_steps": self.num_timesteps, "worker": index,
                         **{key: info.get(key) for key in (
                             "episode", "episode_recovery_level", "recovery_pose", "recovery_tilt_deg", "success",
                             "termination_reason", "recovery_reward_totals", "seconds", "stable_steps")}}
                        for index, info in enumerate(self.locals.get("infos", [])) if "episode" in info]
            if episodes:
                with (self.run_dir / "training_episodes.jsonl").open("a", encoding="utf-8") as handle:
                    for episode in episodes:
                        handle.write(json.dumps(episode) + "\n")
        if time.monotonic() >= self.deadline:
            return False
        if self.skill == "recover" and self.num_timesteps - self.last_probe_step >= min(self.interval, 25_000):
            normalizer = self.model.get_env()
            assert isinstance(normalizer, VecNormalize)
            self._probe_recovery(normalizer)
            if self.abort_reason:
                print(f"[recover] stopping: {self.abort_reason}", flush=True)
                return False
        if self.num_timesteps - self.last_eval < self.interval:
            return True
        self.last_eval = self.num_timesteps
        normalizer = self.model.get_env()
        assert isinstance(normalizer, VecNormalize)
        result = evaluate(self.model, normalizer, self.skill)
        record = {"training_steps": self.num_timesteps, "development": result}
        if self.skill == "recover":
            record.update({key: value for key, value in self.last_probe_record.items()
                           if key != "training_steps"})
            record["probe_steps"] = self.last_probe_step
            probe = record["recovery_probe"]
            next_level = self.recovery_level
        checkpoint_name = f"checkpoint_{self.num_timesteps}"
        self.model.save(str(self.run_dir / checkpoint_name))
        normalizer.save(str(self.run_dir / f"{checkpoint_name}_vecnormalize.pkl"))
        if self.skill == "recover":
            (self.run_dir / f"{checkpoint_name}_training.json").write_text(
                json.dumps({"recovery_level": next_level}) + "\n", encoding="utf-8")
        episodes = result["episodes"]
        mean_duration = float(np.mean([item["seconds"] for item in episodes]))
        if self.skill == "nav":
            near_count = sum(item["distance_to_goal_m"] < 0.25 for item in episodes)
            mean_goal_distance = float(np.mean([item["distance_to_goal_m"] for item in episodes]))
            score = 10.0 * result["pass_rate"] + 0.02 * near_count + 0.01 * mean_duration - 0.2 * mean_goal_distance
        elif self.skill == "recover":
            score = (10.0 * result["pass_rate"] + 2.0 * probe["pass_rate"]
                     + 0.5 * self.recovery_level)
        else:
            score = 10.0 * result["pass_rate"] + 0.001 * mean_duration
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.run_dir / "best"))
            normalizer.save(str(self.run_dir / "best_vecnormalize.pkl"))
        if result["criterion_met"]:
            holdout_seed = 10_051 + 50 * self.holdout_attempt
            self.holdout_attempt += 1
            holdout = evaluate(self.model, normalizer, self.skill, seed_start=holdout_seed)
            record["holdout"] = holdout
            retained = True
            if self.skill == "run":
                nav = evaluate(self.model, normalizer, "nav", seed_start=holdout_seed + 50)
                record["navigation_retention"] = nav
                retained = bool(nav["criterion_met"])
            self.accepted = bool(holdout["criterion_met"] and retained)
            if self.accepted:
                self.model.save(str(self.run_dir / "accepted"))
                normalizer.save(str(self.run_dir / "accepted_vecnormalize.pkl"))
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.logger.record("eval/pass_rate", result["pass_rate"])
        if self.skill == "nav":
            level = min(1.0, self.num_timesteps / 1_500_000)
            normalizer.venv.env_method("set_curriculum_level", level)
            self.logger.record("train/curriculum_level", level)
        elif self.skill == "recover":
            self.logger.record("train/recovery_level", self.recovery_level)
        print(f"[eval] {self.skill} {self.num_timesteps:,} steps: dev={result['pass_rate']:.1%}, "
              f"holdout={record.get('holdout', {}).get('pass_rate', 'pending')}, "
              f"accepted={self.accepted}"
              + (f", curriculum={self.recovery_level}, probe={probe['pass_rate']:.1%}"
                 if self.skill == "recover" else ""), flush=True)
        if self.abort_reason:
            print(f"[recover] stopping: {self.abort_reason}", flush=True)
        return not self.accepted and time.monotonic() < self.deadline and not self.abort_reason


def train_stage(skill: str, run_dir: Path, workers: int, seed: int, hours: float,
                warm_start: Path | None, resume: Path | None, eval_interval: int,
                max_steps: int, recovery_lr: float = 3e-5,
                recovery_leg_std: float | None = None,
                recovery_critic_warmup_steps: int = 0,
                recovery_demonstrations: Path | None = None,
                recovery_stress_seed_start: int | None = None,
                recovery_imitation_weight: float = 100.0,
                recovery_imitation_updates: int = 16,
                rollout_steps: int = 256,
                recovery_stall_steps: int = 1_000_000) -> tuple[bool, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    base = vector_env("run_train" if skill == "run" else skill, workers, seed)
    if resume:
        freeze_statistics = False
        if skill == "recover":
            source_config = json.loads((resume.parent / "config.json").read_text(encoding="utf-8"))
            if source_config.get("recovery_version") not in {4, 5, 6, 7, 8, 9, 10, 11}:
                raise ValueError("Recovery action semantics changed; warm-start a new run from the accepted squat policy")
            freeze_statistics = bool(source_config.get("bootstrap") or
                                     source_config.get("recovery_statistics_frozen"))
            if recovery_stress_seed_start is None:
                recovery_stress_seed_start = source_config.get("recovery_stress_seed_start")
        env = VecNormalize.load(str(_stats_path(resume)), base)
        env.training = not freeze_statistics
        algorithm = RecoveryPPO if skill == "recover" else PPO
        model = algorithm.load(str(resume), env=env, device="cpu", seed=seed)
        model.n_steps = rollout_steps
        model.rollout_buffer = model.rollout_buffer_class(
            rollout_steps, model.observation_space, model.action_space, device=model.device,
            gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=workers,
            **model.rollout_buffer_kwargs)
        if skill == "recover" and source_config.get("recovery_stress_seed_start") != recovery_stress_seed_start:
            model.recovery_stress_validated_levels = []
        if skill == "recover":
            _normalize_recovery_rewards(model, env)
        if skill == "nav":
            env.venv.env_method("set_curriculum_level", min(1.0, model.num_timesteps / 1_500_000))
        recovery_level = _recovery_level_from_checkpoint(resume) if skill == "recover" else 0
        if skill == "recover":
            env.venv.env_method("set_recovery_level", recovery_level)
        learning_rate = 5e-5 if skill == "nav" else recovery_lr if skill == "recover" else 1e-4 if skill == "run" else 3e-4
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        model.clip_range = get_schedule_fn(0.10 if skill == "recover" else 0.15)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        recovery_level = 0
        env = VecNormalize(base, norm_obs=True, norm_reward=skill == "recover", clip_obs=10.0)
        learning_rate = 5e-5 if skill == "nav" else recovery_lr if skill == "recover" else 1e-4 if skill == "run" else 3e-4
        algorithm = RecoveryPPO if skill == "recover" else PPO
        model = algorithm("MlpPolicy", env, device="cpu", seed=seed, verbose=1,
                    n_steps=rollout_steps, batch_size=512, n_epochs=4, learning_rate=learning_rate,
                    gamma=0.995 if skill == "recover" else 0.99,
                    gae_lambda=0.95, clip_range=0.10 if skill == "recover" else 0.15,
                    ent_coef=0.0 if skill == "recover" else 0.001 if skill in {"nav", "run"} else 0.003,
                    target_kl=0.015, max_grad_norm=1.0,
                    policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                                   "activation_fn": torch.nn.ELU, "log_std_init": -1.0})
        if warm_start:
            _warm_start(model, env, warm_start, neutral_action_head=skill == "recover")
    if skill == "recover" and recovery_leg_std is not None:
        with torch.no_grad():
            model.policy.log_std[:12].fill_(np.log(recovery_leg_std))
        model.policy.optimizer.state.pop(model.policy.log_std, None)
        print(f"[recover] initial leg exploration std={recovery_leg_std}", flush=True)
    if skill == "recover":
        sensor_memory = bool(getattr(model, "recovery_sensor_memory", False))
        env.venv.env_method("set_recovery_sensor_memory", sensor_memory)
        phase_features = bool(getattr(model, "recovery_phase_features", False))
        env.venv.env_method("set_recovery_phase_features", phase_features)
        if recovery_demonstrations is None and resume and source_config.get("recovery_demonstrations"):
            recovery_demonstrations = Path(source_config["recovery_demonstrations"])
        if recovery_demonstrations is not None:
            env.training = False
            model.load_demonstrations(recovery_demonstrations, env, seed=seed,
                                      weight=recovery_imitation_weight, updates=recovery_imitation_updates)
            print(f"[recover] bounded imitation updates from {recovery_demonstrations}", flush=True)
        model._diagnostics_path = run_dir / "updates.jsonl"
    config = {"skill": skill, "workers": workers, "seed": seed,
              "torch_threads": torch.get_num_threads(),
              "rollout_steps": rollout_steps, "rollout_batch_size": rollout_steps * workers,
              "hours": hours, "max_steps": max_steps, "eval_interval": eval_interval,
              "learning_rate": learning_rate,
              "navigation_curriculum_steps": 1_500_000 if skill == "nav" else None,
              "initial_recovery_level": recovery_level if skill == "recover" else None,
              "recovery_tilt_ranges_deg": RECOVERY_TILT_RANGES if skill == "recover" else None,
              "neutral_recovery_action_head": skill == "recover" and warm_start is not None,
              "recovery_version": (11 if phase_features else 10 if sensor_memory else 9) if skill == "recover" else None,
              "recovery_sensor_memory": sensor_memory if skill == "recover" else False,
              "recovery_phase_features": phase_features if skill == "recover" else False,
              "recovery_reset_controller": "hold_initial_pose" if skill == "recover" else None,
              "normalize_reward": env.norm_reward,
              "recovery_statistics_frozen": skill == "recover" and not env.training,
              "recovery_critic_warmup_steps": recovery_critic_warmup_steps,
              "recovery_critic_warmup_lr": 1e-4 if recovery_critic_warmup_steps else None,
              "recovery_leg_std_override": recovery_leg_std,
              "recovery_adaptive_pose_sampling": skill == "recover",
              "recovery_boundary_sampling_fraction": 0.80 if skill == "recover" else None,
              "recovery_stall_steps": recovery_stall_steps if skill == "recover" else None,
              "recovery_success_bonus": 500.0 if skill == "recover" else None,
              "recovery_reset_support_fade_deg": [15.0, 45.0] if skill == "recover" else None,
              "recovery_rehearsal_fraction": 0.30 if skill == "recover" else None,
              "recovery_previous_level_rehearsal_share": 0.60 if skill == "recover" else None,
              "recovery_curriculum_fresh_seeds": 50 if skill == "recover" else None,
              "recovery_current_validation_seeds": [30_000, 30_047] if skill == "recover" else None,
              "recovery_stress_seed_start": recovery_stress_seed_start if skill == "recover" else None,
              "recovery_curriculum_fresh_seed_start": _recovery_fresh_seed_start(run_dir) if skill == "recover" else None,
              "recovery_algorithm": ("PPO_with_imitation_updates" if recovery_demonstrations
                                     and recovery_imitation_weight and recovery_imitation_updates else "PPO"),
              "recovery_demonstrations": str(recovery_demonstrations.resolve()) if recovery_demonstrations else None,
              "recovery_imitation_weight": model.imitation_weight if recovery_demonstrations else None,
              "recovery_imitation_updates": model.imitation_updates if recovery_demonstrations else None,
              "recovery_imitation_targets": model.imitation_target_kind if recovery_demonstrations else None,
              "warm_start": str(warm_start) if warm_start else None,
              "resume": str(resume) if resume else None,
              "development_seeds": [10_001, 10_050], "first_holdout_seeds": [10_051, 10_100],
              "source_sha256": {}}
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in ("skill_env.py", "skill_train.py", "skill_eval.py",
                     "skill_supervisor.py", "full_body_env.py", "recovery_bootstrap.py", "recovery_search.py",
                     "recovery_ppo.py", "recovery_retention.py", "recovery_compare.py", "recovery_diagnostics.py"):
        source = Path(__file__).with_name(filename)
        shutil.copy2(source, snapshot / filename)
        config["source_sha256"][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    callback = GateCallback(skill, run_dir, time.monotonic() + hours * 3600, eval_interval,
                            recovery_level=recovery_level,
                            recovery_critic_warmup_steps=recovery_critic_warmup_steps,
                            recovery_stress_seed_start=recovery_stress_seed_start,
                            recovery_stall_steps=recovery_stall_steps)
    try:
        remaining = max_steps - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback,
                        reset_num_timesteps=resume is None)
        model.save(str(run_dir / "final"))
        env.save(str(run_dir / "final_vecnormalize.pkl"))
        if skill == "recover":
            (run_dir / "final_training.json").write_text(
                json.dumps({"recovery_level": callback.recovery_level}) + "\n", encoding="utf-8")
        print(f"[stage] {skill}: accepted={callback.accepted}; steps={model.num_timesteps:,}", flush=True)
        candidate = ("accepted.zip" if callback.accepted else
                     "best.zip" if (run_dir / "best.zip").is_file() else "final.zip")
        return callback.accepted, run_dir / candidate
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pipeline", *STAGE_ORDER), default="pipeline")
    parser.add_argument("--run-dir", type=Path, default=Path("logs/mujoco_rl/multi_skill_v1"))
    parser.add_argument("--hours-per-stage", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100_000_000)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-stage", choices=STAGE_ORDER, default="nav")
    parser.add_argument("--prior-run-dir", type=Path)
    parser.add_argument("--recovery-lr", type=float, default=3e-5)
    parser.add_argument("--recovery-leg-std", type=float)
    parser.add_argument("--recovery-critic-warmup-steps", type=int, default=0)
    parser.add_argument("--recovery-demonstrations", type=Path,
                        help="Successful bootstrap .npz data to retain through bounded imitation updates")
    parser.add_argument("--recovery-stress-seed-start", type=int,
                        help="Additional 50-episode development panel; participates in advancement and pose sampling")
    parser.add_argument("--recovery-imitation-weight", type=float, default=100.0)
    parser.add_argument("--recovery-stall-steps", type=int, default=1_000_000,
                        help="Stop if the recovery curriculum does not advance within this many additional steps")
    parser.add_argument("--recovery-imitation-updates", type=int, default=16,
                        help="Additional imitation updates per PPO cycle; zero disables imitation")
    parser.add_argument("--rollout-steps", type=int, default=256,
                        help="Steps per worker per PPO batch; keep workers * rollout-steps fixed for comparisons")
    args = parser.parse_args()
    if (args.hours_per_stage <= 0 or args.workers < 1 or args.max_steps < 1 or args.eval_interval < 1
            or args.recovery_lr <= 0 or args.torch_threads < 1):
        parser.error("Invalid training budget, workers, or evaluation interval")
    if args.recovery_leg_std is not None and not 0.0 < args.recovery_leg_std <= 0.30:
        parser.error("--recovery-leg-std must be in (0, 0.30]")
    if args.recovery_critic_warmup_steps < 0:
        parser.error("--recovery-critic-warmup-steps must be nonnegative")
    if args.recovery_stress_seed_start is not None and args.recovery_stress_seed_start < 0:
        parser.error("--recovery-stress-seed-start must be nonnegative")
    if (not np.isfinite(args.recovery_imitation_weight) or args.recovery_imitation_weight < 0
            or args.recovery_imitation_updates < 0 or args.rollout_steps < 1 or args.recovery_stall_steps < 1):
        parser.error("Invalid imitation settings or rollout length")
    if args.stage == "pipeline" and args.warm_start:
        parser.error("Use --stage for a custom warm start")
    if args.stage != "pipeline" and (args.resume_stage != "nav" or args.prior_run_dir):
        parser.error("--resume-stage and --prior-run-dir require --stage pipeline")
    torch.set_num_threads(args.torch_threads)
    if args.stage != "pipeline":
        train_stage(args.stage, args.run_dir, args.workers, args.seed,
                    args.hours_per_stage, args.warm_start or (WALK_CHECKPOINT if not args.resume else None),
                    args.resume, args.eval_interval, args.max_steps, args.recovery_lr, args.recovery_leg_std,
                    args.recovery_critic_warmup_steps, args.recovery_demonstrations, args.recovery_stress_seed_start,
                    args.recovery_imitation_weight, args.recovery_imitation_updates, args.rollout_steps,
                    args.recovery_stall_steps)
        return
    start_index = STAGE_ORDER.index(args.resume_stage)
    if start_index and not args.prior_run_dir:
        parser.error("Starting a later pipeline stage requires --prior-run-dir")
    accepted = {}
    for prior_skill in STAGE_ORDER[:start_index]:
        checkpoint = args.prior_run_dir / prior_skill / "accepted.zip"
        if not checkpoint.is_file() or not _stats_path(checkpoint).is_file():
            parser.error(f"Missing accepted prior-stage checkpoint or statistics: {checkpoint}")
        accepted[prior_skill] = checkpoint
    for index in range(start_index, len(STAGE_ORDER)):
        skill = STAGE_ORDER[index]
        source = (WALK_CHECKPOINT if skill == "nav" else
                  accepted["nav"] if skill in {"squat", "run"} else
                  accepted["squat"] if skill in {"recover", "jump"} else None)
        stage_resume = args.resume if index == start_index else None
        passed, checkpoint = train_stage(skill, args.run_dir / skill,
                                         args.workers, args.seed + index, args.hours_per_stage,
                                         None if stage_resume else source, stage_resume,
                                         args.eval_interval, args.max_steps, args.recovery_lr, args.recovery_leg_std,
                                         args.recovery_critic_warmup_steps, args.recovery_demonstrations,
                                         args.recovery_stress_seed_start, args.recovery_imitation_weight,
                                         args.recovery_imitation_updates, args.rollout_steps, args.recovery_stall_steps)
        if not passed:
            print(f"[pipeline] stopped at {skill}; gate not met within budget", flush=True)
            break
        accepted[skill] = checkpoint
    else:
        print("[pipeline] all five skill gates passed", flush=True)


if __name__ == "__main__":
    main()
