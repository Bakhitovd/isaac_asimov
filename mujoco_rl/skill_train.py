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

from .skill_env import JOINT_NAMES, SKILLS, SkillEnv
from .skill_eval import evaluate


ROOT = Path(__file__).resolve().parents[1]
WALK_CHECKPOINT = ROOT / "mujoco_rl" / "checkpoints" / "full_body_v1" / "policy.zip"
STAGE_ORDER = ("nav", "squat", "recover", "run", "jump")


def _factory(skill: str, seed: int):
    def create():
        env = Monitor(SkillEnv(skill, randomize=True,
                               curriculum_level=0.0 if skill == "nav" else 1.0))
        env.reset(seed=seed)
        return env
    return create


def vector_env(skill: str, workers: int, seed: int):
    factories = [_factory(skill, seed + index) for index in range(workers)]
    return SubprocVecEnv(factories, start_method="forkserver") if workers > 1 else DummyVecEnv(factories)


def _stats_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")


def _warm_start(model: PPO, normalizer: VecNormalize, checkpoint: Path) -> None:
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
    normalizer.obs_rms.count = old_norm.obs_rms.count
    print(f"[warm-start] transferred {len(old_mean)} observation channels from {checkpoint}", flush=True)


class GateCallback(BaseCallback):
    def __init__(self, skill: str, run_dir: Path, deadline: float, interval: int):
        super().__init__()
        self.skill = skill
        self.run_dir = run_dir
        self.deadline = deadline
        self.interval = interval
        self.last_eval = 0
        self.best_score = -float("inf")
        self.accepted = False
        self.holdout_attempt = 0

    def _on_training_start(self) -> None:
        self.last_eval = self.model.num_timesteps

    def _on_step(self) -> bool:
        if time.monotonic() >= self.deadline:
            return False
        if self.num_timesteps - self.last_eval < self.interval:
            return True
        self.last_eval = self.num_timesteps
        normalizer = self.model.get_env()
        assert isinstance(normalizer, VecNormalize)
        result = evaluate(self.model, normalizer, self.skill)
        record = {"training_steps": self.num_timesteps, "development": result}
        checkpoint_name = f"checkpoint_{self.num_timesteps}"
        self.model.save(str(self.run_dir / checkpoint_name))
        normalizer.save(str(self.run_dir / f"{checkpoint_name}_vecnormalize.pkl"))
        episodes = result["episodes"]
        mean_duration = float(np.mean([item["seconds"] for item in episodes]))
        if self.skill == "nav":
            near_count = sum(item["distance_to_goal_m"] < 0.25 for item in episodes)
            mean_goal_distance = float(np.mean([item["distance_to_goal_m"] for item in episodes]))
            score = 10.0 * result["pass_rate"] + 0.02 * near_count + 0.01 * mean_duration - 0.2 * mean_goal_distance
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
        print(f"[eval] {self.skill} {self.num_timesteps:,} steps: dev={result['pass_rate']:.1%}, "
              f"holdout={record.get('holdout', {}).get('pass_rate', 'pending')}, "
              f"accepted={self.accepted}", flush=True)
        return not self.accepted and time.monotonic() < self.deadline


def train_stage(skill: str, run_dir: Path, workers: int, seed: int, hours: float,
                warm_start: Path | None, resume: Path | None, eval_interval: int,
                max_steps: int) -> tuple[bool, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    base = vector_env("run_train" if skill == "run" else skill, workers, seed)
    if resume:
        env = VecNormalize.load(str(_stats_path(resume)), base)
        env.training = True
        model = PPO.load(str(resume), env=env, device="cpu")
        learning_rate = 5e-5 if skill == "nav" else 1e-4 if skill == "run" else 3e-4
        model.lr_schedule = get_schedule_fn(learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        env = VecNormalize(base, norm_obs=True, norm_reward=False, clip_obs=10.0)
        learning_rate = 5e-5 if skill == "nav" else 1e-4 if skill == "run" else 3e-4
        model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=1,
                    n_steps=256, batch_size=512, n_epochs=4, learning_rate=learning_rate,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.15,
                    ent_coef=0.001 if skill in {"nav", "run"} else 0.003,
                    target_kl=0.015, max_grad_norm=1.0,
                    policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                                   "activation_fn": torch.nn.ELU, "log_std_init": -1.0})
        if warm_start:
            _warm_start(model, env, warm_start)
    config = {"skill": skill, "workers": workers, "seed": seed,
              "hours": hours, "max_steps": max_steps, "eval_interval": eval_interval,
              "learning_rate": learning_rate,
              "navigation_curriculum_steps": 1_500_000 if skill == "nav" else None,
              "warm_start": str(warm_start) if warm_start else None,
              "resume": str(resume) if resume else None,
              "development_seeds": [10_001, 10_050], "first_holdout_seeds": [10_051, 10_100],
              "source_sha256": {}}
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in ("skill_env.py", "skill_train.py", "skill_eval.py",
                     "skill_supervisor.py", "full_body_env.py"):
        source = Path(__file__).with_name(filename)
        shutil.copy2(source, snapshot / filename)
        config["source_sha256"][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    callback = GateCallback(skill, run_dir, time.monotonic() + hours * 3600, eval_interval)
    try:
        remaining = max_steps - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback,
                        reset_num_timesteps=resume is None)
        model.save(str(run_dir / "final"))
        env.save(str(run_dir / "final_vecnormalize.pkl"))
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100_000_000)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.hours_per_stage <= 0 or args.workers < 1 or args.max_steps < 1 or args.eval_interval < 1:
        parser.error("Invalid training budget, workers, or evaluation interval")
    if args.stage == "pipeline" and (args.resume or args.warm_start):
        parser.error("Use --stage for a custom warm start or resume")
    torch.set_num_threads(4)
    if args.stage != "pipeline":
        train_stage(args.stage, args.run_dir, args.workers, args.seed,
                    args.hours_per_stage, args.warm_start or (WALK_CHECKPOINT if not args.resume else None),
                    args.resume, args.eval_interval, args.max_steps)
        return
    accepted = {}
    for index, skill in enumerate(STAGE_ORDER):
        source = (WALK_CHECKPOINT if skill == "nav" else
                  accepted["nav"] if skill in {"squat", "run"} else
                  accepted["squat"] if skill in {"recover", "jump"} else None)
        passed, checkpoint = train_stage(skill, args.run_dir / skill,
                                         args.workers, args.seed + index, args.hours_per_stage,
                                         source, None, args.eval_interval, args.max_steps)
        if not passed:
            print(f"[pipeline] stopped at {skill}; gate not met within budget", flush=True)
            break
        accepted[skill] = checkpoint
    else:
        print("[pipeline] all five skill gates passed", flush=True)


if __name__ == "__main__":
    main()
