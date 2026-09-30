"""Train Asimov's full body to stand, then walk, on this Loudbox."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from .full_body_env import FullBodyEnv, MirroredFullBodyEnv, JOINT_NAMES, PHYSICS_DT, POLICY_DT
from .full_body_eval import evaluate


def make_env(task: str, seed: int, mirror: bool = False):
    def create():
        physical = MirroredFullBodyEnv(task=task) if mirror else FullBodyEnv(task=task, randomize=True)
        env = Monitor(physical)
        env.reset(seed=seed)
        return env
    return create


def vector_env(task: str, workers: int, seed: int, mirror: bool = False):
    factories = [make_env(task, seed + i, mirror=mirror) for i in range(workers)]
    return (SubprocVecEnv(factories, start_method="forkserver") if workers > 1
            else DummyVecEnv(factories))


def benchmark_workers(seed: int) -> int:
    results = {}
    for workers in (8, 16, 24):
        env = vector_env("stand", workers, seed)
        try:
            env.reset()
            actions = np.zeros((workers, len(JOINT_NAMES)), dtype=np.float32)
            start = time.monotonic()
            for _ in range(128):
                env.step(actions)
            rate = 128 * workers / (time.monotonic() - start)
            results[workers] = rate
            print(f"[benchmark] {workers} workers: {rate:.1f} policy steps/s", flush=True)
        finally:
            env.close()
    winner = max(results, key=results.get)
    print(f"[benchmark] selected {winner} workers", flush=True)
    return winner


class EvaluationCallback(BaseCallback):
    def __init__(self, task: str, run_dir: Path, deadline: float, interval: int):
        super().__init__()
        self.task = task
        self.run_dir = run_dir
        self.deadline = deadline
        self.interval = interval
        self.last_eval = 0
        self.best_score = -float("inf")
        self.passed = False
        self.last_result: dict | None = None

    def _on_training_start(self) -> None:
        self.last_eval = self.model.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for name in ("reward_upright", "reward_height", "reward_tracking", "slip_m_s",
                         "impact", "effort", "action_rate", "action_accel", "overspeed",
                         "rapid_contacts"):
                if name in info:
                    self.logger.record_mean(f"robot/{name}", float(info[name]))
        if time.monotonic() >= self.deadline:
            return False
        if self.num_timesteps - self.last_eval < self.interval:
            return True
        self.last_eval = self.num_timesteps
        normalizer = self.model.get_env()
        assert isinstance(normalizer, VecNormalize)
        result = evaluate(self.model, normalizer, self.task, seeds=50)
        self.last_result = result
        name = f"checkpoint_{self.num_timesteps}"
        self.model.save(str(self.run_dir / name))
        normalizer.save(str(self.run_dir / f"{name}_vecnormalize.pkl"))
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"training_steps": self.num_timesteps, **result}) + "\n")
        score = result["pass_rate"] * 10.0 - 5.0 * result["mean_slip_m_s"]
        if self.task == "walk":
            score -= abs(result["mean_distance_m"] - 2.0)
        else:
            score -= result["mean_action_rate"]
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.run_dir / "best"))
            normalizer.save(str(self.run_dir / "best_vecnormalize.pkl"))
        self.logger.record("eval/pass_rate", result["pass_rate"])
        self.logger.record("eval/mean_distance_m", result["mean_distance_m"])
        self.logger.record("eval/mean_slip_m_s", result["mean_slip_m_s"])
        print(f"[eval] {self.task} {self.num_timesteps:,} steps: "
              f"pass={result['pass_rate']:.1%}, distance={result['mean_distance_m']} m, "
              f"slip={result['mean_slip_m_s']} m/s", flush=True)
        self.passed = bool(result["criterion_met"])
        return not self.passed and time.monotonic() < self.deadline


def train_stage(task: str, run_dir: Path, workers: int, seed: int, deadline: float,
                max_steps: int, resume: Path | None = None, learning_rate: float = 3e-4,
                ent_coef: float = 0.001, target_kl: float = 0.03,
                mirror: bool = False, eval_interval: int = 0,
                clip_range: float = 0.2, n_epochs: int = 5) -> tuple[bool, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    base_env = vector_env(task, workers, seed, mirror=mirror)
    if resume:
        stats = resume.with_name(resume.stem + "_vecnormalize.pkl")
        if not stats.is_file():
            base_env.close()
            raise FileNotFoundError(f"Missing normalization statistics: {stats}")
        env = VecNormalize.load(str(stats), base_env)
        env.training = True
        model = PPO.load(str(resume), env=env, device="cpu", learning_rate=learning_rate,
                         ent_coef=ent_coef, target_kl=target_kl)
        model.lr_schedule = get_schedule_fn(learning_rate)
        model.clip_range = get_schedule_fn(clip_range)
        model.n_epochs = n_epochs
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env, device="cpu", seed=seed, verbose=1,
            n_steps=256, batch_size=512, n_epochs=n_epochs, learning_rate=learning_rate,
            gamma=0.99, gae_lambda=0.95, clip_range=clip_range,
            ent_coef=ent_coef, target_kl=target_kl, max_grad_norm=1.0,
            policy_kwargs={
                "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                "activation_fn": torch.nn.ELU,
                "log_std_init": -1.0,
            },
        )
    config = {
        "task": task, "seed": seed, "workers": workers,
        "model_revision": "732cc60dcb8f2b4fd26c3d7346b35f9b89c3cd47",
        "physics_dt": PHYSICS_DT, "policy_dt": POLICY_DT,
        "max_steps": max_steps, "resume": str(resume) if resume else None,
        "algorithm": "Stable-Baselines3 PPO", "actor_critic": [256, 256, 128],
        "learning_rate": learning_rate, "entropy_coefficient": ent_coef,
        "target_kl": target_kl, "mirror_training_episodes": mirror,
        "clip_range": clip_range, "n_epochs": n_epochs,
        "evaluation_seeds": 50,
        "eval_interval": eval_interval or (100_000 if task == "stand" else 250_000),
    }
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    config["source_sha256"] = {}
    for filename in ("full_body_env.py", "full_body_train.py", "full_body_eval.py"):
        source = Path(__file__).with_name(filename)
        shutil.copy2(source, snapshot_dir / filename)
        config["source_sha256"][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    callback = EvaluationCallback(task, run_dir, deadline,
                                  interval=config["eval_interval"])
    try:
        remaining = max(0, max_steps - (model.num_timesteps if task == "stand" else 0))
        if remaining and time.monotonic() < deadline:
            model.learn(total_timesteps=remaining, callback=callback,
                        reset_num_timesteps=resume is None)
        model.save(str(run_dir / "final"))
        env.save(str(run_dir / "final_vecnormalize.pkl"))
        if callback.last_result is None or not callback.passed:
            result = evaluate(model, env, task, seeds=50)
            callback.last_result = result
            callback.passed = bool(result["criterion_met"])
            with (run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"training_steps": model.num_timesteps,
                                         "final": True, **result}) + "\n")
        print(f"[stage] {task}: passed={callback.passed}; "
              f"steps={model.num_timesteps:,}; artifacts={run_dir}", flush=True)
        checkpoint = run_dir / ("best.zip" if callback.passed and (run_dir / "best.zip").exists()
                                else "final.zip")
        return callback.passed, checkpoint
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pipeline", "stand", "walk"), default="pipeline")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=0, help="0 benchmarks 8, 16, and 24 workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=0, help="0 uses the stage default")
    parser.add_argument("--mirror", action="store_true", help="mirror half of training episodes")
    parser.add_argument("--run-dir", type=Path, default=Path("logs/mujoco_rl/full_body_v1"))
    parser.add_argument("--resume", type=Path, help="checkpoint for a single-stage run")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if (args.hours <= 0 or args.steps <= 0 or args.workers < 0
            or args.learning_rate <= 0 or args.ent_coef < 0 or args.target_kl <= 0
            or args.eval_interval < 0 or not 0 < args.clip_range < 1 or args.n_epochs < 1):
        parser.error("invalid training budget, worker count, or PPO hyperparameters")
    if args.stage == "pipeline" and args.resume:
        parser.error("--resume requires --stage stand or --stage walk")
    if args.stage == "walk" and not args.resume:
        parser.error("--stage walk requires a standing checkpoint via --resume")

    torch.set_num_threads(4)
    if args.benchmark_only:
        benchmark_workers(args.seed)
        return
    workers = args.workers or benchmark_workers(args.seed)
    now = time.monotonic()
    deadline = now + args.hours * 3600
    if args.stage == "pipeline":
        standing_deadline = min(deadline, now + 4 * 3600)
        passed, checkpoint = train_stage("stand", args.run_dir / "stand", workers,
                                         args.seed, standing_deadline, args.steps,
                                         learning_rate=args.learning_rate,
                                         ent_coef=args.ent_coef, target_kl=args.target_kl,
                                         mirror=args.mirror, eval_interval=args.eval_interval,
                                         clip_range=args.clip_range, n_epochs=args.n_epochs)
        if passed and time.monotonic() < deadline:
            train_stage("walk", args.run_dir / "walk", workers, args.seed + 1,
                        deadline, args.steps, resume=checkpoint,
                        learning_rate=args.learning_rate,
                        ent_coef=args.ent_coef, target_kl=args.target_kl,
                        mirror=args.mirror, eval_interval=args.eval_interval,
                        clip_range=args.clip_range, n_epochs=args.n_epochs)
        elif not passed:
            print("[pipeline] standing gate not met; walking stage was not started", flush=True)
    else:
        train_stage(args.stage, args.run_dir, workers, args.seed, deadline,
                    args.steps, resume=args.resume, learning_rate=args.learning_rate,
                    ent_coef=args.ent_coef, target_kl=args.target_kl,
                    mirror=args.mirror, eval_interval=args.eval_interval,
                    clip_range=args.clip_range, n_epochs=args.n_epochs)


if __name__ == "__main__":
    main()
