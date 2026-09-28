"""Train a forward-walking Asimov policy on this machine's CPUs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

from .environment import AsimovForwardEnv


def make_env(seed: int, reward_stage: str = "target"):
    def create():
        env = Monitor(AsimovForwardEnv(reward_stage=reward_stage))
        env.reset(seed=seed)
        return env

    return create


def evaluate_policy(model: PPO, training_env: VecNormalize) -> dict:
    env = VecNormalize(DummyVecEnv([make_env(10_001)]), training=False, norm_reward=False)
    sync_envs_normalization(training_env, env)
    replays = []
    try:
        for seed in range(10_001, 10_006):
            env.seed(seed)
            obs = env.reset()
            final_info: dict = {}
            for steps in range(1, 501):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                final_info = infos[0]
                if dones[0]:
                    break
            replays.append({
                "seed": seed,
                "steps": steps,
                "forward_distance_m": round(float(final_info.get("forward_distance", 0.0)), 3),
                "lateral_distance_m": round(float(final_info.get("lateral_distance", 0.0)), 3),
                "fallen": bool(final_info.get("fallen", False)),
            })
    finally:
        env.close()
    return {
        "min_seconds": min(item["steps"] for item in replays) * 0.02,
        "min_forward_distance_m": min(item["forward_distance_m"] for item in replays),
        "max_forward_distance_m": max(item["forward_distance_m"] for item in replays),
        "max_abs_target_distance_error_m": max(abs(item["forward_distance_m"] - 3.5) for item in replays),
        "max_abs_lateral_distance_m": max(abs(item["lateral_distance_m"]) for item in replays),
        "any_fall": any(item["fallen"] for item in replays),
        "replays": replays,
    }


class ForwardEvalCallback(BaseCallback):
    def __init__(self, run_dir: Path, interval: int = 250_000, stop_on_success: bool = True):
        super().__init__()
        self.run_dir = run_dir
        self.interval = interval
        self.stop_on_success = stop_on_success
        self.last_evaluation = 0
        self.best_score = -float("inf")
        self.succeeded = False

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_evaluation < self.interval:
            return True
        self.last_evaluation = self.num_timesteps
        train_env = self.model.get_env()
        assert isinstance(train_env, VecNormalize)
        metrics = evaluate_policy(self.model, train_env)
        steps = self.num_timesteps
        checkpoint = self.run_dir / f"checkpoint_{steps}"
        self.model.save(str(checkpoint))
        train_env.save(str(self.run_dir / f"checkpoint_{steps}_vecnormalize.pkl"))
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"training_steps": steps, **metrics}) + "\n")
        score = (
            - float(metrics["max_abs_target_distance_error_m"])
            - float(metrics["max_abs_lateral_distance_m"])
            - (5.0 if metrics["any_fall"] else 0.0)
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.run_dir / "best"))
            train_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
        self.logger.record("eval/min_forward_distance_m", metrics["min_forward_distance_m"])
        self.logger.record("eval/max_forward_distance_m", metrics["max_forward_distance_m"])
        self.logger.record("eval/max_abs_lateral_distance_m", metrics["max_abs_lateral_distance_m"])
        self.logger.record("eval/any_fall", float(metrics["any_fall"]))
        print(f"[eval] {steps:,} steps: {metrics}", flush=True)
        self.succeeded = (
            not metrics["any_fall"]
            and float(metrics["min_seconds"]) >= 10.0
            and float(metrics["min_forward_distance_m"]) >= 2.0
            and float(metrics["max_forward_distance_m"]) <= 4.5
            and float(metrics["max_abs_lateral_distance_m"]) <= 0.75
        )
        return not (self.succeeded and self.stop_on_success)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=Path("logs/mujoco_rl/forward_seed42"))
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-stage", choices=("progress", "target"), default="target")
    parser.add_argument("--resume", type=Path, help="Path to a checkpoint ZIP")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.steps < 1 or args.learning_rate <= 0:
        parser.error("--workers, --steps, and --learning-rate must be positive")

    torch.set_num_threads(4)
    constructors = [make_env(args.seed + rank, args.reward_stage) for rank in range(args.workers)]
    vector_env = SubprocVecEnv(constructors, start_method="forkserver") if args.workers > 1 else DummyVecEnv(constructors)
    if args.benchmark_only:
        vector_env.reset()
        actions = np.zeros((args.workers, 12), dtype=np.float32)
        start = time.monotonic()
        for _ in range(256):
            vector_env.step(actions)
        elapsed = time.monotonic() - start
        print(f"{args.workers} workers: {256 * args.workers / elapsed:.1f} env steps/s ({elapsed:.2f}s)")
        vector_env.close()
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": args.seed, "workers": args.workers, "max_steps": args.steps,
        "eval_interval": args.eval_interval, "physics_dt": 0.005,
        "policy_dt": 0.02, "target_forward_speed_m_s": 0.35,
        "network": [256, 256, 128], "algorithm": "PPO",
        "learning_rate": args.learning_rate, "entropy_coefficient": 0.001,
        "target_kl": 0.03,
        "reward_stage": args.reward_stage,
        "resumed_from": str(args.resume) if args.resume else None,
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.resume:
        statistics = args.resume.with_name(args.resume.stem + "_vecnormalize.pkl")
        env = VecNormalize.load(str(statistics), vector_env)
        env.training = True
        model = PPO.load(
            str(args.resume), env=env, device="cpu",
            learning_rate=args.learning_rate, ent_coef=0.001, target_kl=0.03,
        )
        remaining = max(0, args.steps - model.num_timesteps)
    else:
        env = VecNormalize(vector_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
            n_steps=256, batch_size=512, n_epochs=5, learning_rate=args.learning_rate,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.001, max_grad_norm=1.0, target_kl=0.03,
            policy_kwargs={
                "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
                "activation_fn": torch.nn.ELU,
                "log_std_init": -1.0,
            },
            tensorboard_log=None,
        )
        remaining = args.steps

    callback = ForwardEvalCallback(
        args.run_dir, interval=args.eval_interval,
        stop_on_success=args.reward_stage == "target",
    )
    try:
        if remaining:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(str(args.run_dir / "final"))
        env.save(str(args.run_dir / "final_vecnormalize.pkl"))
        print(f"Training finished. Criterion met: {callback.succeeded}. Artifacts: {args.run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
