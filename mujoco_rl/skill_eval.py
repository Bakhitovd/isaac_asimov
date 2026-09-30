"""Evaluate navigation, squat, recovery, running, and jumping checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def evaluate(model, normalizer, skill: str, seed_start: int = 10_001, seeds: int = 50) -> dict:
    from .skill_env import MAX_STEPS, SkillEnv

    if skill not in MAX_STEPS:
        raise ValueError(f"Cannot evaluate {skill}")
    env = SkillEnv(skill=skill, randomize=True)
    episodes = []
    try:
        for seed in range(seed_start, seed_start + seeds):
            observation, _ = env.reset(seed=seed)
            slip, action_rate, impact = [], [], []
            final = {}
            for _ in range(MAX_STEPS[skill]):
                normalized = normalizer.normalize_obs(observation[None, :])
                action, _ = model.predict(normalized, deterministic=True)
                observation, _, terminated, truncated, final = env.step(action[0])
                slip.append(final["slip_m_s"])
                action_rate.append(final["action_rate"])
                impact.append(final["impact_weight"])
                if terminated or truncated:
                    break
            item = {"seed": seed, **final,
                    "mean_slip_m_s": round(float(np.mean(slip)), 4),
                    "mean_action_rate": round(float(np.mean(action_rate)), 5),
                    "max_impact_weight": round(float(np.max(impact)), 3),
                    "passed": bool(final["success"])}
            episodes.append(item)
    finally:
        env.close()
    pass_rate = sum(item["passed"] for item in episodes) / seeds
    group_rates = {}
    if skill == "recover":
        for pose in ("front", "back", "left", "right"):
            group = [item for item in episodes if item["recovery_pose"] == pose]
            group_rates[pose] = sum(item["passed"] for item in group) / len(group) if group else 0.0
    if skill == "jump":
        for variant in ("up", "forward"):
            group = [item for item in episodes if item["jump_variant"] == variant]
            group_rates[variant] = sum(item["passed"] for item in group) / len(group) if group else 0.0
    return {"skill": skill, "seed_start": seed_start, "seeds": seeds,
            "pass_rate": round(pass_rate, 3),
            "group_rates": {key: round(value, 3) for key, value in group_rates.items()},
            "criterion_met": pass_rate >= 0.80 and all(value >= 0.75 for value in group_rates.values()),
            "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skill", required=True, choices=("nav", "squat", "recover", "run", "jump"))
    parser.add_argument("--seed-start", type=int, default=10_001)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-seed", type=int, default=10_001)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .skill_env import MAX_STEPS, SkillEnv

    stats = args.checkpoint.with_name(args.checkpoint.stem + "_vecnormalize.pkl")
    normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: SkillEnv(args.skill)]))
    normalizer.training = False
    model = PPO.load(str(args.checkpoint), device="cpu")
    try:
        print(json.dumps(evaluate(model, normalizer, args.skill, args.seed_start, args.seeds), indent=2))
        if args.video:
            env = SkillEnv(args.skill, render_mode="rgb_array")
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec="libx264")
            try:
                observation, _ = env.reset(seed=args.video_seed)
                for step in range(MAX_STEPS[args.skill]):
                    action, _ = model.predict(normalizer.normalize_obs(observation[None, :]), deterministic=True)
                    observation, _, terminated, truncated, _ = env.step(action[0])
                    if step % 2:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                writer.close()
                env.close()
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
