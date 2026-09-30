"""Evaluate full-body policies over fixed, randomized MuJoCo episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def evaluate(model, normalizer, task: str, seeds: int = 50, seed_start: int = 10_001) -> dict:
    from .full_body_env import EPISODE_STEPS, FullBodyEnv, POLICY_DT

    env = FullBodyEnv(task=task, randomize=True)
    episodes = []
    try:
        for seed in range(seed_start, seed_start + seeds):
            obs, _ = env.reset(seed=seed)
            slips, rates, efforts, tilts, self_contacts = [], [], [], [], []
            final = {}
            for steps in range(1, EPISODE_STEPS + 1):
                normalized = normalizer.normalize_obs(obs[None, :])
                action, _ = model.predict(normalized, deterministic=True)
                obs, _, terminated, truncated, final = env.step(action[0])
                slips.append(final["slip_m_s"])
                rates.append(final["action_rate"])
                efforts.append(final["effort"])
                tilts.append(final["tilt_cos"])
                self_contacts.append(final["self_contacts"])
                if terminated or truncated:
                    break
            item = {
                "seed": seed,
                "seconds": round(steps * POLICY_DT, 2),
                "fallen": final["fallen"],
                "distance_m": round(final["forward_distance_m"], 3),
                "lateral_m": round(final["lateral_distance_m"], 3),
                "mean_slip_m_s": round(float(np.mean(slips)), 4),
                "mean_action_rate": round(float(np.mean(rates)), 5),
                "mean_effort": round(float(np.mean(efforts)), 4),
                "min_tilt_cos": round(float(np.min(tilts)), 4),
                "contact_entries": final["contact_entries"],
                "self_contact_steps": sum(value > 0 for value in self_contacts),
            }
            if task == "stand":
                item["passed"] = (steps == EPISODE_STEPS and not final["fallen"]
                                  and item["mean_slip_m_s"] < 0.05
                                  and item["mean_action_rate"] < 0.05
                                  and abs(item["distance_m"]) < 0.20
                                  and abs(item["lateral_m"]) < 0.20)
            else:
                item["passed"] = (steps == EPISODE_STEPS and not final["fallen"]
                                  and 1.5 <= item["distance_m"] <= 2.5
                                  and abs(item["lateral_m"]) < 0.30
                                  and item["mean_slip_m_s"] < 0.05
                                  and item["mean_action_rate"] < 0.05
                                  and max(item["contact_entries"]) <= 20
                                  and item["self_contact_steps"] == 0)
            episodes.append(item)
    finally:
        env.close()
    pass_rate = sum(item["passed"] for item in episodes) / seeds
    return {
        "task": task,
        "seeds": seeds,
        "seed_start": seed_start,
        "pass_rate": round(pass_rate, 3),
        "criterion_met": pass_rate >= (0.90 if task == "stand" else 0.80),
        "mean_distance_m": round(float(np.mean([item["distance_m"] for item in episodes])), 3),
        "mean_slip_m_s": round(float(np.mean([item["mean_slip_m_s"] for item in episodes])), 4),
        "mean_action_rate": round(float(np.mean([item["mean_action_rate"] for item in episodes])), 5),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=("stand", "walk"), required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10_001)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-seconds", type=float, default=10.0)
    parser.add_argument("--video-seed", type=int, default=10_001)
    args = parser.parse_args()
    if args.seeds < 1 or args.video_seconds <= 0:
        parser.error("--seeds and --video-seconds must be positive")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .full_body_env import EPISODE_STEPS, FullBodyEnv

    stats = args.checkpoint.with_name(args.checkpoint.stem + "_vecnormalize.pkl")
    normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: FullBodyEnv(task=args.task)]))
    normalizer.training = False
    model = PPO.load(str(args.checkpoint), device="cpu")
    try:
        result = evaluate(model, normalizer, args.task, args.seeds, args.seed_start)
        print(json.dumps(result, indent=2))
        if args.video:
            replay = FullBodyEnv(task=args.task, render_mode="rgb_array", randomize=True)
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec="libx264")
            try:
                obs, _ = replay.reset(seed=args.video_seed)
                for step in range(min(EPISODE_STEPS, int(args.video_seconds / 0.02))):
                    action, _ = model.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
                    obs, _, terminated, truncated, _ = replay.step(action[0])
                    if step % 2:
                        writer.append_data(replay.render())
                    if terminated or truncated:
                        break
            finally:
                writer.close()
                replay.close()
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
