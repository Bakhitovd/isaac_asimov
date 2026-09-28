"""Replay a forward policy, report progress, and optionally record video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PPO checkpoint ZIP")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--video", type=Path, help="Output MP4 path")
    parser.add_argument("--seed", type=int, default=10_001)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    if args.video:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .environment import AsimovForwardEnv

    statistics = args.checkpoint.with_name(args.checkpoint.stem + "_vecnormalize.pkl")
    physical_env = AsimovForwardEnv(render_mode="rgb_array" if args.video else None)
    env = VecNormalize.load(str(statistics), DummyVecEnv([lambda: physical_env]))
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(args.checkpoint), env=env, device="cpu")
    env.seed(args.seed)
    obs = env.reset()
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.video), fps=25, codec="libx264")
    final_info = {}
    steps = 0
    try:
        for steps in range(1, int(args.seconds / 0.02) + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            final_info = infos[0]
            if writer is not None and steps % 2 == 0:
                writer.append_data(physical_env.render())
            if dones[0]:
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()
    result = {
        "steps": steps,
        "seconds": round(steps * 0.02, 2),
        "forward_distance_m": final_info.get("forward_distance"),
        "mean_forward_speed_m_s": float(final_info.get("forward_distance", 0.0)) / (steps * 0.02),
        "target_forward_speed_m_s": 0.35,
        "lateral_distance_m": final_info.get("lateral_distance"),
        "fallen": final_info.get("fallen"),
        "video": str(args.video) if args.video else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
