"""Record recovery failures with sensor, actuator, and contact trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .skill_env import SkillEnv
from .skill_train import _stats_path


def record(checkpoint: Path, output: Path, seeds: list[int], level: int) -> list[dict]:
    model = PPO.load(checkpoint, device="cpu")
    normalizer = VecNormalize.load(str(_stats_path(checkpoint)), DummyVecEnv([lambda: SkillEnv("recover")]))
    normalizer.training = False
    env = SkillEnv("recover", recovery_level=level,
                   recovery_sensor_memory=getattr(model, "recovery_sensor_memory", False),
                   recovery_phase_features=getattr(model, "recovery_phase_features", False))
    summaries = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            frames = []
            total_reward = 0.0
            for step in range(600):
                action, _ = model.predict(normalizer.normalize_obs(observation[None, :]), deterministic=True)
                # State and observation precede the action; target and torque follow it.
                state = {"seconds": step * 0.02, "observation": observation.copy(),
                         "root_position": env.data.qpos[:3].copy(),
                         "root_quaternion": env.data.qpos[3:7].copy(),
                         "center_of_mass": env.data.subtree_com[1].copy(),
                         "upright_cos": float(env.data.site_xmat[env.imu_site_id].reshape(3, 3)[2, 2]),
                         "gyro": env.data.sensor("imu_ang_vel").data.copy(),
                         "root_velocity": env.data.qvel[:6].copy(),
                         "joint_position": env.data.qpos[env.qpos_ids].copy(),
                         "joint_velocity": env.data.qvel[env.qvel_ids].copy(), "action": action[0].copy()}
                observation, reward, terminated, truncated, info = env.step(action[0])
                total_reward += reward
                state.update(target=env.target.copy(), torque=env.data.ctrl[env.ctrl_ids].copy(),
                             contacts=env.last_contacts.copy(), reward=reward)
                frames.append(state)
                if terminated or truncated:
                    break
            arrays = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
            np.savez_compressed(output / f"seed_{seed}.npz", **arrays,
                                joint_ranges=env.joint_ranges, speed_limits=env.speed, effort_limits=env.effort)
            late = arrays["seconds"] >= 0.8
            summary = {"seed": seed, "checkpoint": str(checkpoint), **info, "total_reward": total_reward,
                       "early_leg_action_rms": float(np.sqrt(np.mean(arrays["action"][~late, :12] ** 2))),
                       "late_leg_action_rms": float(np.sqrt(np.mean(arrays["action"][late, :12] ** 2)))
                       if late.any() else None,
                       "peak_root_angular_speed": float(np.max(
                           np.linalg.norm(arrays["root_velocity"][:, 3:6], axis=1)))}
            if not np.isclose(sum(info["recovery_reward_totals"].values()), total_reward):
                raise RuntimeError("Reward components do not account for the episode return")
            summaries.append(summary)
            print(json.dumps({key: summary[key] for key in (
                "seed", "recovery_pose", "recovery_tilt_deg", "success", "seconds", "termination_reason",
                "early_leg_action_rms", "late_leg_action_rms", "peak_root_angular_speed")}), flush=True)
    finally:
        env.close()
        normalizer.close()
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--level", type=int, default=9)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record(args.checkpoint, args.output, args.seeds, args.level)


if __name__ == "__main__":
    main()
