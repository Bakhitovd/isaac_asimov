"""Record a validated neural policy's successful rollouts for PPO retention."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .skill_env import RECOVERY_POSES, RECOVERY_TILT_RANGES, SkillEnv
from .skill_train import _recovery_level_from_checkpoint, _stats_path


_model = None
_normalizer = None
_environment = None
_level = 0
_seed_start = 0


def _initialize(checkpoint: str, level: int, seed_start: int) -> None:
    global _model, _normalizer, _environment, _level, _seed_start
    torch.set_num_threads(1)
    path = Path(checkpoint)
    _model = PPO.load(path, device="cpu")
    _normalizer = VecNormalize.load(str(_stats_path(path)), DummyVecEnv([lambda: SkillEnv("recover")]))
    _normalizer.training = False
    _environment = SkillEnv("recover", recovery_level=level,
                            recovery_sensor_memory=getattr(_model, "recovery_sensor_memory", False),
                            recovery_phase_features=getattr(_model, "recovery_phase_features", False))
    _level, _seed_start = level, seed_start


def collection_case(index: int, level: int) -> tuple[int, str]:
    """Balance all four poses within each difficulty slot, including rehearsal."""
    block, slot = divmod(index, 40)
    difficulty, pose_index = divmod(slot, 4)
    episode_level = (level if difficulty < 5 else min(level + 1, len(RECOVERY_TILT_RANGES) - 1)
                     if difficulty < 8 else (2 * block + difficulty - 8) % max(1, level))
    return episode_level, RECOVERY_POSES[pose_index]


def _trial(index: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    level, pose = collection_case(index, _level)
    _environment.set_recovery_level(level)
    observation, _ = _environment.reset(seed=_seed_start + index, options={"recovery_pose": pose})
    observations, actions, policy_means = [], [], []
    for _ in range(600):
        normalized = _normalizer.normalize_obs(observation[None, :])
        with torch.no_grad():
            mean = _model.policy.get_distribution(torch.as_tensor(normalized, device=_model.device)).distribution.mean
        mean = mean.cpu().numpy()[0]
        action = np.clip(mean, -1.0, 1.0)
        observations.append(observation.copy())
        actions.append(action.copy())
        policy_means.append(mean.copy())
        observation, _, terminated, truncated, info = _environment.step(action)
        if terminated or truncated:
            break
    summary = {"seed": _seed_start + index, "level": level, "tilt_deg": _environment.recovery_tilt_deg,
               "pose": info["recovery_pose"], "success": bool(info["success"])}
    if not info["success"]:
        return summary, np.empty((0, 83), np.float32), np.empty((0, 23), np.float32), np.empty((0, 23), np.float32)
    return summary, np.asarray(observations), np.asarray(actions), np.asarray(policy_means)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=310000)
    args = parser.parse_args()
    if args.episodes < 20 or args.workers < 1 or args.seed_start < 0:
        parser.error("Use at least 20 episodes, one worker, and a nonnegative seed")
    level = _recovery_level_from_checkpoint(args.checkpoint)
    summaries, observations, actions, policy_means, starts = [], [], [], [], []
    frame_count = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_initialize,
                                      initargs=(str(args.checkpoint), level, args.seed_start)) as pool:
        for summary, obs, act, means in pool.imap(_trial, range(args.episodes)):
            summaries.append(summary)
            if len(obs):
                starts.append(frame_count)
                frame_count += len(obs)
                observations.append(obs)
                actions.append(act)
                policy_means.append(means)
            if len(summaries) % 64 == 0:
                print(f"[retention] episodes={len(summaries)}, successful={len(observations)}", flush=True)
    if not observations:
        raise RuntimeError("Reference policy produced no successful retention rollouts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, observations=np.concatenate(observations),
                        actions=np.concatenate(actions), policy_means=np.concatenate(policy_means),
                        episode_starts=np.asarray(starts))
    report = {"reference_checkpoint": str(args.checkpoint), "episodes": args.episodes,
              "successful_episodes": len(observations), "frames": frame_count,
              "seed_start": args.seed_start, "imitation_targets": "unclipped_policy_mean", "episodes_detail": summaries}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[retention] saved {frame_count} frames to {args.output}", flush=True)


if __name__ == "__main__":
    main()
