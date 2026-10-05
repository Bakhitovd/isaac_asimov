"""Evaluate navigation, squat, recovery, running, and jumping checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def evaluate(model, normalizer, skill: str, seed_start: int = 10_001, seeds: int = 50,
             recovery_level: int | None = None, scenario: str = "standard") -> dict:
    from .skill_env import MAX_STEPS, SkillEnv
    from .skill_scenarios import reset_case

    if skill not in MAX_STEPS:
        raise ValueError(f"Cannot evaluate {skill}")
    from .run_contract import environment_for
    env = environment_for(model, skill, randomize=True, recovery_level=recovery_level,
                   recovery_sensor_memory=getattr(model, "recovery_sensor_memory", False),
                   recovery_phase_features=getattr(model, "recovery_phase_features", False))
    episodes = []
    try:
        for seed in range(seed_start, seed_start + seeds):
            observation, initial = reset_case(env, seed, scenario)
            slip, action_rate, impact = [], [], []
            quality = {key: [] for key in ("target_velocity_rms", "joint_acceleration_rms",
                                           "torque_saturation_fraction")}
            final = {}
            for _ in range(MAX_STEPS[skill]):
                normalized = normalizer.normalize_obs(observation[None, :])
                action, _ = model.predict(normalized, deterministic=True)
                observation, _, terminated, truncated, final = env.step(action[0])
                slip.append(final["slip_m_s"])
                action_rate.append(final["action_rate"])
                impact.append(final["impact_weight"])
                for key in quality:
                    quality[key].append(final[key])
                if terminated or truncated:
                    break
            item = {"seed": seed, **final,
                    "mean_slip_m_s": round(float(np.mean(slip)), 4),
                    "mean_action_rate": round(float(np.mean(action_rate)), 5),
                    "max_impact_weight": round(float(np.max(impact)), 3),
                    "scenario": scenario,
                    "disturbance_produced_fall": initial.get("disturbance_produced_fall"),
                    "passed": bool(final["success"] and initial.get("disturbance_produced_fall", True)),
                    **{f"p95_{key}": float(np.quantile(values, 0.95)) for key, values in quality.items()}}
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
            "scenario": scenario, "success_count": sum(item["passed"] for item in episodes),
            "wilson_95": wilson_interval(sum(item["passed"] for item in episodes), seeds),
            "recovery_level": recovery_level if skill == "recover" else None,
            "pass_rate": round(pass_rate, 3),
            "group_rates": {key: round(value, 3) for key, value in group_rates.items()},
            "criterion_met": pass_rate >= 0.80 and all(value >= 0.75 for value in group_rates.values()),
            "episodes": episodes}


def wilson_interval(successes: int, count: int) -> list[float]:
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial counts")
    z = 1.95996398454
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skill", required=True, choices=("nav", "squat", "recover", "run", "jump"))
    parser.add_argument("--seed-start", type=int, default=10_001)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--recovery-level", type=int,
                        help="Evaluate a starter curriculum level; omitted means full lying recovery")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-seed", type=int, default=10_001)
    parser.add_argument("--scenario", choices=("standard", "settled_floor", "dynamic_fall"), default="standard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    import imageio.v2 as imageio
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .skill_env import MAX_STEPS, RECOVERY_TILT_RANGES, SkillEnv
    from .run_contract import environment_for

    torch.set_num_threads(1)

    if args.recovery_level is not None and (args.skill != "recover" or
                                           args.recovery_level not in range(len(RECOVERY_TILT_RANGES))):
        parser.error("--recovery-level requires --skill recover and a valid curriculum index")

    stats = args.checkpoint.with_name(args.checkpoint.stem + "_vecnormalize.pkl")
    normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: SkillEnv(args.skill)]))
    normalizer.training = False
    model = PPO.load(str(args.checkpoint), device="cpu")
    try:
        result = evaluate(model, normalizer, args.skill, args.seed_start, args.seeds,
                          args.recovery_level, args.scenario)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
        if args.video:
            env = environment_for(model, args.skill, render_mode="rgb_array", randomize=True, recovery_level=args.recovery_level,
                           recovery_sensor_memory=getattr(model, "recovery_sensor_memory", False),
                           recovery_phase_features=getattr(model, "recovery_phase_features", False))
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec="libx264")
            try:
                from .skill_scenarios import reset_case
                observation, _ = reset_case(env, args.video_seed, args.scenario)
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
