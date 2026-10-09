"""Evaluate and render command walking by independent command family."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .command_walk import CONTRACT, FAMILIES, CommandWalkEnv
from .skill_campaign import atomic_json
from .skill_eval import wilson_interval
from .skill_train import _stats_path


def require_contract(model):
    contract = getattr(model, "command_walk_contract", {})
    if contract.get("contract") != CONTRACT or model.observation_space.shape != (85,):
        raise ValueError("Use a command-walking checkpoint with its matching normalization")


def evaluate(model, normalizer, stage=3, seed_start=10051, count=50, families=FAMILIES, progress=None):
    require_contract(model)
    if count < 1 or seed_start < 0:
        raise ValueError("Invalid evaluation seeds/count")
    reports = {}
    for family in families:
        env = CommandWalkEnv(stage, family)
        episodes = []
        try:
            for index in range(count):
                seed = seed_start + FAMILIES.index(family) * 100_000 + index
                obs, _ = env.reset(seed=seed)
                while True:
                    if progress and env.step_count % 100 == 0:
                        progress(family, index, env.step_count)
                    action, _ = model.predict(normalizer.normalize_obs(obs[None]), deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action[0])
                    if terminated or truncated:
                        break
                episodes.append({"seed": seed, "success": info["success"], "failures": info["failures"],
                                 "metrics": info["metrics"], "violation": info["violation"],
                                 "seconds": info["seconds"], "schedule": env.schedule})
        finally:
            env.close()
        passed = sum(e["success"] for e in episodes)
        failures = Counter(key for e in episodes for key in e["failures"])
        keys = {key for e in episodes for key in e["metrics"]}
        reports[family] = {"pass_rate": passed / count, "success_count": passed, "count": count,
                           "wilson_95": wilson_interval(passed, count),
                           "mean_violation": float(np.mean([e["violation"] for e in episodes])),
                           "mean_seconds": float(np.mean([e["seconds"] for e in episodes])),
                           "failure_counts": dict(failures),
                           "metrics": {key: float(np.mean([e['metrics'][key] for e in episodes
                                                          if key in e['metrics']])) for key in sorted(keys)},
                           "episodes": episodes}
    return {"contract": CONTRACT, "stage": stage, "seed_start": seed_start,
            "training_steps": model.num_timesteps, "families": reports,
            "minimum_pass_rate": min(r["pass_rate"] for r in reports.values()),
            "mean_pass_rate": float(np.mean([r["pass_rate"] for r in reports.values()])),
            "mean_violation": float(np.mean([r["mean_violation"] for r in reports.values()])),
            "criterion_met": all(r["pass_rate"] >= .9 for r in reports.values())}


def compact(report):
    return {**report, "families": {k: {a: b for a, b in v.items() if a != "episodes"}
                                    for k, v in report["families"].items()}}


def rank(report):
    return (report["minimum_pass_rate"], report["mean_pass_rate"], -report["mean_violation"])


def progress_made(before, after):
    """Only unmet criteria contribute; improving a satisfied speed bound cannot extend a run."""
    if after["stage"] > before["stage"]:
        return True
    if after["minimum_pass_rate"] >= before["minimum_pass_rate"] + .1 - 1e-9:
        return True
    if after["mean_pass_rate"] >= before["mean_pass_rate"] + .05 - 1e-9:
        return True
    return (before["mean_violation"] > .01 and after["mean_violation"] <= .9 * before["mean_violation"]
            and all(after["families"][k]["mean_seconds"] >= .95 * v["mean_seconds"]
                    for k, v in before["families"].items()))


def render_episode(model, normalizer, output, family, stage, seed):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    require_contract(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = CommandWalkEnv(stage, family, render_mode="rgb_array")
    writer = imageio.get_writer(str(output), fps=25, codec="libx264")
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(normalizer.normalize_obs(obs[None]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            if env.step_count % 2 == 0:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 480, 58), fill="black")
                draw.text((7, 4), f"{family} | simulated policy | t={env.step_count * .02:.1f}s", fill="white")
                draw.text((7, 21), f"command: {info['command_v']:.2f} m/s, {info['command_w']:+.2f} rad/s", fill="white")
                draw.text((7, 38), f"heading error: {np.degrees(info['heading_error']):+.1f} deg; lateral: {info['lateral']:+.2f} m", fill="white")
                writer.append_data(np.asarray(frame))
            if terminated or truncated:
                break
        atomic_json(output.with_suffix(".json"), {"checkpoint_steps": model.num_timesteps, "seed": seed,
                    "family": family, "stage": stage, "success": info["success"], "failures": info["failures"],
                    "metrics": info["metrics"], "schedule": env.schedule, "trace": env.trace})
    finally:
        writer.close()
        env.close()


def load_pair(checkpoint):
    model = PPO.load(checkpoint, device="cpu")
    require_contract(model)
    norm = VecNormalize.load(str(_stats_path(Path(checkpoint))), DummyVecEnv([lambda: CommandWalkEnv()]))
    norm.training = False
    norm.norm_reward = False
    return model, norm


def main():
    import os
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", type=int, default=3, choices=range(4))
    parser.add_argument("--seed-start", type=int, default=10051)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.set_num_threads(1)
    model, norm = load_pair(args.checkpoint)
    try:
        report = evaluate(model, norm, args.stage, args.seed_start, args.seeds,
                          (args.family,) if args.family else FAMILIES)
        atomic_json(args.output, report)
        print({k: v['pass_rate'] for k, v in report['families'].items()}, flush=True)
        if args.video:
            family = args.family or "mixed"
            seed = report['families'][family]['episodes'][0]['seed']
            render_episode(model, norm, args.video, family, args.stage, seed)
    finally:
        norm.close()


if __name__ == "__main__":
    main()
