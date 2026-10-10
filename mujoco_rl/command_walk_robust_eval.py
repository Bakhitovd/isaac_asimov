"""Batched inference on fixed, profile-independent command cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .command_walk import FAMILIES, CommandWalkEnv
from .command_walk_robust import CONTRACT, PROFILES, RobustCommandEnv
from .skill_campaign import atomic_json
from .skill_eval import wilson_interval
from .skill_train import _stats_path


def load_pair(checkpoint):
    model = PPO.load(checkpoint, device="cpu")
    size = model.observation_space.shape[0]
    if size not in (85, 87):
        raise ValueError("Expected a versioned v1 or v2 command policy")
    expected = "asimov_command_walk_v1" if size == 85 else CONTRACT
    if model.command_walk_contract.get("contract") != expected:
        raise ValueError("Incompatible policy contract")
    env = DummyVecEnv([lambda: CommandWalkEnv() if size == 85 else RobustCommandEnv()])
    norm = VecNormalize.load(str(_stats_path(Path(checkpoint))), env)
    norm.training, norm.norm_reward = False, False
    return model, norm


def summarize(episodes):
    reports = {}
    for family, cases in episodes.items():
        passed = sum(e["success"] for e in cases)
        failures = Counter(k for e in cases for k in e["failures"])
        keys = set().union(*(e["metrics"] for e in cases))
        reports[family] = dict(
            pass_rate=passed / len(cases),
            success_count=passed,
            count=len(cases),
            wilson_95=wilson_interval(passed, len(cases)),
            failure_counts=dict(failures),
            mean_violation=float(np.mean([e["violation"] for e in cases])),
            mean_seconds=float(np.mean([e["seconds"] for e in cases])),
            episodes=cases,
            metrics={k: float(np.mean([e["metrics"][k] for e in cases if k in e["metrics"]])) for k in keys},
        )
    return dict(
        families=reports,
        minimum_pass_rate=min(v["pass_rate"] for v in reports.values()),
        mean_pass_rate=float(np.mean([v["pass_rate"] for v in reports.values()])),
        mean_violation=float(np.mean([v["mean_violation"] for v in reports.values()])),
        criterion_met=all(v["pass_rate"] >= 0.9 for v in reports.values()),
    )


def evaluate(model, normalizer, level=0, seed_start=1001, count=50, batch=10, progress=None, families=FAMILIES):
    if count < 1 or batch < 1 or level not in range(len(PROFILES)):
        raise ValueError("Invalid evaluation configuration")
    episodes = {k: [] for k in families}
    size = model.observation_space.shape[0]
    variant = model.command_walk_contract.get("variant", "control")
    for family in families:
        for offset in range(0, count, batch):
            envs, observations, seeds = [], [], []
            try:
                for index in range(offset, min(count, offset + batch)):
                    seed = seed_start + FAMILIES.index(family) * 100_000 + index
                    env = RobustCommandEnv(level=level, variant=variant, family=family)
                    envs.append(env)
                    obs, _ = env.reset(seed=seed)
                    observations.append(obs)
                    seeds.append(seed)
                active = list(range(len(envs)))
                while active:
                    actions, _ = model.predict(
                        normalizer.normalize_obs(np.array([observations[i][:size] for i in active])), deterministic=True
                    )
                    remaining = []
                    for i, action in zip(active, actions):
                        env = envs[i]
                        if progress and env.step_count % 100 == 0:
                            progress(family, offset + i, env.step_count)
                        observations[i], _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            episodes[family].append(
                                {k: info[k] for k in ["success", "failures", "metrics", "violation", "seconds"]}
                                | {
                                    "seed": seeds[i],
                                    "schedule": env.schedule,
                                    "action_lag": env.action_lag,
                                    "sensor_lag": env.sensor_lag,
                                }
                            )
                        else:
                            remaining.append(i)
                    active = remaining
            finally:
                for env in envs:
                    env.close()
        episodes[family].sort(key=lambda e: e["seed"])
    return dict(
        contract=CONTRACT,
        level=level,
        profile=PROFILES[level].name,
        seed_start=seed_start,
        training_steps=model.num_timesteps,
        **summarize(episodes),
    )


def _evaluate_family(job):
    checkpoint, level, seed_start, count, family = job
    torch.set_num_threads(1)
    model, norm = load_pair(checkpoint)
    try:
        return evaluate(model, norm, level, seed_start, count, families=(family,))
    finally:
        norm.close()


def parallel_evaluate(checkpoint, level=0, seed_start=1001, count=50, workers=4, progress=None):
    """Independent CPU workers, each with batched inference and isolated MuJoCo models."""
    import multiprocessing as mp
    import time

    pool = mp.get_context("spawn").Pool(workers)
    try:
        jobs = [(checkpoint, level, seed_start, count, family) for family in FAMILIES]
        pending = pool.map_async(_evaluate_family, jobs)
        while not pending.ready():
            if progress:
                progress("parallel_panel", 0, 0)
            time.sleep(0.5)
        reports = pending.get()
        episodes = {k: v["episodes"] for r in reports for k, v in r["families"].items()}
        return dict(
            contract=CONTRACT,
            level=level,
            profile=PROFILES[level].name,
            seed_start=seed_start,
            training_steps=reports[0]["training_steps"],
            **summarize(episodes),
        )
    finally:
        pool.terminate()
        pool.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=10051)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, norm = load_pair(args.checkpoint)
    try:
        report = evaluate(model, norm, args.level, args.seed_start, args.seeds)
        atomic_json(args.output, report)
        print({k: v["pass_rate"] for k, v in report["families"].items()}, flush=True)
    finally:
        norm.close()


if __name__ == "__main__":
    main()
