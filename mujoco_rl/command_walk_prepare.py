"""Preserve the baseline and collect only successful, non-evaluation teacher episodes."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_robust import RobustCommandEnv
from .command_walk_robust_eval import load_pair
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path


def prepare(checkpoint, output, episodes=8):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, output / "policy.zip")
    shutil.copy2(_stats_path(checkpoint), output / "policy_vecnormalize.pkl")
    model, norm = load_pair(checkpoint)
    observations, means, accepted, rejected = [], [], [], []
    try:
        for fi, family in enumerate(FAMILIES):
            successes = 0
            env = RobustCommandEnv(level=0, variant="control", family=family)
            try:
                for attempt in range(100):
                    # Disjoint from development (1001) and holdout (10051) seed ranges.
                    seed = 2_000_001 + fi * 100_000 + attempt
                    obs, _ = env.reset(seed=seed)
                    episode_obs, episode_means = [], []
                    while True:
                        with torch.no_grad():
                            tensor = torch.as_tensor(norm.normalize_obs(obs[None, :85]), dtype=torch.float32)
                            mean = model.policy.get_distribution(tensor).distribution.mean.numpy()[0]
                        episode_obs.append(obs.copy())
                        episode_means.append(mean.copy())
                        obs, _, done, truncated, info = env.step(np.clip(mean, -1, 1))
                        if done or truncated:
                            break
                    record = dict(family=family, seed=seed, failures=info["failures"], metrics=info["metrics"])
                    if info["success"]:
                        # Equal frames per episode and equal accepted episodes per family.
                        indices = np.linspace(0, len(episode_obs) - 1, 128, dtype=int)
                        observations.extend(np.asarray(episode_obs)[indices])
                        means.extend(np.asarray(episode_means)[indices])
                        accepted.append(record)
                        successes += 1
                    else:
                        rejected.append(record)
                    print(f"[reference] {family}: {successes}/{episodes} successful episodes", flush=True)
                    if successes == episodes:
                        break
                if successes != episodes:
                    raise RuntimeError(f"Insufficient successful reference cases: {family}")
            finally:
                env.close()
        path = output / "reference.npz"
        np.savez_compressed(
            path,
            observations=np.asarray(observations),
            means=np.asarray(means),
            log_std=model.policy.log_std.detach().numpy(),
        )
        atomic_json(
            output / "manifest.json",
            dict(
                source=str(checkpoint.resolve()),
                policy_sha256=sha256(checkpoint),
                normalization_sha256=sha256(_stats_path(checkpoint)),
                reference_sha256=sha256(path),
                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                accepted=accepted,
                rejected=rejected,
                frames=len(observations),
                note="Only successful teacher episodes; distinct from development and holdout seeds",
            ),
        )
    finally:
        norm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    prepare(args.checkpoint, args.output, args.episodes)
