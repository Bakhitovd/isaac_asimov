"""Replay a paired correction case and plot its unchanged targets and physical motion."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from .command_walk_correction import BASELINE, CommandCorrection, Correction, fingerprint
from .command_walk_robust_eval import load_pair
from .command_walk_v4 import CASES, MotionEnv
from .skill_campaign import atomic_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def replay(root, case, seed=None, panel="development"):
    reports = {label: json.loads((root / panel / f"{label}.json").read_text()) for label in ("baseline", "corrected")}
    indexed = {label: {e["seed"]: e for e in r["families"][case]["episodes"]} for label, r in reports.items()}
    if seed is None:
        improved = [
            s for s, e in indexed["baseline"].items() if not e["success"] and indexed["corrected"][s]["success"]
        ]
        seed = improved[0] if improved else next(iter(indexed["baseline"]))
    ordered_seeds = sorted(indexed["baseline"])
    offset = ordered_seeds.index(seed) // 10 * 10
    batch_seeds = ordered_seeds[offset : offset + 10]
    target = batch_seeds.index(seed)
    selected = Correction(**json.loads((root / "selection.json").read_text())["configuration"])
    output = root / "replays" / f"{panel}_{case}_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    model, norm = load_pair(root / "baseline" / "policy.zip")
    before = fingerprint(model, norm)
    traces = {}
    try:
        for label, config in (("baseline", BASELINE), ("corrected", selected)):
            # Preserve the original inference batch, including peers terminating early.
            # Tiny scalar-versus-batch float differences can amplify through contacts.
            envs = [MotionEnv(case=case, render_mode="rgb_array" if s == seed else None) for s in batch_seeds]
            controllers = [CommandCorrection(config) for _ in batch_seeds]
            env = envs[target]
            rows = []
            writer = imageio.get_writer(str(output / f"{label}.mp4"), fps=25, codec="libx264")
            try:
                observations = [e.reset(seed=s)[0] for e, s in zip(envs, batch_seeds)]
                active = list(range(len(envs)))
                while True:
                    controls = {i: controllers[i].apply(observations[i]) for i in active}
                    modified, correction, limited = controls[target]
                    actions, _ = model.predict(
                        norm.normalize_obs(
                            np.array([controls[i][0][: model.observation_space.shape[0]] for i in active])
                        ),
                        deterministic=True,
                    )
                    remaining = []
                    for i, action in zip(active, actions):
                        observations[i], _, ended, expired, row = envs[i].step(action)
                        if i == target:
                            done, truncated, info = ended, expired, row
                        if not (ended or expired):
                            remaining.append(i)
                    active = remaining
                    rows.append(
                        dict(
                            env.trace[-1],
                            control_v=float(modified[6]),
                            control_w=float(modified[78]),
                            correction=correction.tolist(),
                            limited=limited,
                        )
                    )
                    if env.step_count % 2 == 0:
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 480, 64), fill="black")
                        draw.text((5, 3), f"{label} | {case} | simulated | {env.step_count*.02:.1f}s", fill="white")
                        draw.text(
                            (5, 23),
                            f"yaw requested={info['command_w']:+.2f}, policy input={modified[78]:+.2f}",
                            fill="white",
                        )
                        draw.text((5, 43), f"heading error={np.degrees(info['heading_error']):+.1f} deg", fill="white")
                        writer.append_data(np.asarray(frame))
                    if done or truncated:
                        break
                expected = indexed[label][seed]
                if (
                    info["success"] != expected["success"]
                    or info["failures"] != expected["failures"]
                    or any(
                        not np.isclose(info["metrics"][k], v, atol=1e-5, rtol=1e-5)
                        for k, v in expected["metrics"].items()
                    )
                ):
                    raise RuntimeError("Replay differs from paired evaluation")
                atomic_json(
                    output / f"{label}.json",
                    dict(
                        case=case,
                        seed=seed,
                        correction=asdict(config),
                        success=info["success"],
                        failures=info["failures"],
                        metrics=info["metrics"],
                        trace=rows,
                    ),
                )
                traces[label] = rows
                print(label, case, seed, info["success"], info["failures"], flush=True)
            finally:
                writer.close()
                for e in envs:
                    e.close()
        if fingerprint(model, norm) != before:
            raise RuntimeError("Replay modified frozen model/statistics")
    finally:
        norm.close()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for label, rows in traces.items():
        t = np.arange(1, len(rows) + 1) * 0.02
        for ax, field, scale in zip(axes, ("heading_error", "yaw_rate", "speed"), (180 / np.pi, 1, 1)):
            ax.plot(t, np.array([r[field] for r in rows]) * scale, label=label, linewidth=1.1)
    rows = traces["baseline"]
    t = np.arange(1, len(rows) + 1) * 0.02
    axes[0].axhline(10, color="gray", linestyle=":", label="10 deg RMS limit (reference)")
    axes[0].axhline(-10, color="gray", linestyle=":")
    axes[1].plot(t, [r["command_w"] for r in rows], color="black", linestyle="--", label="original requested yaw rate")
    axes[2].plot(t, [r["command_v"] for r in rows], color="black", linestyle="--", label="original forward command")
    for ax, name in zip(axes, ("Heading error (deg)", "Yaw rate (rad/s)", "Horizontal speed (m/s)")):
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(f"{case}, seed {seed}: same policy and trajectory; bounded heading correction")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--case", choices=CASES, default="left_pivot")
    p.add_argument("--panel", choices=("development", "confirmation", "sequences"), default="development")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.set_num_threads(1)
    replay(args.root, args.case, args.seed, args.panel)
