"""Replay baseline successes and failures with physical traces and annotated videos."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .command_walk_robust import RobustCommandEnv
from .command_walk_robust_eval import load_pair
from .skill_campaign import atomic_json


def audit(checkpoint, panel, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads(panel.read_text())
    report = report.get("nominal", report)
    atomic_json(
        output / "failures.json",
        {
            f: dict(pass_rate=v["pass_rate"], failures=v["failure_counts"], metrics=v.get("metrics", {}))
            for f, v in report["families"].items()
        },
    )
    model, norm = load_pair(checkpoint)
    try:
        for family in ("left_pivot", "mixed"):
            for successful in (True, False):
                episode = next(e for e in report["families"][family]["episodes"] if e["success"] == successful)
                label = f'{family}_{"success" if successful else "failure"}_{episode["seed"]}'
                env = RobustCommandEnv(level=0, variant="control", family=family, render_mode="rgb_array")
                writer = imageio.get_writer(str(output / f"{label}.mp4"), fps=25, codec="libx264")
                joint_rows = []
                try:
                    obs, _ = env.reset(seed=episode["seed"])
                    while True:
                        action, _ = model.predict(
                            norm.normalize_obs(obs[None, : model.observation_space.shape[0]]), deterministic=True
                        )
                        obs, _, done, truncated, info = env.step(action[0])
                        joint_rows.append(
                            dict(
                                position=env.data.qpos[env.qpos_ids].tolist(),
                                velocity=env.data.qvel[env.qvel_ids].tolist(),
                                action=action[0].tolist(),
                            )
                        )
                        if env.step_count % 2 == 0:
                            frame = Image.fromarray(env.render())
                            draw = ImageDraw.Draw(frame)
                            draw.rectangle((0, 0, 480, 62), fill="black")
                            draw.text((5, 3), f"{label} | nominal simulation", fill="white")
                            draw.text(
                                (5, 22),
                                f't={env.step_count*.02:.2f}s v={info["command_v"]:.2f} yaw={info["command_w"]:+.2f}',
                                fill="white",
                            )
                            draw.text(
                                (5, 41),
                                f'heading={np.degrees(info["heading_error"]):+.1f}deg speed={info["speed"]:.3f}',
                                fill="white",
                            )
                            writer.append_data(np.asarray(frame))
                        if done or truncated:
                            break
                    if info["success"] != successful:
                        raise RuntimeError("Replay differs from source panel")
                    atomic_json(
                        output / f"{label}.json",
                        dict(
                            seed=episode["seed"],
                            family=family,
                            success=info["success"],
                            failures=info["failures"],
                            metrics=info["metrics"],
                            trace=env.trace,
                            joints=joint_rows,
                        ),
                    )
                    print(label, info["failures"], flush=True)
                finally:
                    writer.close()
                    env.close()
    finally:
        norm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.checkpoint, args.panel, args.output)
