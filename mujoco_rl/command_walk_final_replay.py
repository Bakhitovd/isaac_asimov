"""Render clearly labelled standalone demonstrations of the complete controller."""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .command_walk import FAMILIES
from .command_walk_final import CASES, FinalEnv
from .command_walk_robust_eval import load_pair
from .skill_campaign import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, norm = load_pair(args.bundle / "policy.zip")
    metadata = json.loads((args.bundle / "controller.json").read_text())
    results = {}
    try:
        failure_seeds = {}
        for report_path in metadata["reports"]:
            report_path = Path(report_path)
            if not report_path.is_absolute():
                report_path = args.bundle / report_path
            report = json.loads(report_path.read_text())
            for case, family in report["families"].items():
                failed = [e["seed"] for e in family["episodes"] if not e["success"]]
                if failed:
                    failure_seeds.setdefault(case, failed[0])
        for case in (*FAMILIES, "brake_forward", "brake_left_arc", "sequence"):
            summary = args.output / f"{case}.json"
            if summary.exists():
                results[case] = json.loads(summary.read_text())
                continue
            env = FinalEnv(case=case, render_mode="rgb_array")
            seed = failure_seeds.get(case, 110_000_001 + CASES.index(case) * 100_000)
            writer = imageio.get_writer(str(args.output / f"{case}.mp4"), fps=10, codec="libx264")
            try:
                obs, _ = env.reset(seed=seed)
                while True:
                    action, _ = model.predict(
                        norm.normalize_obs(obs[None, : model.observation_space.shape[0]]), deterministic=True
                    )
                    obs, _, done, truncated, info = env.step(action[0])
                    if env.step_count % 5 == 0:
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 480, 40), fill="black")
                        draw.text(
                            (5, 5), f"{case} | nominal simulation | qualified={metadata['qualified']}", fill="white"
                        )
                        draw.text(
                            (5, 22),
                            f"{env.step_count*.02:.1f}s | v={info['command_v']:.2f} yaw={info['command_w']:.2f}",
                            fill="white",
                        )
                        writer.append_data(np.asarray(frame))
                    if done or truncated:
                        break
                result = {k: info[k] for k in ("success", "failures", "metrics", "quality")}
                result.update(
                    seed=seed,
                    case=case,
                    standalone_replay=True,
                    selected_from_failed_evaluation=case in failure_seeds,
                    note="Independent scalar-inference demonstration; qualification uses separate batched panels",
                )
                atomic_json(summary, result)
                results[case] = result
            finally:
                writer.close()
                env.close()
        atomic_json(args.output / "status.json", dict(state="completed", heartbeat=time.time(), cases=results))
    finally:
        norm.close()


if __name__ == "__main__":
    main()
