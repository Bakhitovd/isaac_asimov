"""Nominal holdout/long-sequence qualification and reproducible release artifacts."""

import json
import shutil
from pathlib import Path

import numpy as np

from .command_walk import FAMILIES
from .command_walk_robust_eval import load_pair
from .command_walk_v4 import MotionEnv, panel, passes
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path


def replay(model, norm, case, episode, output, progress):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    env = MotionEnv(case=case, render_mode="rgb_array")
    writer = imageio.get_writer(str(output), fps=25, codec="libx264")
    try:
        obs, _ = env.reset(seed=episode["seed"])
        while True:
            if env.step_count % 100 == 0:
                progress()
            action, _ = model.predict(
                norm.normalize_obs(obs[None, : model.observation_space.shape[0]]), deterministic=True
            )
            obs, _, done, truncated, info = env.step(action[0])
            if env.step_count % 2 == 0:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 480, 58), fill="black")
                draw.text((5, 4), f"{case} | nominal simulation | {env.step_count*.02:.1f}s", fill="white")
                draw.text((5, 22), f'v={info["command_v"]:.2f} yaw={info["command_w"]:+.2f}', fill="white")
                draw.text((5, 40), f'heading error={np.degrees(info["heading_error"]):+.1f} deg', fill="white")
                writer.append_data(np.asarray(frame))
            if done or truncated:
                break
        if info["success"] != episode["success"]:
            raise RuntimeError("Replay result differs from qualification")
        atomic_json(
            output.with_suffix(".json"),
            dict(
                case=case,
                seed=episode["seed"],
                success=info["success"],
                failures=info["failures"],
                metrics=info["metrics"],
                trace=env.trace,
            ),
        )
    finally:
        writer.close()
        env.close()


def qualify(checkpoint, output, progress):
    output.mkdir(parents=True, exist_ok=False)
    model, norm = load_pair(checkpoint)
    reports = {}
    try:
        for label, cases, count, seed in (
            ("development", FAMILIES, 50, 1001),
            ("holdout", FAMILIES, 50, 10051),
            ("sequence_development", ("sequence",), 20, 1001),
            ("sequence_holdout", ("sequence",), 20, 10051),
        ):
            report = panel(model, norm, cases, count, seed, progress=lambda *_: progress())
            reports[label] = report
            atomic_json(output / f"{label}.json", report)
        passed = all(
            passes(r, r["families"], 0.95 if "sequence" in label else 0.96, 20 if "sequence" in label else 50)
            for label, r in reports.items()
        )
        atomic_json(
            output / "result.json",
            dict(
                passed=passed,
                scope="nominal simulation only",
                checkpoint=str(checkpoint),
                policy_sha256=sha256(checkpoint),
            ),
        )
        if not passed:
            return False
        for case in (*FAMILIES, "sequence"):
            report = reports["sequence_holdout" if case == "sequence" else "holdout"]
            episode = next(e for e in report["families"][case]["episodes"] if e["success"])
            replay(model, norm, case, episode, output / f"{case}.mp4", progress)
        failures = [
            (case, e)
            for label, r in reports.items()
            for case, v in r["families"].items()
            for e in v["episodes"]
            if not e["success"]
        ]
        for i, (case, e) in enumerate(failures[:2]):
            replay(model, norm, case, e, output / f"failure_{i}_{case}.mp4", progress)
        shutil.copy2(checkpoint, output / "policy.zip")
        shutil.copy2(_stats_path(checkpoint), output / "policy_vecnormalize.pkl")
        atomic_json(output / "contract.json", model.command_walk_contract)
        atomic_json(output / "manifest.json", {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
        return True
    finally:
        norm.close()
