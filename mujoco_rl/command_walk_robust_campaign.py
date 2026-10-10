"""Compare paired v2 pilots, extend only retained improvements, then qualify."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .command_walk import FAMILIES
from .command_walk_eval import compact, rank
from .command_walk_robust import CHALLENGE_LEVEL, PROFILES, RobustCommandEnv
from .command_walk_robust_eval import evaluate, load_pair, parallel_evaluate
from .command_walk_robust_train import retention_passed
from .skill_campaign import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def qualifies_for_extension(reference, nominal, challenge):
    retained = retention_passed(reference["nominal"], nominal)
    old = reference["challenge"]
    improved = (
        challenge["minimum_pass_rate"] >= old["minimum_pass_rate"] + 0.1 - 1e-9
        or challenge["mean_pass_rate"] >= old["mean_pass_rate"] + 0.05 - 1e-9
        or (old["mean_violation"] > 1e-6 and challenge["mean_violation"] < 0.9 * old["mean_violation"])
    )
    return retained and improved


def render(model, norm, checkpoint, output, family, seed):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    env = RobustCommandEnv(
        level=len(PROFILES) - 1, variant=model.command_walk_contract["variant"], family=family, render_mode="rgb_array"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=25, codec="libx264")
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            obs, _, done, truncated, info = env.step(action[0])
            if env.step_count % 2 == 0:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 480, 58), fill="black")
                draw.text((6, 4), f"{family} | simulation | {env.step_count*.02:.1f}s", fill="white")
                draw.text(
                    (6, 21), f'command {info["command_v"]:.2f} m/s, yaw {info["command_w"]:+.2f} rad/s', fill="white"
                )
                draw.text((6, 38), f'heading error {np.degrees(info["heading_error"]):+.1f} deg', fill="white")
                writer.append_data(np.asarray(frame))
            if done or truncated:
                break
        if not info["success"]:
            raise RuntimeError("Qualified replay did not reproduce its result")
        atomic_json(
            output.with_suffix(".json"),
            dict(
                checkpoint=str(checkpoint), seed=seed, trace=env.trace, success=info["success"], metrics=info["metrics"]
            ),
        )
    finally:
        writer.close()
        env.close()


class Campaign:
    def __init__(self, args):
        self.args, self.root = args, args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.deadline = self.started + args.hours * 3600
        self.cancelled = False
        self.processes = []
        self.status = {"state": "selecting_source", "started": self.started}
        self.last_status = 0.0
        self.environment = dict(
            os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="osmesa"
        )
        mesa = Path.home() / ".local/mujoco-osmesa/usr/lib/x86_64-linux-gnu"
        if mesa.exists():
            self.environment["LD_LIBRARY_PATH"] = str(mesa) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        snapshot = self.root / "source_snapshot"
        snapshot.mkdir()
        for f in (ROOT / "mujoco_rl").glob("*.py"):
            shutil.copy2(f, snapshot / f.name)
        atomic_json(
            self.root / "arguments.json",
            {
                k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
                for k, v in vars(args).items()
            },
        )

    def write(self, **kwargs):
        self.status.update(kwargs, heartbeat=time.time(), elapsed_hours=(time.time() - self.started) / 3600)
        atomic_json(self.root / "status.json", self.status)
        self.last_status = time.time()

    def stop(self, *_):
        self.cancelled = True
        for p in self.processes:
            if p.poll() is None:
                p.terminate()

    def progress(self, family="", index=0, step=0):
        if self.cancelled or time.time() >= self.deadline:
            raise TimeoutError("interrupted" if self.cancelled else "time_budget")
        if time.time() - self.last_status > 10:
            self.write(family=family, episode=index, episode_step=step)
            print(f"[evaluation] {family} case={index+1} step={step}", flush=True)

    def panel(self, checkpoint, level, seed_start=1001):
        return parallel_evaluate(checkpoint, level, seed_start, self.args.cases, progress=self.progress)

    def jobs(self, jobs, steps, pilot):
        # Two simultaneous eight-environment runs leave CPU capacity for inference/logging.
        results = []
        for offset in range(0, len(jobs), 2):
            group = []
            for name, checkpoint, variant, seed in jobs[offset : offset + 2]:
                self.progress()
                output = self.root / name
                handle = (self.root / f"{name}.log").open("w")
                hours = min(12.0, max(0.01, (self.deadline - time.time()) / 3600 - min(1.0, self.args.hours / 10)))
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mujoco_rl.command_walk_robust_train",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(steps),
                    "--hours",
                    str(hours),
                    "--level",
                    "2",
                    "--full-count",
                    str(self.args.cases),
                ]
                if pilot:
                    command.append("--pilot")
                process = subprocess.Popen(
                    command, cwd=ROOT, env=self.environment, stdout=handle, stderr=subprocess.STDOUT
                )
                self.processes.append(process)
                group.append((name, output, process, handle))
                print(f"[launch] {name} pid={process.pid}", flush=True)
            try:
                while any(p.poll() is None for _, _, p, _ in group):
                    self.progress()
                    current = []
                    for name, output, p, _ in group:
                        f = output / "status.json"
                        current.append(
                            dict(
                                name=name, exit_code=p.poll(), training=json.loads(f.read_text()) if f.exists() else {}
                            )
                        )
                    self.write(jobs=current)
                    time.sleep(5)
                for name, output, p, _ in group:
                    if p.wait() != 0:
                        raise RuntimeError(f"{name} failed; see {name}.log")
                    report = json.loads((output / "report.json").read_text())
                    results.append((name, report))
            finally:
                for _, _, p, h in group:
                    if p.poll() is None:
                        p.terminate()
                        try:
                            p.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait()
                    h.close()
        return results

    def run(self):
        baselines = []
        for index, checkpoint in enumerate(self.args.checkpoints):
            self.write(source=str(checkpoint))
            nominal = self.panel(checkpoint, 0)
            challenge = self.panel(checkpoint, CHALLENGE_LEVEL)
            report = dict(checkpoint=str(checkpoint.resolve()), nominal=nominal, challenge=challenge)
            atomic_json(self.root / f"source_{index}.json", report)
            baselines.append(report)
        # Protect the weakest nominal family first; challenge breaks nominal ties.
        baseline = max(baselines, key=lambda b: (rank(b["nominal"]), rank(b["challenge"])))
        source = Path(baseline["checkpoint"])
        self.write(state="pilots", selected_source=str(source))
        jobs = [
            (f"pilot_{variant}_{seed}", source, variant, seed)
            for seed in (44, 45)
            for variant in ("control", "aligned")
        ]
        pilot_reports = self.jobs(jobs, self.args.pilot_steps, True)
        self.write(state="comparing_pilots")
        eligible = []
        comparison = []
        for name, report in pilot_reports:
            if report["final"] is None:
                continue
            checkpoint = Path(report["checkpoint"])
            nominal = report["final"]["nominal"]
            challenge = self.panel(checkpoint, CHALLENGE_LEVEL)
            entry = dict(
                name=name,
                checkpoint=str(checkpoint),
                nominal=nominal,
                challenge=challenge,
                eligible=qualifies_for_extension(baseline, nominal, challenge),
            )
            comparison.append(entry)
            if entry["eligible"]:
                eligible.append(entry)
        atomic_json(self.root / "pilot_comparison.json", dict(baseline=baseline, trials=comparison))
        # Require replication: both independent seeds must improve and retain behavior.
        variants = [v for v in ("control", "aligned") if sum(f"_{v}_" in e["name"] for e in eligible) == 2]
        if not variants:
            self.write(state="review_required", reason="No pilot variant improved with retention in both seeds")
            return
        winner = max(
            variants, key=lambda v: sum(e["challenge"]["mean_pass_rate"] for e in eligible if f"_{v}_" in e["name"])
        )
        selected = [e for e in eligible if f"_{winner}_" in e["name"]]
        self.write(state="extending", selected_variant=winner)
        jobs = [
            (f"extended_{winner}_{seed}", Path(e["checkpoint"]), winner, seed) for seed, e in zip((44, 45), selected)
        ]
        extended = self.jobs(jobs, self.args.extension_steps, False)
        candidates = []
        for name, report in extended:
            best = report["best_by_level"].get(str(len(PROFILES) - 1))
            if best and best["report"]["current"]["criterion_met"] and best["report"]["nominal"]["criterion_met"]:
                candidates.append(best)
        candidates.sort(key=lambda e: rank(e["report"]["current"]), reverse=True)
        self.write(state="qualifying")
        failed_seeds = []
        for attempt, best in enumerate(candidates):
            checkpoint = Path(best["checkpoint"])
            level = len(PROFILES) - 1
            regressions = [
                {"full": self.panel(checkpoint, level, s), "nominal": self.panel(checkpoint, 0, s)}
                for s in failed_seeds
            ]
            if any(not r["full"]["criterion_met"] or not r["nominal"]["criterion_met"] for r in regressions):
                continue
            seed = 10051 + attempt * 1_000_000
            holdout = self.panel(checkpoint, level, seed)
            nominal = self.panel(checkpoint, 0, seed)
            qualification = dict(
                checkpoint=str(checkpoint),
                development=best["report"],
                holdout=holdout,
                nominal_holdout=nominal,
                regressions=regressions,
            )
            atomic_json(self.root / f"qualification_{attempt}.json", qualification)
            if not holdout["criterion_met"] or not nominal["criterion_met"]:
                failed_seeds.append(seed)
                continue
            if self.args.cases < 50:
                raise RuntimeError("Smoke evaluations cannot qualify a policy")
            accepted = self.root / "accepted"
            accepted.mkdir()
            shutil.copy2(checkpoint, accepted / "accepted.zip")
            shutil.copy2(checkpoint.with_name("policy_vecnormalize.pkl"), accepted / "accepted_vecnormalize.pkl")
            model, norm = load_pair(checkpoint)
            try:
                for family in FAMILIES:
                    self.progress()
                    case = next(e for e in holdout["families"][family]["episodes"] if e["success"])
                    render(model, norm, checkpoint, accepted / "demos" / f"{family}.mp4", family, case["seed"])
            finally:
                norm.close()
            qualification["sha256"] = {
                str(f.relative_to(accepted)): sha256(f) for f in accepted.rglob("*") if f.is_file()
            }
            atomic_json(accepted / "qualification.json", qualification)
            if self.args.publish:
                branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
                staged = subprocess.check_output(
                    ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
                ).strip()
                if branch != "main" or staged:
                    raise RuntimeError("Publication requires main and an empty Git index")
                destination = ROOT / "mujoco_rl/checkpoints/command_walk_v2"
                shutil.copytree(accepted, destination)
                (destination / "README.md").write_text(
                    "Command walking v2: 23 actions, 87 observations. Keep the policy and normalization pair together.\nSee qualification.json and demos/ for simulation evidence.\n"
                )
                subprocess.run(["git", "add", "--", str(destination.relative_to(ROOT))], cwd=ROOT, check=True)
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=Codex",
                        "-c",
                        "user.email=codex@openai.com",
                        "commit",
                        "-m",
                        "Publish qualified command walking v2 policy and demos",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                subprocess.run(["git", "push", "fork", "main"], cwd=ROOT, check=True)
            self.write(state="qualified", accepted=str(accepted), published=self.args.publish)
            return
        self.write(state="review_required", reason="No policy passed full physics and nominal holdout qualification")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--pilot-steps", type=int, default=500_000)
    p.add_argument("--extension-steps", type=int, default=6_000_000)
    p.add_argument("--hours", type=float, default=48.0)
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    if (
        args.cases < 1
        or not math.isfinite(args.hours)
        or not 0 < args.hours <= 72
        or args.pilot_steps < 0
        or args.extension_steps < 0
        or (args.publish and args.cases < 50)
    ):
        p.error("Invalid campaign budget")
    torch.set_num_threads(1)
    campaign = Campaign(args)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, campaign.stop)
    try:
        campaign.run()
    except TimeoutError as error:
        campaign.stop()
        campaign.write(state="stopped", reason=str(error))
    except Exception as error:
        campaign.stop()
        campaign.write(state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
