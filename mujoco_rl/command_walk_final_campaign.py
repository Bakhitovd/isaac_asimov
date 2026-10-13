"""Resumable 48-hour moderate walking campaign with empirical promotion gates."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from .command_walk import FAMILIES
from .command_walk_final import BRAKES, CASES, TARGETS, contract
from .command_walk_final_eval import bundle, panel, passes, quality_gate, rank, retain
from .command_walk_final_train import reference, train_block
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path

PILOT_TARGETS = (*TARGETS, "sequence")
REGRESSION_CASES = tuple(dict.fromkeys((*FAMILIES, *PILOT_TARGETS)))


def read(path):
    return json.loads(Path(path).read_text())


def subset(report, cases):
    families = {k: report["families"][k] for k in cases}
    return dict(
        families=families,
        minimum_pass_rate=min(v["pass_rate"] for v in families.values()),
        mean_pass_rate=float(np.mean([v["pass_rate"] for v in families.values()])),
        mean_violation=float(np.mean([v["mean_violation"] for v in families.values()])),
    )


def task(job):
    output = Path(job[1])
    if (output / "block.json").exists():
        return read(output / "block.json")
    return train_block(job)


class Campaign:
    def __init__(self, args):
        self.root = args.output.resolve()
        if args.resume:
            self.manifest = read(self.root / "manifest.json")
            self.state = read(self.root / "state.json")
            if self.manifest["contract"] != contract():
                raise ValueError("Resume contract mismatch")
            for name, digest in self.manifest["source_hashes"].items():
                if sha256(Path(__file__).parent / name) != digest:
                    raise ValueError(f"Resume source changed: {name}; restore the recorded source snapshot")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "baseline").mkdir()
            baseline = self.root / "baseline/policy.zip"
            shutil.copy2(args.checkpoint, baseline)
            shutil.copy2(_stats_path(args.checkpoint), _stats_path(baseline))
            snapshot = self.root / "source_snapshot"
            snapshot.mkdir()
            for p in Path(__file__).parent.glob("*.py"):
                shutil.copy2(p, snapshot / p.name)
            started = time.time()
            self.manifest = dict(
                contract=contract(),
                started=started,
                deadline=started + 48 * 3600,
                stage_deadlines={
                    k: started + h * 3600
                    for k, h in [("setup", 2), ("pilot", 8), ("motion", 22), ("sequence", 40), ("qualification", 48)]
                },
                source=str(args.checkpoint.resolve()),
                source_policy_sha256=sha256(args.checkpoint),
                normalization_sha256=sha256(_stats_path(args.checkpoint)),
                revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                source_hashes={p.name: sha256(p) for p in snapshot.glob("*.py")},
                holdouts=dict(families=90_000_001, sequence_a=94_000_001, sequence_b=98_000_001, grid=102_000_001),
                training_seeds=[44, 45],
                reference_seed_base=70_000_001,
                development_seed=1001,
                regression_confirmation_seed=10051,
                regression_recheck_seed=5001,
                wall_budget_hours=48,
                auto_publish=False,
                resume_semantics="Restore policy, optimizer, statistics and curriculum; reset environments at rollout boundary",
            )
            atomic_json(self.root / "manifest.json", self.manifest)
            (self.root / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff"]))
            self.state = dict(
                stage="setup",
                arms={},
                round=0,
                sequence_seconds=20,
                best=str(baseline),
                stopped=False,
                qualified=False,
                holdouts_opened=False,
            )
            self.save()
        self.baseline = self.root / "baseline/policy.zip"
        self.anchor = self.root / "reference.npz"

    def save(self):
        atomic_json(self.root / "state.json", self.state)

    def status(self, stage, **extra):
        atomic_json(
            self.root / "status.json",
            dict(
                state="running",
                stage=stage,
                heartbeat=time.time(),
                elapsed_hours=(time.time() - self.manifest["started"]) / 3600,
                **extra,
            ),
        )

    def evaluate(
        self, checkpoint, label, cases=REGRESSION_CASES, count=50, start=1001, seconds=60, stage=None, grid=False
    ):
        self.status("evaluation", evaluation=label, checkpoint=str(checkpoint))
        deadline = self.manifest["stage_deadlines"][stage or self.state["stage"]]
        return panel(
            checkpoint, self.root / "evaluations" / f"{label}.json", cases, count, start, seconds, deadline, grid
        )

    def train(self, jobs, stage):
        results = {}
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            pending = {pool.submit(task, job): name for name, job in jobs.items()}
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for f in done:
                    name = pending.pop(f)
                    results[name] = f.result()
                progress = {}
                for name, job in jobs.items():
                    status_path = Path(job[1]) / "status.json"
                    if status_path.exists():
                        progress[name] = read(status_path)
                self.status(stage, completed=len(results), total=len(jobs), jobs=progress)
                print(f"[{stage}] {len(results)}/{len(jobs)} jobs complete; " + str(progress), flush=True)
        return results

    def make_job(self, name, checkpoint, reward, seed, rollouts, fresh, phase="motion", seconds=60):
        label = f"{self.state['stage']}_{seconds}_{self.state['round']:04d}_{name}"
        return (
            str(checkpoint),
            str(self.root / "jobs" / label),
            str(self.anchor),
            seed,
            reward,
            phase,
            seconds,
            rollouts,
            fresh,
            self.manifest["stage_deadlines"][self.state["stage"]],
        )

    def stop(self, reason):
        self.state.update(stage="delivery", stopped=True, stop_reason=reason)
        self.save()

    def setup(self):
        baseline = self.evaluate(self.baseline, "baseline_development")
        confirmation = self.evaluate(self.baseline, "baseline_regression", start=10051)
        # These are development checks. Locked final holdouts are not touched here.
        sequences = [
            self.evaluate(self.baseline, f"baseline_sequence_{i}", cases=("sequence",), start=start)
            for i, start in enumerate((24_000_001, 28_000_001))
        ]
        grid = self.evaluate(self.baseline, "baseline_grid", count=20, start=32_000_001, grid=True)
        atomic_json(
            self.root / "failure_cases.json",
            {
                panel_name: {
                    case: [e for e in v["episodes"] if not e["success"]] for case, v in report["families"].items()
                }
                for panel_name, report in [("development", baseline), ("regression", confirmation)]
            },
        )
        if passes(baseline) and passes(confirmation) and all(passes(s) for s in sequences) and passes(grid, 0.95):
            self.state["stage"] = "qualification"
            self.save()
            return
        self.status("reference_collection")
        reference(self.baseline, self.anchor, self.manifest["stage_deadlines"]["setup"])
        self.state["stage"] = "pilot"
        self.save()

    def pilot(self):
        arms = self.state["arms"]
        if not arms:
            for reward in ("original", "aligned"):
                for seed in (44, 45):
                    name = f"{reward}_{seed}"
                    arms[name] = dict(checkpoint=str(self.baseline), reward=reward, seed=seed)
            self.save()
        # Two blocks give exact 16- and 64-rollout physical evaluations.
        for round_index, rollouts in enumerate((16, 48)):
            if self.state["round"] > round_index:
                continue
            jobs = {
                name: self.make_job(name, a["checkpoint"], a["reward"], a["seed"], rollouts, round_index == 0)
                for name, a in arms.items()
            }
            results = self.train(jobs, "pilot")
            for name, result in results.items():
                a = arms[name]
                a["checkpoint"] = result["checkpoint"]
                a["technical_passed"] = a.get("technical_passed", True) and result["technical_passed"]
                report = self.evaluate(a["checkpoint"], f"pilot_{round_index}_{name}")
                a["development"] = report
                if round_index == 1:
                    a["confirmation"] = self.evaluate(a["checkpoint"], f"pilot_confirmation_{name}", start=10051)
            self.state["round"] = round_index + 1
            self.save()
        baseline = read(self.root / "evaluations/baseline_development.json")
        confirmation = read(self.root / "evaluations/baseline_regression.json")
        for name, a in arms.items():
            a["pilot_panels"] = {}
            for stage, before, after in [
                ("development", baseline, a["development"]),
                ("confirmation", confirmation, a["confirmation"]),
            ]:
                b, c = subset(before, PILOT_TARGETS), subset(after, PILOT_TARGETS)
                improved = c["mean_pass_rate"] >= b["mean_pass_rate"] + 0.099999 or (
                    b["mean_violation"] > 1e-8
                    and c["mean_violation"] <= 0.8 * b["mean_violation"]
                    and c["mean_pass_rate"] >= b["mean_pass_rate"] - 1e-9
                )
                a["pilot_panels"][stage] = dict(retained=retain(before, after), improved=improved)
            a["eligible"] = a["technical_passed"] and all(
                p["retained"] and p["improved"] for p in a["pilot_panels"].values()
            )
        eligible = [
            reward for reward in ("original", "aligned") if all(arms[f"{reward}_{s}"]["eligible"] for s in (44, 45))
        ]
        atomic_json(
            self.root / "pilot_decision.json",
            dict(eligible=eligible, arms={k: a["pilot_panels"] for k, a in arms.items()}),
        )
        if not eligible:
            self.stop("pilot_no_replicated_improvement")
            return

        def rating(reward):
            scores = [rank(subset(arms[f"{reward}_{s}"]["development"], PILOT_TARGETS)) for s in (44, 45)]
            return min(scores), reward == "original"

        selected = max(eligible, key=rating)
        self.state["arms"] = {k: a for k, a in arms.items() if a["reward"] == selected}
        for a in self.state["arms"].values():
            a.update(
                safe=a["checkpoint"],
                spent=0,
                last_full=0,
                regressions=0,
                streak=0,
                progress_step=0,
                progress_rank=list(rank(subset(a["development"], TARGETS))),
                progress_violation=subset(a["development"], TARGETS)["mean_violation"],
                promoted=False,
            )
        self.state.update(stage="motion", round=0)
        self.state["best"] = max(self.state["arms"].values(), key=lambda a: rank(a["development"]))["checkpoint"]
        self.save()

    def refinement(self):
        stage = self.state["stage"]
        seconds = self.state["sequence_seconds"] if stage == "sequence" else 60
        deadline = self.manifest["stage_deadlines"][stage]
        baseline = read(self.root / "evaluations/baseline_development.json")
        if stage == "sequence" and self.state["round"] == 0:
            for name, arm in self.state["arms"].items():
                entry = self.evaluate(
                    arm["checkpoint"], f"entry_sequence_{seconds}_{name}", cases=("sequence",), seconds=seconds
                )
                arm["progress_rank"] = list(rank(entry))
                arm["progress_violation"] = entry["mean_violation"]
            self.save()
        while time.time() < deadline:
            arms = self.state["arms"]
            active = {k: a for k, a in arms.items() if not a["promoted"]}
            if not active:
                if stage == "motion":
                    self.state.update(stage="sequence", round=0, sequence_seconds=20)
                elif seconds < 60:
                    self.state.update(round=0, sequence_seconds=seconds + 20)
                else:
                    self.state["stage"] = "qualification"
                    self.save()
                    return
                for a in arms.values():
                    a.update(
                        promoted=False,
                        streak=0,
                        regressions=0,
                        progress_step=a["spent"],
                        last_full=a["spent"],
                        progress_rank=[0.0, 0.0, -1e9],
                        progress_violation=1e9,
                    )
                self.save()
                return
            jobs = {
                name: self.make_job(name, a["checkpoint"], a["reward"], a["seed"], 17, False, stage, seconds)
                for name, a in active.items()
            }
            results = self.train(jobs, stage)
            for name, result in results.items():
                a = arms[name]
                a["checkpoint"] = result["checkpoint"]
                a["spent"] += result["additional_steps"]
                if not result["technical_passed"]:
                    self.stop(
                        "stage_time_budget"
                        if result["state"] == "budget_reached"
                        else "optimizer_technical_gate_failed"
                    )
                    return
                full = a["spent"] - a["last_full"] >= 500_000
                label = f'{stage}_{seconds}_{self.state["round"]:04d}_{name}'
                report = self.evaluate(a["checkpoint"], label, count=50 if full else 10, seconds=seconds)
                if not full:
                    quick_reference = self.evaluate(self.baseline, "baseline_quick", count=10)
                    if retain(quick_reference, report):
                        continue
                    report = self.evaluate(a["checkpoint"], label + "_full", count=50, seconds=seconds)
                    full = True
                a["last_full"] = a["spent"]
                reference = baseline
                if not retain(reference, report):
                    recheck = self.evaluate(a["checkpoint"], label + "_recheck", start=5001, seconds=seconds)
                    bcheck = self.evaluate(self.baseline, "baseline_recheck", start=5001)
                    if not retain(bcheck, recheck):
                        a["regressions"] += 1
                        a["checkpoint"] = a["safe"]
                        a["streak"] = 0
                        atomic_json(
                            self.root / "rollbacks" / f"{label}.json", dict(restored=a["safe"], spent_steps=a["spent"])
                        )
                        if a["regressions"] >= 2:
                            self.stop("repeated_confirmed_regression")
                            return
                        continue
                else:
                    a["safe"] = a["checkpoint"]
                    a["development"] = report
                    bests = self.state.setdefault("best_by_stage", {})
                    key = f"{stage}_{seconds}"
                    if key not in bests or rank(report) > tuple(bests[key]["rank"]):
                        bests[key] = dict(checkpoint=a["checkpoint"], rank=list(rank(report)))
                    self.state["best"] = bests[key]["checkpoint"]
                    targets = ("sequence",) if stage == "sequence" else TARGETS
                    target = subset(report, targets)
                    a["streak"] = a["streak"] + 1 if passes(target) else 0
                    a["promoted"] = a["streak"] >= 2
                    improved = target["mean_pass_rate"] > a["progress_rank"][1] + 1e-9 or (
                        a["progress_violation"] > 1e-8 and target["mean_violation"] <= 0.9 * a["progress_violation"]
                    )
                    if improved:
                        a["progress_step"] = a["spent"]
                        a["progress_rank"] = list(rank(target))
                        a["progress_violation"] = target["mean_violation"]
                    if a["spent"] - a["progress_step"] >= 1_000_000:
                        self.stop("one_million_steps_without_target_progress")
                        return
            self.state["round"] += 1
            self.save()
        self.stop(f"{stage}_budget_without_promotion")

    def qualification(self):
        # Candidate is chosen on development results before any locked panels open.
        self.state["holdouts_opened"] = True
        self.save()
        checkpoint = Path(self.state["best"])
        results = {}
        h = self.manifest["holdouts"]
        reports = []
        for label, start, cases, count, grid in [
            ("families", h["families"], (*FAMILIES, *BRAKES), 50, False),
            ("sequence_a", h["sequence_a"], ("sequence",), 50, False),
            ("sequence_b", h["sequence_b"], ("sequence",), 50, False),
            ("grid", h["grid"], FAMILIES, 20, True),
        ]:
            after = self.evaluate(checkpoint, "final_" + label, cases, count, start, stage="qualification", grid=grid)
            before = self.evaluate(
                self.baseline, "final_baseline_" + label, cases, count, start, stage="qualification", grid=grid
            )
            quality = quality_gate(before, after)
            no_accidents = not any(
                "fall" in e["failures"] or "self_collision" in e["failures"]
                for f in after["families"].values()
                for e in f["episodes"]
            )
            results[label] = dict(
                passed=passes(after, 0.95 if grid else 0.96) and quality["passed"] and no_accidents,
                quality=quality,
                no_accidents=no_accidents,
                counts={k: [v["success_count"], v["count"]] for k, v in after["families"].items()},
            )
            reports.append(str((self.root / "evaluations" / f"final_{label}.json").resolve()))
        qualified = all(r["passed"] for r in results.values())
        atomic_json(
            self.root / "qualification.json", dict(qualified=qualified, results=results, checkpoint=str(checkpoint))
        )
        self.state.update(
            qualified=qualified,
            stage="delivery",
            stop_reason="qualified" if qualified else "qualification_failed",
            reports=reports,
        )
        self.save()

    def delivery(self):
        checkpoint = Path(self.state["best"])
        evidence = self.state.get("reports", [])
        if not evidence:
            digest = sha256(checkpoint)
            for path in sorted((self.root / "evaluations").glob("*.json")):
                report = read(path)
                if report.get("checkpoint_sha256") == digest and "families" in report:
                    evidence.append(str(path.resolve()))
        target = bundle(checkpoint, self.root / "controller", self.state["qualified"], evidence)
        result = dict(
            qualified=self.state["qualified"],
            reason=self.state.get("stop_reason"),
            controller=str(target.parent),
            elapsed_hours=(time.time() - self.manifest["started"]) / 3600,
            training_continues=False,
            holdouts_opened=self.state["holdouts_opened"],
        )
        atomic_json(self.root / "result.json", result)
        (self.root / "SUMMARY.md").write_text(
            "# Moderate walking campaign\n\n"
            + f"- Qualified: {result['qualified']}\n- Stop reason: {result['reason']}\n"
            + f"- Elapsed hours: {result['elapsed_hours']:.2f}\n- Controller: `{target.parent}`\n\n"
            + "See pilot_decision.json or qualification.json and evaluations/ for motion-specific evidence.\n"
        )
        # Rendering uses a separate process because MuJoCo's GL backend is chosen at import.
        if not (self.root / "videos/status.json").exists() and time.time() < self.manifest["deadline"] - 120:
            env = dict(
                os.environ,
                MUJOCO_GL="osmesa",
                LP_NUM_THREADS="1",
                LD_LIBRARY_PATH="/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu",
            )
            with (self.root / "video.log").open("a") as log:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "mujoco_rl.command_walk_final_replay",
                            str(target.parent),
                            str(self.root / "videos"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, self.manifest["deadline"] - time.time() - 10),
                        check=True,
                    )
                except (subprocess.SubprocessError, OSError) as error:
                    atomic_json(self.root / "video_error.json", dict(error=repr(error)))
        self.state["stage"] = "completed"
        self.save()
        atomic_json(self.root / "status.json", dict(state="completed", heartbeat=time.time(), **result))
        print("[completed] " + str(result), flush=True)

    def run(self):
        try:
            while self.state["stage"] != "completed":
                if time.time() >= self.manifest["deadline"]:
                    self.stop("48_hour_wall_budget")
                stage = self.state["stage"]
                self.status(stage)
                if stage == "setup":
                    self.setup()
                elif stage == "pilot":
                    self.pilot()
                elif stage in ("motion", "sequence"):
                    self.refinement()
                elif stage == "qualification":
                    self.qualification()
                elif stage == "delivery":
                    self.delivery()
                else:
                    raise ValueError(f"Unknown stage {stage}")
        except TimeoutError:
            self.stop("stage_time_budget")
            self.delivery()
        except BaseException as error:
            atomic_json(
                self.root / "status.json",
                dict(state="failed", heartbeat=time.time(), error=repr(error), stage=self.state["stage"]),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "disable")
    campaign = Campaign(args)
    with (campaign.root / "campaign.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This campaign is already running") from error
        campaign.run()


if __name__ == "__main__":
    main()
