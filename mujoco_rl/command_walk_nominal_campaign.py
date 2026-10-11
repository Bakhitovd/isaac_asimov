"""Matched nominal-only trials; qualify retained improvements before adding difficulty."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .command_walk import FAMILIES
from .command_walk_eval import compact, rank
from .command_walk_robust_eval import parallel_evaluate
from .command_walk_robust_train import retention_passed
from .skill_campaign import atomic_json, sha256


def eligible(baseline, candidate):
    """A replicated improvement at trained difficulty, with per-family retention."""
    retained = retention_passed(baseline, candidate, tolerance=0.06)
    weakest = min(FAMILIES, key=lambda f: baseline["families"][f]["pass_rate"])
    before, after = baseline["families"][weakest], candidate["families"][weakest]
    improved = (
        after["pass_rate"] >= before["pass_rate"] + 0.04 - 1e-9
        and candidate["minimum_pass_rate"] >= baseline["minimum_pass_rate"]
    )
    return retained and improved


def qualified(report):
    return all(v["count"] >= 50 and v["pass_rate"] >= 0.95 for v in report["families"].values())


class NominalCampaign:
    def __init__(self, root):
        self.root = root.resolve()
        self.started = time.time()
        self.children = []
        self.cancelled = False
        self.env = dict(
            os.environ, MUJOCO_GL="disable", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )

    def status(self, state, **extra):
        atomic_json(
            self.root / "status.json",
            dict(state=state, heartbeat=time.time(), elapsed_hours=(time.time() - self.started) / 3600, **extra),
        )

    def stop(self, *_):
        self.cancelled = True
        for p in self.children:
            if p.poll() is None:
                p.terminate()

    def run_group(self, jobs, steps):
        handles, self.children = [], []
        try:
            for name, source, seed, weight in jobs:
                handle = (self.root / f"{name}.log").open("w")
                handles.append(handle)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mujoco_rl.command_walk_robust_train",
                    str(source),
                    "--output",
                    str(self.root / name),
                    "--variant",
                    "control",
                    "--level",
                    "0",
                    "--pilot",
                    "--steps",
                    str(steps),
                    "--seed",
                    str(seed),
                    "--reference",
                    str(self.root / "baseline/reference.npz"),
                    "--reference-weight",
                    str(weight),
                    "--quick-interval",
                    "50000",
                    "--full-interval",
                    "150000",
                    "--retention-tolerance",
                    "0.06",
                ]
                self.children.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=self.env))
            while any(p.poll() is None for p in self.children):
                if self.cancelled:
                    raise InterruptedError("Campaign interrupted")
                self.status("training", trials=[j[0] for j in jobs])
                time.sleep(5)
            if any(p.returncode for p in self.children):
                raise RuntimeError("A trainer failed; inspect its log")
            for name, *_ in jobs:
                status = json.loads((self.root / name / "status.json").read_text())
                if status["state"] != "completed":
                    raise RuntimeError(f"Incomplete trial: {name}")
        finally:
            for p in self.children:
                if p.poll() is None:
                    p.terminate()
                p.wait()
            for handle in handles:
                handle.close()

    def best(self, name, baseline):
        folder = self.root / name
        candidates = []
        for path in folder.glob("evaluations/*.json"):
            report = json.loads(path.read_text())
            if not report["full"]:
                continue
            nominal = report["nominal"]
            if retention_passed(baseline, nominal, tolerance=0.06):
                step, label = path.stem.split("_", 1)
                checkpoint = folder / "checkpoints" / f"{int(step):09d}_{label}" / "policy.zip"
                if checkpoint.exists():
                    candidates.append((rank(nominal), checkpoint, nominal))
        if not candidates:
            raise RuntimeError(f"No retained checkpoint in {name}")
        _, checkpoint, report = max(candidates, key=lambda x: x[0])
        return dict(name=name, checkpoint=str(checkpoint), nominal=compact(report), eligible=eligible(baseline, report))

    def run(self):
        baseline = json.loads((self.root / "baseline.json").read_text())
        prior = json.loads((self.root / "baseline_expected.json").read_text())
        if any(baseline["families"][f]["success_count"] != prior["families"][f]["success_count"] for f in FAMILIES):
            raise RuntimeError("Baseline did not reproduce; diagnose before training")
        source = self.root / "baseline/policy.zip"
        snapshot = self.root / "source_snapshot"
        snapshot.mkdir(exist_ok=False)
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, snapshot / path.name)
        atomic_json(
            self.root / "provenance.json",
            dict(
                revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                source_hashes={p.name: sha256(p) for p in snapshot.iterdir()},
                baseline_sha256=sha256(source),
                reference_sha256=sha256(self.root / "baseline/reference.npz"),
            ),
        )
        trials = []
        for seed in (44, 45):
            jobs = [
                (f"pilot_{variant}_{seed}", source, seed, weight)
                for variant, weight in (("control", 0.0), ("protected", 0.1))
            ]
            self.run_group(jobs, 500_000)
            trials.extend(self.best(name, baseline) for name, *_ in jobs)
            atomic_json(self.root / "comparison.json", dict(baseline=compact(baseline), trials=trials))
        variants = [
            v for v in ("control", "protected") if sum(t["eligible"] for t in trials if f"_{v}_" in t["name"]) == 2
        ]
        if not variants:
            self.status(
                "review_required",
                reason="No approach improved nominal behavior with retention in both seeds; inspect failure causes before reward changes",
            )
            return
        winner = max(
            variants, key=lambda v: sum(t["nominal"]["minimum_pass_rate"] for t in trials if f"_{v}_" in t["name"])
        )
        selected = [t for t in trials if f"_{winner}_" in t["name"]]
        if not all(qualified(t["nominal"]) for t in selected):
            jobs = [
                (f"extended_{winner}_{seed}", Path(t["checkpoint"]), seed, 0.1 if winner == "protected" else 0.0)
                for seed, t in zip((44, 45), selected)
            ]
            self.run_group(jobs, 1_000_000)
            selected = [self.best(name, baseline) for name, *_ in jobs]
        atomic_json(self.root / "selected.json", selected)
        for trial in selected:
            if not qualified(trial["nominal"]):
                continue
            self.status("qualifying", candidate=trial["checkpoint"])
            holdout = parallel_evaluate(
                Path(trial["checkpoint"]),
                level=0,
                seed_start=10051,
                count=50,
                progress=lambda *_: self.status("qualifying", candidate=trial["checkpoint"]),
            )
            atomic_json(self.root / f'{trial["name"]}_holdout.json', holdout)
            if qualified(holdout):
                atomic_json(
                    self.root / "qualified.json",
                    dict(
                        **trial,
                        holdout=compact(holdout),
                        policy_sha256=sha256(Path(trial["checkpoint"])),
                        scope="nominal simulation only",
                    ),
                )
                self.status(
                    "nominal_qualified",
                    reason="Nominal holdout passed; longer sequences and demos next, robustness remains unqualified",
                )
                return
        self.status(
            "review_required", reason="Nominal qualification incomplete; inspect selected checkpoint failure reports"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    campaign = NominalCampaign(args.root)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, campaign.stop)
    try:
        campaign.run()
    except Exception as error:
        campaign.stop()
        campaign.status("failed", reason=repr(error))
        raise
