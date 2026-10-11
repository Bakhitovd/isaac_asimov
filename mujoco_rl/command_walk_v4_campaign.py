"""Persisted 24-hour nominal-motion campaign with explicit behavioral gates."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .command_walk import FAMILIES
from .command_walk_eval import compact, rank
from .command_walk_v4 import TARGETS, Spec, passes, select_initialization, specialist_passes
from .command_walk_v4_release import qualify
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = dict(diagnosis=4, specialists=8, distillation=4, transitions=6, qualification=2)


class GateFailure(RuntimeError):
    pass


class Campaign:
    def __init__(self, args):
        self.args, self.root = args, args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "campaign.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.children, self.cancelled = [], False
        path = self.root / "campaign.json"
        if path.exists():
            if not args.resume:
                raise ValueError("Campaign exists; use --resume to preserve deadlines")
            self.state = json.loads(path.read_text())
            if self.state["source_sha256"] != sha256(args.baseline) or self.state["pivot_sha256"] != sha256(args.pivot):
                raise ValueError("Campaign sources changed")
            if self.state["normalization_sha256"] != {
                "baseline": sha256(_stats_path(args.baseline)),
                "pivot": sha256(_stats_path(args.pivot)),
            }:
                raise ValueError("Campaign normalization changed")
            hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
            if hashes != self.state["source_hashes"]:
                raise ValueError("Code changed since campaign launch; do not mix implementations in one comparison")
        else:
            self.state = dict(
                started=time.time(),
                deadline=time.time() + 24 * 3600,
                stages={},
                jobs={},
                source_sha256=sha256(args.baseline),
                pivot_sha256=sha256(args.pivot),
                normalization_sha256={
                    "baseline": sha256(_stats_path(args.baseline)),
                    "pivot": sha256(_stats_path(args.pivot)),
                },
                revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                publish=args.publish,
                source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            )
            snapshot = self.root / "source_snapshot"
            snapshot.mkdir()
            for p in Path(__file__).parent.glob("*.py"):
                shutil.copy2(p, snapshot / p.name)
            for name, source in (("baseline", args.baseline), ("pivot", args.pivot)):
                folder = self.root / name
                folder.mkdir()
                shutil.copy2(source, folder / "policy.zip")
                shutil.copy2(_stats_path(source), folder / "policy_vecnormalize.pkl")
            self.persist()
        self.args.publish = self.state["publish"]
        self.env = dict(
            os.environ, MUJOCO_GL="osmesa", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        mesa = Path.home() / ".local/mujoco-osmesa/usr/lib/x86_64-linux-gnu"
        self.env["LD_LIBRARY_PATH"] = str(mesa) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        self.last_status = 0.0
        self.last_journal = None

    def persist(self):
        atomic_json(self.root / "campaign.json", self.state)

    def write(self, state="running", **extra):
        atomic_json(
            self.root / "status.json",
            dict(
                state=state,
                stage=self.state.get("stage"),
                heartbeat=time.time(),
                elapsed_hours=(time.time() - self.state["started"]) / 3600,
                deadline=self.state["deadline"],
                stage_deadline=self.state.get("stage_deadline"),
                **extra,
            ),
        )
        self.last_status = time.time()
        event = (state, self.state.get("stage"), extra.get("reason"))
        if event != self.last_journal:
            with (self.root / "campaign_journal.md").open("a") as handle:
                handle.write(
                    f'\n## {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}\n\n'
                    f'State: {state}. Stage: {self.state.get("stage")}.\n\n'
                    f'{extra.get("reason","Stage started; results remain unqualified.")}\n'
                )
            self.last_journal = event

    def stop(self, *_):
        self.cancelled = True
        for p in self.children:
            if p.poll() is None:
                p.terminate()

    def progress(self):
        if self.cancelled:
            raise InterruptedError("Campaign interrupted")
        if time.time() >= min(self.state["deadline"], self.state["stage_deadline"]):
            raise TimeoutError("Persisted stage/global deadline reached")
        if time.time() - self.last_status > 10:
            self.write()

    def enter(self, name):
        self.state["stage"] = name
        stages = self.state["stages"]
        if name not in stages:
            stages[name] = dict(
                started=time.time(), deadline=min(self.state["deadline"], time.time() + BUDGETS[name] * 3600)
            )
        self.state["stage_deadline"] = stages[name]["deadline"]
        self.persist()
        self.write()
        self.progress()
        print(f'[stage] {name}, deadline={self.state["stage_deadline"]}', flush=True)

    def jobs(self, jobs):
        """Run at most two jobs. Resume interrupted PPO jobs into new immutable attempt folders."""
        for offset in range(0, len(jobs), 2):
            self.children = []
            running = []
            handles = []
            try:
                for name, source, spec, seed, steps, diagnostic in jobs[offset : offset + 2]:
                    self.progress()
                    job = self.state["jobs"].setdefault(name, dict(attempts=[]))
                    if job.get("finished"):
                        continue
                    resume = False
                    if job["attempts"]:
                        old = Path(job["attempts"][-1])
                        status = json.loads((old / "status.json").read_text()) if (old / "status.json").exists() else {}
                        if status.get("state") == "completed":
                            continue
                        if status.get("stop_reason") != "interrupted" or not status.get("checkpoint"):
                            raise GateFailure(f"Job {name} stopped: {status}")
                        source = Path(status["checkpoint"])
                        resume = True
                    output = self.root / name / f'attempt_{len(job["attempts"])}'
                    output.parent.mkdir(exist_ok=True)
                    job["attempts"].append(str(output))
                    self.persist()
                    handle = (output.parent / f"{output.name}.log").open("w")
                    handles.append(handle)
                    cmd = [
                        sys.executable,
                        "-u",
                        "-m",
                        "mujoco_rl.command_walk_v4_train",
                        str(source),
                        "--output",
                        str(output),
                        "--deadline",
                        str(self.state["stage_deadline"]),
                        "--initialization",
                        spec.initialization,
                        "--task",
                        spec.task,
                        "--seed",
                        str(seed),
                        "--steps",
                        str(steps),
                    ]
                    if diagnostic:
                        cmd.append("--diagnostic")
                    if spec.heading_penalty:
                        cmd.append("--heading-penalty")
                    if resume:
                        cmd.append("--resume")
                    self.children.append(subprocess.Popen(cmd, env=self.env, stdout=handle, stderr=subprocess.STDOUT))
                    running.append((name, output))
                while any(p.poll() is None for p in self.children):
                    self.progress()
                    time.sleep(5)
                if any(p.returncode for p in self.children):
                    raise GateFailure("Trainer failed; inspect attempt logs")
                for name, output in running:
                    status = json.loads((output / "status.json").read_text())
                    if status["stop_reason"] not in ("step_budget", "repeated_regression"):
                        raise GateFailure(f'{name}: {status["stop_reason"]}')
                    self.state["jobs"][name]["finished"] = True
                    self.persist()
            finally:
                for p in self.children:
                    if p.poll() is None:
                        p.terminate()
                for p in self.children:
                    try:
                        p.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                for handle in handles:
                    handle.close()

    def reports(self, name):
        results = []
        for attempt in self.state["jobs"][name]["attempts"]:
            for path in (Path(attempt) / "evaluations").glob("*.json"):
                row = json.loads(path.read_text())
                if row.get("full"):
                    results.append(row)
        return sorted(results, key=lambda r: r["report"]["training_steps"])

    def best(self, name, task):
        rows = [r for r in self.reports(name) if r["retained"]]
        if not rows:
            raise GateFailure(f"No retained checkpoint: {name}")

        def ordering(r):
            rates = [r["report"]["families"][k]["pass_rate"] for k in TARGETS[task]]
            return min(rates), sum(rates) / len(rates)

        return max(rows, key=ordering)

    def plateaued(self, name, task):
        rows = self.reports(name)
        newest = rows[-1]
        older = [r for r in rows if r["actor_steps"] <= newest["actor_steps"] - 100_000]
        if not older:
            return False

        def metrics(row):
            families = row["report"]["families"]
            return (
                min(families[k]["pass_rate"] for k in TARGETS[task]),
                sum(families[k]["mean_violation"] for k in TARGETS[task]) / len(TARGETS[task]),
            )

        before, after = metrics(older[-1]), metrics(newest)
        return not (after[0] >= before[0] + 0.04 - 1e-9 or before[1] > 1e-6 and after[1] < 0.9 * before[1])

    def run(self):
        baseline = self.root / "baseline/policy.zip"
        pivot = self.root / "pivot/policy.zip"
        if "initialization" not in self.state:
            self.enter("diagnosis")
            self.jobs(
                [
                    (f"diagnostic_{init}_{seed}", baseline, Spec(init), seed, 100_000, True)
                    for init in ("current", "preserved", "calibrated")
                    for seed in (44, 45)
                ]
            )
            comparison = {}
            for init in ("current", "preserved", "calibrated"):
                comparison[init] = []
                for seed in (44, 45):
                    rows = self.reports(f"diagnostic_{init}_{seed}")
                    comparison[init].append(dict(initial=compact(rows[0]["report"]), final=compact(rows[-1]["report"])))
            atomic_json(self.root / "initialization_comparison.json", comparison)
            choice = select_initialization(comparison)
            if choice is None:
                raise GateFailure(
                    "All initialization packages failed retention in at least one seed; do not start specialists"
                )
            self.state["initialization"] = choice
            self.persist()
        init = self.state["initialization"]
        if "teachers" not in self.state:
            self.enter("specialists")
            selected = {"baseline": str(baseline)}
            for task in ("arcs", "pivots", "braking"):
                source = pivot if task == "pivots" else baseline
                names = [f"specialist_{task}_{seed}" for seed in (44, 45)]
                self.jobs(
                    [(name, source, Spec(init, task), seed, 500_000, False) for name, seed in zip(names, (44, 45))]
                )
                rows = [self.best(name, task) for name in names]
                if not all(specialist_passes(r["report"], task) for r in rows):
                    if not all(
                        specialist_passes(row["report"], task) or self.plateaued(name, task)
                        for name, row in zip(names, rows)
                    ):
                        raise GateFailure(
                            f"{task} budget ended without qualification; no confirmed plateau for automatic reward change"
                        )
                    # Only a heading-dominated failure permits the predeclared reward comparison.
                    counts = {}
                    for row in rows:
                        for case in TARGETS[task]:
                            for failure, count in row["report"]["families"][case]["failure_counts"].items():
                                key = failure.split("/")[-1]
                                key = "heading" if key.startswith("heading_") else key
                                counts[key] = counts.get(key, 0) + count
                    if not counts or max(counts, key=counts.get) != "heading":
                        raise GateFailure(f"{task} specialist gate failed; dominant failures={counts}")
                    jobs = []
                    for seed, row in zip((44, 45), rows):
                        for variant, penalty in (("control", False), ("heading", True)):
                            jobs.append(
                                (
                                    f"reward_{task}_{variant}_{seed}",
                                    Path(row["checkpoint"]),
                                    Spec(init, task, penalty),
                                    seed,
                                    200_000,
                                    False,
                                )
                            )
                    self.jobs(jobs)
                    options = []
                    for variant in ("control", "heading"):
                        pair = [self.best(f"reward_{task}_{variant}_{seed}", task) for seed in (44, 45)]
                        if all(specialist_passes(r["report"], task) for r in pair):
                            options.append(pair)
                    if not options:
                        raise GateFailure(f"{task}: bounded reward comparison did not qualify both seeds")
                    rows = max(options, key=lambda pair: min(r["report"]["minimum_pass_rate"] for r in pair))
                chosen = max(rows, key=lambda r: rank(r["report"]))
                selected[task] = chosen["checkpoint"]
                atomic_json(self.root / f"{task}_selection.json", rows)
            self.state["teachers"] = selected
            atomic_json(self.root / "teachers.json", selected)
            self.persist()
        if "student" not in self.state:
            self.enter("distillation")
            attempts = self.state.setdefault("distill_attempts", [])
            resume_from = None
            completed = None
            if attempts:
                old = Path(attempts[-1])
                status = json.loads((old / "status.json").read_text())
                if status["state"] == "completed":
                    completed = old
                elif status.get("reason") == "interrupted":
                    resume_from = old
                else:
                    raise GateFailure(f"Distillation stopped: {status}")
            output = completed or self.root / "distillation" / f"attempt_{len(attempts)}"
            if completed is None:
                output.parent.mkdir(exist_ok=True)
                attempts.append(str(output))
                self.persist()
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mujoco_rl.command_walk_v4_distill",
                    str(self.root / "teachers.json"),
                    "--output",
                    str(output),
                    "--deadline",
                    str(self.state["stage_deadline"]),
                    "--initialization",
                    init,
                ]
                if resume_from:
                    command.extend(["--resume-from", str(resume_from)])
                with (output.parent / f"{output.name}.log").open("w") as handle:
                    child = subprocess.Popen(command, env=self.env, stdout=handle, stderr=subprocess.STDOUT)
                    self.children = [child]
                    try:
                        while child.poll() is None:
                            self.progress()
                            time.sleep(5)
                    finally:
                        if child.poll() is None:
                            child.terminate()
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    if child.returncode:
                        raise GateFailure("Distillation failed")
            if not (output / "report.json").exists():
                raise GateFailure("Distillation stopped before a complete comparison")
            result = json.loads((output / "report.json").read_text())
            if not result["passed"]:
                raise GateFailure("Consolidation did not reach 90% in every family")
            self.state["student"] = result["best"]["checkpoint"]
            self.persist()
        if "candidate" not in self.state:
            self.enter("transitions")
            self.jobs(
                [
                    (
                        f"transitions_{seed}",
                        Path(self.state["student"]),
                        Spec(init, "transitions"),
                        seed,
                        1_000_000,
                        False,
                    )
                    for seed in (44, 45)
                ]
            )
            rows = [self.best(f"transitions_{seed}", "transitions") for seed in (44, 45)]
            candidates = [r for r in rows if passes(r["report"], FAMILIES)]
            if not candidates:
                raise GateFailure("No transition policy met the 48/50 per-family development gate")
            self.state["candidate"] = max(candidates, key=lambda r: rank(r["report"]))["checkpoint"]
            self.persist()
        self.enter("qualification")
        # Never retry the same holdout after it has informed a decision.
        output = self.root / "qualification"
        if output.exists():
            raise GateFailure("Qualification already attempted; preserve evidence and reserve fresh holdout seeds")
        if not qualify(Path(self.state["candidate"]), output, self.progress):
            raise GateFailure("Nominal or 60-second sequence holdout gate failed")
        if self.args.publish:
            if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "main":
                raise GateFailure("Publication requires main")
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
                raise GateFailure("Qualified artifacts saved; working tree must be clean for publication")
            destination = ROOT / "mujoco_rl/checkpoints/command_walk_v4"
            if destination.exists():
                raise GateFailure("Refusing to overwrite an existing published policy")
            shutil.copytree(output, destination)
            subprocess.run(["git", "add", "-f", str(destination)], cwd=ROOT, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex",
                    "-c",
                    "user.email=codex@openai.com",
                    "commit",
                    "-m",
                    "Publish qualified nominal command walking policy",
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(["git", "push", "fork", "main"], cwd=ROOT, check=True)
        self.write("qualified", reason="Nominal and long-sequence holdouts passed; robustness remains unqualified")


def show(root):
    status = root / "status.json"
    print(status.read_text() if status.exists() else "Waiting for campaign startup")
    for p in sorted(root.glob("*/attempt_*/status.json")):
        r = json.loads(p.read_text())
        print(
            p.parent.parent.name,
            p.parent.name,
            r.get("state"),
            r.get("phase", r.get("stop_reason")),
            "actor steps",
            r.get("actor_steps"),
            "calibration steps",
            r.get("calibration_steps"),
        )
        evaluation = r.get("evaluation", {}).get("nominal")
        if evaluation is None and (p.parent / "report.json").exists():
            evaluation = json.loads((p.parent / "report.json").read_text()).get("final")
        if evaluation:
            families = evaluation["families"]
            print("  " + " | ".join(f'{k}: {v["pass_rate"]:.0%}' for k, v in families.items()))
            print(f'  {next(iter(families.values()))["count"]} cases per motion; development, not qualification')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("logs/mujoco_rl/command_walk_v1/seed45/checkpoints/005953536_probe/policy.zip"),
    )
    p.add_argument(
        "--pivot",
        type=Path,
        default=Path("logs/mujoco_rl/command_walk_v3/pilot_protected_44/checkpoints/006340608_full/policy.zip"),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    if args.watch:
        while True:
            print("\033[2J\033[H", end="")
            show(args.output)
            time.sleep(5)
    else:
        campaign = Campaign(args)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, campaign.stop)
        try:
            campaign.run()
        except (GateFailure, TimeoutError, InterruptedError) as error:
            campaign.stop()
            campaign.write("review_required", reason=str(error))
            print(f"[stop] {error}", flush=True)
        except Exception as error:
            campaign.stop()
            campaign.write("failed", reason=repr(error))
            raise
