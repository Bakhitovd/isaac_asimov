"""Run a bounded, auditable multi-skill campaign without hiding failed stages."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {"audit": 6, "recover": 20, "nav_squat": 4, "run": 12, "jump": 12, "integration": 10, "reserve": 8}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_campaign(directory: Path, checkpoints: dict[str, Path]) -> None:
    if directory.exists():
        raise FileExistsError("Choose a new campaign directory; existing evidence is never overwritten")
    for skill, checkpoint in checkpoints.items():
        if not checkpoint.is_file() or not checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl").is_file():
            raise FileNotFoundError(f"Missing {skill} checkpoint or normalization: {checkpoint}")
    directory.mkdir(parents=True)
    registry = {}
    for skill, checkpoint in checkpoints.items():
        checkpoint = checkpoint.resolve()
        folder = directory.resolve() / "baselines" / skill
        folder.mkdir(parents=True)
        paths = [checkpoint, checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")]
        for optional in (checkpoint.parent / "config.json", checkpoint.with_name(checkpoint.stem + "_training.json"),
                         checkpoint.with_name(checkpoint.stem + "_report.json")):
            if optional.exists():
                paths.append(optional)
        hashes = {}
        for source in paths:
            shutil.copy2(source, folder / source.name)
            hashes[source.name] = sha256(folder / source.name)
        registry[skill] = {"source": str(checkpoint), "checkpoint": str(folder / checkpoint.name),
                           "sha256": hashes, "status": "validated_tilt_balance_only" if skill == "recover" else
                           "published" if skill == "walk" else "previously_accepted"}
    atomic_json(directory / "registry.json", registry)
    (directory / "prior_work.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    (directory / "prior_git_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))


class SeedRegistry:
    """Explicit case ranges; inspected holdouts can never become fresh again."""
    def __init__(self, path: Path, origin: int = 2_000_000_000):
        self.path = path
        self.origin = origin
        self.panels = json.loads(path.read_text()) if path.exists() else {}

    def reserve(self, name: str, role: str, count: int) -> int:
        if role not in {"training", "development", "holdout"} or not 1 <= count <= 100000:
            raise ValueError("Invalid seed role or count")
        if name in self.panels:
            panel = self.panels[name]
            if panel["original_role"] != role or panel["count"] != count:
                raise ValueError("Cannot change an existing seed reservation")
            return panel["start"]
        start = self.origin + 100000 * len(self.panels)
        if start + count >= 2 ** 32:
            raise ValueError("Seed registry exhausted")
        self.panels[name] = {"start": start, "count": count, "role": role, "original_role": role,
                             "inspected": False, "results": []}
        atomic_json(self.path, self.panels)
        return start

    def record(self, name: str, report: Path, passed: bool) -> None:
        panel = self.panels[name]
        panel["inspected"] = True
        panel["results"].append({"report": str(report), "passed": passed})
        if panel["role"] == "holdout" and not passed:
            panel["role"] = "development_retry"
        atomic_json(self.path, self.panels)


class Campaign:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "campaign.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.registry = json.loads((self.directory / "registry.json").read_text())
        self.seeds = SeedRegistry(self.directory / "seeds.json")
        self.status_path = self.directory / "status.json"
        if self.status_path.exists():
            raise FileExistsError("Campaign already started; inspect status instead of silently restarting")
        self.started = time.time()
        continuation_path = self.directory / "continuation.json"
        continuation = json.loads(continuation_path.read_text()) if continuation_path.exists() else {}
        self.started = float(continuation.get("budget_started_at", self.started))
        self.prior_cpu_hours = float(continuation.get("completed_child_cpu_hours", 0))
        self.status = {"state": "running", "started_at": self.started, "pid": os.getpid(),
                       "budget_hours": BUDGETS, "stage": "audit", "jobs": [], "stages": {},
                       "accepted": {}, "next_decision": "Pass simulation audit before training"}
        self.status["continuation"] = continuation
        snapshot = self.directory / "source_snapshot"
        snapshot.mkdir(exist_ok=False)
        source_hashes = {}
        for source in (ROOT / "mujoco_rl").glob("*.py"):
            shutil.copy2(source, snapshot / source.name)
            source_hashes[source.name] = sha256(source)
        atomic_json(self.directory / "campaign_config.json", {
            "budget_hours": BUDGETS, "maximum_machine_hours": 72, "maximum_parallel_trainers": 2,
            "workers_per_trainer": 8, "learner_threads": 1, "source_sha256": source_hashes,
            "environment_version": 12, "python": sys.version, "argv": sys.argv,
            "implementation_smoke_cases": [2100100000, 2100200000]})
        self.stage_started = self.started
        self.process = None
        self.stopping = False
        self.environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="osmesa")
        mesa = Path.home() / ".local/mujoco-osmesa/usr/lib/x86_64-linux-gnu"
        if mesa.exists():
            self.environment["LD_LIBRARY_PATH"] = str(mesa) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.publish()

    def stop(self, *_):
        self.stopping = True

    def publish(self):
        self.status["heartbeat_at"] = time.time()
        self.status["elapsed_hours"] = (time.time() - self.started) / 3600
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.status["completed_child_cpu_hours"] = self.prior_cpu_hours + (usage.ru_utime + usage.ru_stime) / 3600
        atomic_json(self.status_path, self.status)

    def remaining(self) -> float:
        stage = self.status["stage"]
        return max(0.0, min(72 * 3600 - (time.time() - self.started),
                            BUDGETS[stage] * 3600 - (time.time() - self.stage_started)))

    def begin(self, stage: str, hypothesis: str):
        self.status["stage"] = stage
        self.stage_started = time.time()
        self.status["stages"][stage] = {"state": "running", "started_at": self.stage_started,
                                          "hypothesis": hypothesis}
        self.status["next_decision"] = hypothesis
        self.publish()
        print(f"[campaign] {stage}: {hypothesis}", flush=True)

    def finish(self, state: str, reason: str):
        self.status["stages"][self.status["stage"]].update(
            state=state, reason=reason, elapsed_hours=(time.time() - self.stage_started) / 3600)
        self.publish()

    def run(self, name: str, module: str, arguments: list, hours: float, hypothesis: str) -> bool:
        if self.stopping or self.remaining() <= 0:
            return False
        command = [sys.executable, "-u", "-m", f"mujoco_rl.{module}", *map(str, arguments)]
        folder = self.directory / "jobs" / name
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / "experiment.json", {"hypothesis": hypothesis, "command": command,
                    "budget_hours": min(hours, self.remaining() / 3600), "stage": self.status["stage"],
                    "success_rule": "Read the saved evaluation; process exit alone never accepts a policy"})
        deadline = time.time() + min(hours * 3600, self.remaining())
        job = {"name": name, "log": str(folder / "output.log"), "state": "running", "started_at": time.time()}
        self.status["jobs"].append(job)
        with (folder / "output.log").open("w") as handle:
            self.process = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=handle,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = self.process.pid
            self.publish()
            while self.process.poll() is None:
                if self.stopping or time.time() >= deadline:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait()
                    job["state"] = "stopped" if self.stopping else "budget_exhausted"
                    break
                self.publish()
                print(f"[campaign] {self.status['stage']}/{name} running; "
                      f"elapsed={self.status['elapsed_hours']:.2f}h; log={job['log']}", flush=True)
                time.sleep(20)
            job["exit_code"] = self.process.returncode
            if job["state"] == "running":
                job["state"] = "completed" if self.process.returncode == 0 else "failed"
            job["finished_at"] = time.time()
            self.process = None
        self.publish()
        return job["state"] == "completed"

    def evaluate(self, checkpoint: Path, skill: str, label: str, role: str = "development",
                 count: int = 50, level: int | None = None, scenario: str = "standard",
                 reuse_panel: str | None = None) -> dict | None:
        panel = reuse_panel or label
        seed = self.seeds.reserve(panel, role, count)
        output = self.directory / "evaluations" / f"{label}.json"
        arguments = [checkpoint, "--skill", skill, "--seed-start", seed, "--seeds", count,
                     "--scenario", scenario, "--output", output]
        if level is not None:
            arguments += ["--recovery-level", level]
        if not self.run(label, "skill_eval", arguments, 1,
                        f"Evaluate {skill}/{scenario} on recorded {role} cases"):
            return None
        result = json.loads(output.read_text())
        self.seeds.record(panel, output, result["criterion_met"])
        return result

    def qualify(self, checkpoint: Path, skill: str, label: str) -> bool:
        scenarios = ("standard", "settled_floor", "dynamic_fall") if skill == "recover" else ("standard",)
        reports = []
        # Failed confirmations stay mandatory after repair; never replace a hard
        # panel with a lucky newly drawn holdout.
        for name, panel in list(self.seeds.panels.items()):
            if panel["role"] != "development_retry" or not panel["results"]:
                continue
            previous_path = Path(panel["results"][-1]["report"])
            if not previous_path.exists():
                continue
            previous = json.loads(previous_path.read_text())
            if previous.get("skill") != skill:
                continue
            result = self.evaluate(checkpoint, skill, label + "_retry_" + name, "holdout", panel["count"],
                                   level=previous.get("recovery_level"),
                                   scenario=previous.get("scenario", "standard"), reuse_panel=name)
            if not result or not result["criterion_met"]:
                return False
            reports.append(result)
        for scenario in scenarios:
            for role in ("development", "holdout"):
                result = self.evaluate(checkpoint, skill, f"{label}_{scenario}_{role}", role,
                                       100 if skill == "recover" else 50, scenario=scenario)
                if not result or not result["criterion_met"]:
                    return False
                reports.append(result)
        if skill in {"recover", "run"}:
            retained = self.evaluate(checkpoint, "recover" if skill == "recover" else "nav",
                                     label + "_retention", count=50, level=9 if skill == "recover" else None)
            if not retained or not retained["criterion_met"]:
                return False
            reports.append(retained)
        destination = self.directory / "accepted" / skill
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination / "accepted.zip")
        shutil.copy2(checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl"), destination / "accepted_vecnormalize.pkl")
        atomic_json(destination / "qualification.json", {"skill": skill, "status": "accepted_simulation_only",
                    "environment_version": 12, "source_checkpoint": str(checkpoint), "reports": reports,
                    "sha256": {p.name: sha256(p) for p in destination.glob("*.zip")},
                    "normalization_sha256": sha256(destination / "accepted_vecnormalize.pkl")})
        self.status["accepted"][skill] = str(destination / "accepted.zip")
        self.publish()
        self.videos(checkpoint, skill, label, reports[0])
        return True

    def videos(self, checkpoint: Path, skill: str, label: str, result: dict,
               level: int | None = None):
        for passed in (True, False):
            case = next((case for case in result["episodes"] if case["passed"] == passed), None)
            if case is None:
                continue
            name = f"{label}_{'success' if passed else 'failure'}_seed{case['seed']}"
            video = self.directory / "videos" / f"{name}.mp4"
            arguments = [checkpoint, "--skill", skill, "--seeds", 1, "--seed-start", case["seed"],
                         "--video", video, "--video-seed", case["seed"],
                         "--scenario", result.get("scenario", "standard")]
            if level is not None:
                arguments += ["--recovery-level", level]
            self.run(name, "skill_eval", arguments, 0.25,
                     "Replay the first recorded success/failure; selection does not depend on appearance")

    def baseline(self, skill: str) -> Path:
        return Path(self.registry[skill]["checkpoint"])

    def verify_preserved(self):
        for entry in self.registry.values():
            folder = Path(entry["checkpoint"]).parent
            for name, digest in entry["sha256"].items():
                if sha256(folder / name) != digest:
                    raise RuntimeError(f"Preserved baseline changed: {folder / name}")

    def teachers(self, skill: str) -> Path | None:
        poses = ("front", "back", "left", "right") if skill == "recover" else ("up", "forward")
        datasets = []
        for pose in poses:
            accepted = False
            for attempt in range(3):
                label = f"{skill}_{pose}_teacher_{attempt}"
                seed = self.seeds.reserve(label, "training", 10000)
                path = self.directory / "teachers" / f"{label}.json"
                completed = self.run(label, "motion_teacher", ["--skill", skill, "--pose", pose, "--output", path,
                    "--seed", 44 + attempt, "--seed-start", seed, "--hours", 0.9,
                    "--workers", 8, "--population", 32, "--iterations", 8], 1,
                    f"A six-phase whole-body {pose} trajectory can complete {skill} under randomized physics")
                if not completed or not path.exists():
                    continue
                report = self.directory / "teachers" / f"{label}_verification.json"
                data = path.with_suffix(".npz")
                dev_seed = self.seeds.reserve(label + "_verification", "development", 50)
                if not self.run(label + "_verify", "motion_teacher", ["--skill", skill, "--pose", pose,
                    "--verify", path, "--seed-start", dev_seed, "--episodes", 50, "--output", report], 0.5,
                    "Search score must reproduce on separate physics cases"):
                    continue
                result = json.loads(report.read_text())
                self.seeds.record(label + "_verification", report, result["criterion_met"])
                if not result["criterion_met"]:
                    # A failed template is investigated twice before further variations stop.
                    if attempt >= 1:
                        break
                    continue
                train_seed = self.seeds.reserve(label + "_collection", "training", 512)
                collected = self.run(label + "_collect", "motion_teacher", ["--skill", skill, "--pose", pose,
                    "--verify", path, "--seed-start", train_seed, "--episodes", 512,
                    "--output", report.with_name(label + "_collection.json"), "--dataset", data,
                    "--action-noise", 0.02], 1,
                    "Collect successful demonstrations without using evaluation cases for training")
                if collected and data.exists():
                    datasets.append(data)
                    accepted = True
                    break
            if not accepted:
                self.status["stages"][skill].setdefault("teacher_blockers", []).append(pose)
                self.publish()
        if len(datasets) != len(poses):
            return None
        import numpy as np
        combined, starts, offset = {}, [], 0
        for dataset in datasets:
            with np.load(dataset, allow_pickle=False) as values:
                starts.extend((values["episode_starts"] + offset).tolist())
                for key in values.files:
                    if key != "episode_starts":
                        combined.setdefault(key, []).append(values[key].copy())
                offset += len(values["actions"])
        output = self.directory / "teachers" / f"{skill}_demonstrations.npz"
        np.savez_compressed(output, episode_starts=np.array(starts),
                            **{key: np.concatenate(values) for key, values in combined.items()})
        return output

    def trials(self, skill: str, source: Path, dataset: Path | None = None) -> Path | None:
        replay_args = []
        if dataset and skill == "recover":
            replay = self.directory / "teachers" / "balance_replay.npz"
            seed = self.seeds.reserve("balance_replay", "training", 520)
            if not self.run("balance_replay", "recovery_retention", [self.baseline("recover"), "--output", replay,
                            "--episodes", 520, "--seed-start", seed, "--workers", 8], 1,
                            "Rehearse the preserved balance policy while fitting floor-recovery demonstrations"):
                return None
            replay_args = ["--replay", replay]
        if dataset:
            fit_dir = self.directory / "trials" / f"{skill}_fit"
            dev_seed = self.seeds.reserve(skill + "_trial_development", "development", 200)
            if not self.run(skill + "_fit", "skill_refine", [source, "--skill", skill, "--output", fit_dir,
                    "--dataset", dataset, "--fit-updates", 8000, "--steps", 0, "--seed", 44,
                    "--dev-seed", dev_seed, *replay_args], 1,
                            "Fitting the verified teacher must produce closed-loop capability"):
                return None
            source = fit_dir / "initialized.zip"
            fitted = json.loads((fit_dir / "initial_evaluation.json").read_text())
            fitted_report = json.loads((fit_dir / "report.json").read_text())
            self.seeds.record(skill + "_trial_development", fit_dir / "initial_evaluation.json", fitted["criterion_met"])
            if fitted["pass_rate"] < 0.5 or (fitted_report["initial_retention"] is not None
                                               and not fitted_report["initial_retention"]["criterion_met"]):
                self.status["stages"][skill]["blocker"] = "Fitted policy failed 50% task floor or balance retention; inspect before PPO"
                return None
        else:
            dev_seed = self.seeds.reserve(skill + "_trial_development", "development", 200)
        results = []
        for seed in (44, 45):
            directory = self.directory / "trials" / f"{skill}_seed{seed}"
            args = [source, "--skill", skill, "--output", directory, "--seed", seed,
                    "--steps", 196608, "--hours", 2, "--dev-seed", dev_seed, "--warmup-steps", 50000]
            if dataset:
                args += ["--dataset", dataset, *replay_args]
            if not self.run(f"{skill}_seed{seed}", "skill_refine", args, 2.25,
                            "PPO improves closed-loop success after critic warmup while retaining prerequisite behavior"):
                continue
            report = json.loads((directory / "report.json").read_text())
            self.seeds.record(skill + "_trial_development", directory / "report.json", report["final"]["criterion_met"])
            results.append((report, directory))
        if len(results) != 2:
            return None
        # An initialized actor is a legitimate candidate; improvement is credited to fitting, not PPO.
        candidates = [(r["initial"]["pass_rate"], d / "initialized.zip") for r, d in results]
        candidates += [(r["final"]["pass_rate"], d / "final.zip") for r, d in results]
        score, candidate = max(candidates, key=lambda item: item[0])
        if score >= 0.8:
            return candidate
        gain = all(r["final"]["pass_rate"] - r["initial"]["pass_rate"] >= 0.10 - 1e-9
                   and (r["final_retention"] is None or r["final_retention"]["criterion_met"])
                   and r["stop_reason"] not in {"retention_failed_three_times", "nonfinite_training_metric",
                                                 "repeated_high_or_nonfinite_kl"}
                   for r, _ in results)
        if gain:
            self.status["stages"][skill]["next_evidence"] = "Both seeds improved; one bounded extension is justified"
            directory = self.directory / "trials" / f"{skill}_extension"
            args = [candidate, "--skill", skill, "--output", directory, "--seed", 46,
                    "--steps", 250000, "--hours", 2, "--dev-seed", dev_seed, "--warmup-steps", 50000]
            if dataset:
                args += ["--dataset", dataset, *replay_args]
            if self.run(skill + "_extension", "skill_refine", args, 2.25, "Reproduce the two-seed improvement"):
                report = json.loads((directory / "report.json").read_text())
                if report["final"]["pass_rate"] >= 0.8:
                    return directory / "final.zip"
        return None

    def execute(self):
        try:
            self.verify_preserved()
            self.begin("audit", "Confirm physics, baseline integrity, and fixed failure cases")
            if not self.run("audit", "skill_audit", ["--output", self.directory / "audit",
                            "--checkpoint", self.baseline("recover")], 2, "The control path is internally consistent"):
                self.finish("blocked", "Simulation audit did not pass")
                return
            tilt = self.evaluate(self.baseline("recover"), "recover", "tilt_baseline_development", level=9)
            self.evaluate(self.baseline("recover"), "recover", "tilt_baseline_holdout", "holdout", level=9)
            if tilt:
                self.videos(self.baseline("recover"), "recover", "tilt_baseline", tilt, level=9)
            self.finish("completed", "Audit passed; tilt results do not qualify floor recovery")

            self.begin("nav_squat", "Existing policies retain their documented capabilities after audited fixes")
            for skill in ("nav", "squat"):
                if not self.qualify(self.baseline(skill), skill, skill + "_baseline"):
                    candidate = self.trials(skill, self.baseline(skill))
                    if candidate:
                        self.qualify(candidate, skill, skill + "_repair")
            self.finish("completed" if all(k in self.status["accepted"] for k in ("nav", "squat")) else "blocked",
                        "See recorded development and holdout results")

            for skill in ("recover", "run", "jump"):
                if self.stopping or time.time() - self.started >= 72 * 3600:
                    break
                self.begin(skill, "Demonstrate a feasible motion, then test bounded neural-policy improvement")
                dataset = self.teachers(skill) if skill in {"recover", "jump"} else None
                if skill in {"recover", "jump"} and dataset is None:
                    self.finish("blocked", "Required teacher directions failed independent validation; PPO not launched")
                    continue
                prerequisite = "nav" if skill == "run" else "squat" if skill == "jump" else "recover"
                source = Path(self.status["accepted"].get(prerequisite, self.baseline(prerequisite)))
                candidate = self.trials(skill, source, dataset)
                passed = candidate is not None and self.qualify(candidate, skill, skill + "_candidate")
                self.finish("completed" if passed else "blocked", "Qualification passed" if passed else
                            "Bounded trials did not meet qualification; retain prior policies and review evidence")

            self.begin("integration", "Accepted policies complete continuous command chains, including physical falls")
            required = {"nav", "squat", "recover", "run", "jump"}
            if not required <= self.status["accepted"].keys():
                self.finish("blocked", "Missing accepted skills: " + ", ".join(sorted(required - self.status["accepted"].keys())))
            else:
                passed = True
                for mode in ("prepared", "dynamic"):
                    for role in ("development", "holdout"):
                        label = f"chain_{mode}_{role}"
                        seed = self.seeds.reserve(label, role, 50)
                        output = self.directory / "evaluations" / f"{label}.json"
                        ok = self.run(label, "skill_supervisor", [self.directory / "accepted", "--seed-start", seed,
                                      "--seeds", 50, "--fall-mode", mode, "--output", output], 2,
                                      "Complete all commands without simulator resets")
                        result = json.loads(output.read_text()) if ok else {"criterion_met": False}
                        self.seeds.record(label, output, result["criterion_met"])
                        passed &= result["criterion_met"]
                        if not passed:
                            break
                    if not passed:
                        break
                self.finish("completed" if passed else "blocked", "See chain reports")
        finally:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            self.verify_preserved()
            self.status["state"] = ("stopped" if self.stopping else "completed" if
                self.status["stages"].get("integration", {}).get("state") == "completed" else "review_required")
            self.status["next_decision"] = "Review per-stage evidence; no automatic parameter-search restart"
            self.publish()
            lines = ["# Campaign Results", "", f"State: **{self.status['state']}**", "",
                     f"Elapsed machine-hours: {self.status['elapsed_hours']:.2f} / 72", "",
                     "| Stage | State | Evidence / reason |", "| --- | --- | --- |"]
            for stage, row in self.status["stages"].items():
                lines.append(f"| {stage} | {row['state']} | {row.get('reason', 'Interrupted')} |")
            lines += ["", "Only checkpoints under `accepted/` passed this campaign's qualification.",
                      "Tilt-balance baselines do not establish floor recovery. All results are simulation-only.",
                      "", f"Completed child CPU-hours: {self.status['completed_child_cpu_hours']:.2f}",
                      "The repair reserve remains unused unless a specific reproducible defect justifies it."]
            (self.directory / "RESULTS.md").write_text("\n".join(lines) + "\n")
            print(f"[campaign] {self.status['state']}; report={self.directory / 'RESULTS.md'}", flush=True)


def watch(directory: Path, logs: bool = False) -> None:
    offsets = {}
    while True:
        path = directory / "status.json"
        if not path.exists():
            print("Campaign has not started.", flush=True)
            return
        state = json.loads(path.read_text())
        age = time.time() - state["heartbeat_at"]
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {state['state']} | stage={state['stage']} | "
              f"{state['elapsed_hours']:.2f}/72h | heartbeat_age={age:.0f}s | accepted={list(state['accepted'])}", flush=True)
        for job in state["jobs"]:
            if job["state"] == "running":
                print(f"  {job['name']} pid={job['pid']} log={job['log']}", flush=True)
        if logs and state["jobs"]:
            job = state["jobs"][-1]
            path = Path(job["log"])
            if path.exists():
                with path.open() as handle:
                    handle.seek(offsets.get(str(path), 0))
                    content = handle.read()
                    offsets[str(path)] = handle.tell()
                if content:
                    print(content[-12000:], end="", flush=True)
        if state["state"] != "running":
            print(f"Stopped. Read {directory / 'RESULTS.md'}", flush=True)
            return
        if age > 90:
            print("STALE HEARTBEAT: inspect the campaign process; training health is unknown.", flush=True)
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--logs", action="store_true", help="With --watch, follow the current experiment log")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--nav-checkpoint", type=Path)
    parser.add_argument("--squat-checkpoint", type=Path)
    parser.add_argument("--recovery-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Run the versioned running-first campaign")
    parser.add_argument("--budget-started-at", type=float, help="Preserve an earlier campaign's wall-clock budget")
    args = parser.parse_args()
    if args.config:
        from .run_campaign import RunningCampaign, watch as watch_running
        if args.watch:
            watch_running(args.directory, args.logs)
        else:
            RunningCampaign(args).execute()
        return
    if args.init:
        if args.watch or any(path is None for path in (args.nav_checkpoint, args.squat_checkpoint, args.recovery_checkpoint)):
            parser.error("--init requires nav, squat, and recovery checkpoints and cannot be combined with --watch")
        initialize_campaign(args.directory, {"walk": ROOT / "mujoco_rl/checkpoints/full_body_v1/policy.zip",
                            "nav": args.nav_checkpoint, "squat": args.squat_checkpoint,
                            "recover": args.recovery_checkpoint})
    elif args.watch:
        watch(args.directory, args.logs)
    else:
        Campaign(args.directory).execute()


if __name__ == "__main__":
    main()
