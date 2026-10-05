"""Running-first experiment matrix, continuation, qualification, and live status."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from .run_contract import CONTRACT, STAGES, config_from, made_progress
from .skill_campaign import SeedRegistry, atomic_json, initialize_campaign, sha256


ROOT = Path(__file__).resolve().parents[1]


def ranking(report):
    result = report["final"]
    error = result["summary"]["speed_error_m_s"]
    completed = [e["speed_error_m_s"] for e in result["episodes"] if e["seconds"] >= 12 and not e.get("fallen", False)
                 and e.get("speed_error_m_s") is not None]
    return (result["stage"], result["summary"]["pass_rate"],
            -float(np.mean(completed)) if completed else -(error if error is not None else 100.))


def preflight(checkpoint, output, progress=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from .full_body_env import STANDING_POSE
    from .skill_audit import audit_contract
    from .skill_env import SkillEnv
    from .skill_train import _stats_path

    torch.set_num_threads(1)
    audit = audit_contract()
    atomic_json(output / "physics.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Physics contract failed")
    model = PPO.load(checkpoint, device="cpu")
    norm = VecNormalize.load(str(_stats_path(checkpoint)), DummyVecEnv([lambda: SkillEnv("run")]))
    norm.training = False

    class MappingEnv(SkillEnv):
        def _target_from_action(self, action):
            return np.clip(STANDING_POSE + .25 * action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    results = {}
    try:
        for name, cls in (("legacy_035", SkillEnv), ("fixed_025", MappingEnv)):
            env, episodes = cls("run", randomize=True), []
            try:
                for seed in range(3100000000, 3100000050):
                    if progress:
                        progress()
                    obs, _ = env.reset(seed=seed)
                    clips, errors, targets = [], [], []
                    for _ in range(600):
                        action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                        clips.append(float(np.mean(np.abs(action[0]) >= .99)))
                        targets.append(float(np.sqrt(np.mean((env._target_from_action(action[0]) - STANDING_POSE) ** 2))))
                        obs, _, terminated, truncated, info = env.step(action[0])
                        if 100 <= env.step_count < 500:
                            errors.append(abs(float(env.data.sensor("imu_lin_vel").data[0]) - .6))
                        if terminated or truncated:
                            break
                    episodes.append({"seed": seed, "seconds": info["seconds"], "fallen": info["fallen"],
                                     "speed_error": float(np.mean(errors)) if errors else None,
                                     "clipping": float(np.mean(clips)), "target_offset_rms": float(np.mean(targets))})
            finally:
                env.close()
            results[name] = {"episodes": episodes, "fall_rate": float(np.mean([e["fallen"] for e in episodes])),
                             "mean_seconds": float(np.mean([e["seconds"] for e in episodes]))}
            print(f"[preflight] {name}: falls={results[name]['fall_rate']:.0%} "
                  f"survival={results[name]['mean_seconds']:.2f}s", flush=True)
        atomic_json(output / "mapping_comparison.json", results)
    finally:
        norm.close()


class RunningCampaign:
    def __init__(self, args):
        self.directory, self.config = args.directory.resolve(), config_from(args.config)
        if not self.directory.exists():
            if any(p is None for p in (args.nav_checkpoint, args.squat_checkpoint, args.recovery_checkpoint)):
                raise ValueError("New running campaign requires navigation, squat, and balance baselines")
            initialize_campaign(self.directory, {"nav": args.nav_checkpoint, "squat": args.squat_checkpoint,
                                                "recover": args.recovery_checkpoint})
        self.lock = (self.directory / "running.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.directory / "status.json").exists():
            raise FileExistsError("Campaign already started; use its saved bundles for explicit continuation")
        self.registry = json.loads((self.directory / "registry.json").read_text())
        self.nav = Path(self.registry["nav"]["checkpoint"])
        self.started = getattr(args, "budget_started_at", None) or time.time()
        if not np.isfinite(self.started) or not 0 < self.started <= time.time():
            raise ValueError("Invalid original campaign start time")
        self.deadline = self.started + self.config["maximum_hours"] * 3600
        self.training_deadline = self.deadline - self.config["qualification_hours"] * 3600
        self.stopped = False
        self.last_plot = 0.
        self.seeds = SeedRegistry(self.directory / "seeds.json", origin=3101000000)
        # Distinct from previous campaigns' reused registry origin.
        self.dev_seed = 3100100000
        self.status = {"state": "running", "stage": "preflight", "started_at": self.started,
                       "pid": os.getpid(), "contract": CONTRACT, "jobs": [], "accepted": {}}
        atomic_json(self.directory / "config.json", self.config)
        snapshot = self.directory / "source_snapshot"
        snapshot.mkdir()
        for path in (ROOT / "mujoco_rl").glob("*.py"):
            shutil.copy2(path, snapshot / path.name)
        atomic_json(snapshot / "hashes.json", {p.name: sha256(p) for p in snapshot.glob("*.py")})
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGALRM, self.timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.deadline - time.time()))

    def stop(self, *_):
        self.stopped = True

    def timeout(self, *_):
        self.stopped = True
        raise TimeoutError("Campaign wall-clock ceiling reached")

    def publish(self):
        self.status.update(heartbeat=time.time(), elapsed_hours=(time.time() - self.started) / 3600,
                           remaining_hours=max(0., self.deadline - time.time()) / 3600)
        for job in self.status["jobs"]:
            path = Path(job["output"]) / "status.json"
            if path.exists():
                job["training"] = json.loads(path.read_text())
        atomic_json(self.directory / "status.json", self.status)
        if time.time() - self.last_plot > 300 and self.status["jobs"]:
            self.curves()

    def curves(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        self.last_plot = time.time()
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for job in self.status["jobs"]:
            path = Path(job["output"])
            rows = []
            initial = path / "initial.json"
            if initial.exists():
                initial_report = json.loads(initial.read_text())
                rows.append((initial_report.get("training_steps", 0), initial_report))
            for evaluation in sorted((path / "evaluations").glob("*.json"), key=lambda p: int(p.stem)):
                rows.append((int(evaluation.stem), json.loads(evaluation.read_text())))
            if not rows:
                continue
            for axis, key, title in zip(axes.flat,
                    ("pass_rate", "survival_seconds", "speed_error_m_s", "stage"),
                    ("Stage success fraction", "Mean survival (seconds)", "Speed error (m/s)", "Curriculum stage")):
                values = [r["stage"] if key == "stage" else r["summary"][key] for _, r in rows]
                axis.plot([step for step, _ in rows], values, marker=".", label=job["name"])
                axis.set(title=title, xlabel="Total environment steps")
                axis.grid(alpha=.25)
        if axes[0, 0].lines:
            axes[0, 0].legend(fontsize=6)
        temporary = self.directory / "learning_curves.tmp.png"
        fig.savefig(temporary, dpi=120)
        plt.close(fig)
        temporary.replace(self.directory / "learning_curves.png")

    def verify_preserved(self):
        for record in self.registry.values():
            for name, digest in record["sha256"].items():
                if sha256(Path(record["checkpoint"]).parent / name) != digest:
                    raise RuntimeError("Preserved baseline modified")

    def jobs(self, specifications):
        pending, active, reports = list(specifications), [], {}
        try:
            while pending or active:
                while pending and len(active) < 2 and time.time() < self.training_deadline and not self.stopped:
                    name, config, seed, checkpoint, steps = pending.pop(0)
                    path = self.directory / "trials" / name
                    cfg = self.directory / "configs" / f"{name}.json"
                    atomic_json(cfg, config)
                    log = self.directory / "logs" / f"{name}.log"
                    log.parent.mkdir(exist_ok=True)
                    handle = log.open("w")
                    command = [sys.executable, "-u", "-m", "mujoco_rl.skill_refine", str(checkpoint),
                               "--skill", "run", "--config", str(cfg), "--output", str(path),
                               "--seed", str(seed), "--dev-seed", str(self.dev_seed), "--steps", str(steps),
                               "--hours", str(max(.001, (self.training_deadline - time.time()) / 3600))]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True,
                                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "osmesa"})
                    job = {"name": name, "pid": process.pid, "output": str(path), "log": str(log),
                           "state": "running", "seed": seed, "initial_std": config["initial_std"]}
                    self.status["jobs"].append(job)
                    active.append((process, handle, job))
                    print(f"[campaign] started {name}; log={log}", flush=True)
                if (self.stopped or time.time() >= self.training_deadline) and pending:
                    self.status["unstarted_jobs"] = [p[0] for p in pending]
                    pending.clear()
                for process, handle, job in active[:]:
                    if (self.stopped or time.time() >= self.training_deadline) and process.poll() is None:
                        process.send_signal(signal.SIGTERM)
                    if process.poll() is not None:
                        handle.close()
                        job.update(state="completed" if process.returncode == 0 else "failed", exit_code=process.returncode)
                        report = Path(job["output"]) / "report.json"
                        if process.returncode == 0 and report.exists():
                            reports[job["name"]] = json.loads(report.read_text())
                        active.remove((process, handle, job))
                        print(f"[campaign] finished {job['name']}: {job['state']}", flush=True)
                self.publish()
                if active:
                    time.sleep(5)
            return reports
        finally:
            for process, handle, job in active:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                handle.close()

    def qualify(self, checkpoint, attempt):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from .run_contract import RunningEnv
        from .run_training import evaluate_stage
        from .run_supervisor import evaluate_transitions
        from .skill_train import _stats_path

        torch.set_num_threads(1)
        model = PPO.load(checkpoint, device="cpu")
        norm = VecNormalize.load(str(_stats_path(checkpoint)), DummyVecEnv([lambda: RunningEnv(model.run_contract, 7)]))
        norm.training, norm.norm_reward = False, False
        output = self.directory / "qualification" / f"candidate_{attempt}"
        try:
            # Inspected failed confirmations remain mandatory regression cases.
            panels = [(name, p["start"], p["count"]) for name, p in self.seeds.panels.items()
                      if p["role"] == "development_retry"]
            panels.append(("development", self.dev_seed + 10000, 50))
            for name, seed, count in panels:
                report = evaluate_stage(model, norm, 7, seed, count)
                atomic_json(output / f"{name}.json", report)
                if report["summary"]["pass_rate"] < .8:
                    return False
            name = f"holdout_{attempt}"
            seed = self.seeds.reserve(name, "holdout", 50)
            report = evaluate_stage(model, norm, 7, seed, 50)
            path = output / "holdout.json"
            atomic_json(path, report)
            passed = report["summary"]["pass_rate"] >= .8
            self.seeds.record(name, path, passed)
            if not passed:
                return False
            transitions = evaluate_transitions(self.nav, checkpoint, self.dev_seed + 20000, 20)
            atomic_json(output / "transitions.json", transitions)
            if not transitions["criterion_met"]:
                return False
            for success in (True, False):
                case = next((e for e in report["episodes"] if e["success"] == success), None)
                if case:
                    evaluate_stage(model, norm, 7, case["seed"], 1, output / f"holdout_{success}.mp4")
            accepted = self.directory / "accepted" / "run"
            accepted.mkdir(parents=True)
            shutil.copy2(checkpoint, accepted / "accepted.zip")
            shutil.copy2(_stats_path(checkpoint), accepted / "accepted_vecnormalize.pkl")
            atomic_json(accepted / "contract.json", model.run_contract)
            atomic_json(accepted / "qualification.json", {"status": "accepted_simulation_only",
                "environment_version": 12, "training_contract": CONTRACT, "reports": str(output),
                "sha256": {"accepted.zip": sha256(accepted / "accepted.zip")},
                "normalization_sha256": sha256(accepted / "accepted_vecnormalize.pkl")})
            self.status["accepted"]["run"] = str(accepted / "accepted.zip")
            return True
        finally:
            norm.close()

    def execute(self):
        reports = {}
        try:
            print(f"[campaign] running-first preflight; evidence={self.directory}; "
                  f"ceiling={self.config['maximum_hours']}h", flush=True)
            self.publish()
            preflight(self.nav, self.directory / "preflight", self.publish)
            self.verify_preserved()
            self.status["stage"] = "exploration_matrix"
            specs = [(f"std{std:.2f}_seed{seed}_block1", {**self.config, "initial_std": std}, seed,
                      self.nav, self.config["block_steps"])
                     for std in self.config["exploration_settings"] for seed in self.config["seeds"]]
            reports = self.jobs(specs)
            viable = []
            for std in self.config["exploration_settings"]:
                names = [f"std{std:.2f}_seed{seed}_block1" for seed in self.config["seeds"]]
                if all(n in reports and reports[n]["stop_reason"] in ("step_budget", "ready_for_qualification") for n in names):
                    scores = [ranking(reports[n]) for n in names]
                    viable.append((min(scores), float(np.mean([s[1] for s in scores])), -std, names))
            if not viable:
                self.status["reason"] = "No complete two-seed experiment; inspect child reports/errors"
                return
            _, _, neg_std, selected = max(viable)
            std = -neg_std
            self.status.update(stage="continuation", selected_initial_std=std)
            branches = {seed: {"report": reports[name], "stalls": 0 if made_progress(reports[name]["initial"], reports[name]["final"]) else 1,
                               "block": 1} for seed, name in zip(self.config["seeds"], selected)}
            while not self.stopped and time.time() < self.training_deadline:
                specs = []
                for seed, branch in branches.items():
                    report = branch["report"]
                    if (report["stop_reason"] != "step_budget" or branch["stalls"] >= 2
                            or self.config["maximum_steps"] - report["steps"] < self.config["workers"] * self.config["rollout_steps"]):
                        continue
                    branch["block"] += 1
                    specs.append((f"std{std:.2f}_seed{seed}_block{branch['block']}", {**self.config, "initial_std": std}, seed,
                                  Path(report["checkpoint"]), min(self.config["block_steps"], self.config["maximum_steps"] - report["steps"])))
                if not specs:
                    break
                additions = self.jobs(specs)
                reports.update(additions)
                for name, _, seed, _, _ in specs:
                    branch = branches[seed]
                    if name not in additions:
                        branch["stalls"] = 2
                        continue
                    report = additions[name]
                    branch["stalls"] = 0 if made_progress(report["initial"], report["final"]) else branch["stalls"] + 1
                    branch["report"] = report
                atomic_json(self.directory / "branches.json", branches)
            self.status["stage"] = "qualification"
            self.publish()
            for attempt, report in enumerate(sorted(reports.values(), key=ranking, reverse=True)):
                if self.stopped or time.time() >= self.deadline:
                    break
                if report["final"]["stage"] == 7 and report["final"]["summary"]["pass_rate"] >= .8:
                    if self.qualify(Path(report["checkpoint"]), attempt):
                        self.status["state"] = "accepted"
                        break
            self.status.setdefault("reason", "Training ended at qualification, a measured plateau, or the configured budget")
        except TimeoutError as error:
            self.status.update(state="incomplete", reason=str(error))
        except Exception as error:
            self.status.update(state="failed", reason=repr(error))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self.status["state"] == "running":
                self.status["state"] = "incomplete"
            self.verify_preserved()
            self.curves()
            self.publish()
            rows = ["# Running campaign results", "", f"State: **{self.status['state']}**", "",
                    f"Elapsed machine-hours: {self.status['elapsed_hours']:.2f}", "",
                    "| Trial | Stage | Pass rate | Survival | Speed error | Stop |",
                    "|---|---|---:|---:|---:|---|"]
            for name, report in reports.items():
                result, metric = report["final"], report["final"]["summary"]
                rows.append(f"| {name} | {result['stage_name']} | {metric['pass_rate']:.0%} | "
                            f"{metric['survival_seconds']:.2f}s | {metric['speed_error_m_s']} | {report['stop_reason']} |")
            rows += ["", "Only accepted/ contains qualified policies. Partial curriculum success is not running qualification.",
                     "", str(self.status.get("reason", ""))]
            (self.directory / "RESULTS.md").write_text("\n".join(rows) + "\n")


def watch(directory, logs=False):
    offsets = {}
    while True:
        path = directory / "status.json"
        if not path.exists():
            print("Waiting for running campaign status", flush=True)
            time.sleep(5)
            continue
        status = json.loads(path.read_text())
        print(f"[status] state={status['state']} phase={status['stage']} "
              f"elapsed={status.get('elapsed_hours', 0):.2f}h "
              f"remaining={status.get('remaining_hours', 0):.2f}h", flush=True)
        if logs:
            for job in status.get("jobs", []):
                log = Path(job["log"])
                if log.exists():
                    with log.open() as f:
                        f.seek(offsets.get(str(log), max(0, log.stat().st_size - 6000)))
                        content = f.read()
                        offsets[str(log)] = f.tell()
                    if content:
                        print(f"\n[{job['name']}]\n{content}", flush=True)
        else:
            print(json.dumps(status, indent=2), flush=True)
        if status["state"] != "running":
            print(f"Campaign finished: {status['state']}", flush=True)
            return
        time.sleep(10)
