"""Bounded experiments and explicit evidence gates before long walking training."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_correction import BASELINE, Correction, paired
from .command_walk_fast_experiments import HEADING, AuditCallback, evaluate, jobs_for, status
from .command_walk_guarded_ppo import GuardedPPO
from .command_walk_train import save_bundle
from .command_walk_v4 import BRAKES, Spec
from .command_walk_v4_train import build
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path

CANDIDATE = Correction("speed_002", heading_gain=0.5, turn_speed_reduction=0.02)


class Callback(AuditCallback):
    def _on_rollout_start(self):
        super()._on_rollout_start()
        # Hold exploration fixed to isolate mean-policy optimization from std learning.
        self.model.policy.log_std.requires_grad_(False)


def train_job(job):
    checkpoint, root, seed, scale, enabled, rollouts, deadline = job
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with (
        (root / "train.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        norm = None
        try:
            model, norm, rng = build(Path(checkpoint), Spec(), seed)
            model.__class__ = GuardedPPO
            model.audit_lr = 3e-5
            model.guard_enabled = enabled
            model.guard_mean_kl, model.guard_max_kl = 0.01, 0.1
            model.guard_path = root / "guard.jsonl"
            model._diagnostics_path = root / "updates.jsonl"
            with torch.no_grad():
                model.policy.log_std.add_(float(np.log(scale)))
            actor_start = model.command_walk_state["warmup_until"]
            checkpoints = {}

            def after_update(updated):
                number = (updated.num_timesteps - actor_start) // 6144
                if number in (0, 1, 4, rollouts):
                    checkpoints[str(number)] = str(save_bundle(updated, norm, root, f"audit_{number}").resolve())
                    atomic_json(root / "checkpoints.json", checkpoints)
                status(root, "updated", actor_rollouts=max(0, number), steps=updated.num_timesteps)

            model.audit_hook = after_update
            atomic_json(
                root / "manifest.json",
                dict(
                    seed=seed,
                    scale=scale,
                    guarded=enabled,
                    actor_rollouts=rollouts,
                    mean_kl_limit=0.01,
                    frame_kl_limit=0.1,
                    max_attempts=9,
                    learning_rate=3e-5,
                    frozen_std=True,
                    correction="none",
                    warmup_rollouts=2,
                    rng_hash=rng,
                    source_sha256=sha256(Path(checkpoint)),
                ),
            )
            model.learn(
                total_timesteps=(2 + rollouts) * 6144, reset_num_timesteps=False, callback=Callback(root, deadline)
            )
            atomic_json(root / "status.json", dict(state="completed", heartbeat=time.time()))
            return str(root), checkpoints
        except BaseException as error:
            atomic_json(root / "status.json", dict(state="failed", heartbeat=time.time(), error=repr(error)))
            raise
        finally:
            if norm is not None:
                norm.close()


def train_group(root, jobs, stage):
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(train_job, job) for job in jobs}
        while pending:
            finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                results.append(future.result())
            progress = {}
            for job in jobs:
                path = Path(job[1]) / "status.json"
                if path.exists():
                    progress[path.parent.name] = json.loads(path.read_text())
            status(root, stage, completed=len(results), total=len(jobs), jobs=progress)
            print(f"[{stage}] {len(results)}/{len(jobs)} complete; " + str(progress), flush=True)
    return results


def read(path):
    return json.loads(Path(path).read_text())


def retained(before, after):
    return all(after["families"][k]["pass_rate"] + 0.040001 >= before["families"][k]["pass_rate"] for k in FAMILIES)


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started, deadline = time.time(), time.time() + args.minutes * 60
    checkpoint = root / "baseline" / "policy.zip"
    checkpoint.parent.mkdir()
    shutil.copy2(args.checkpoint, checkpoint)
    shutil.copy2(_stats_path(args.checkpoint), _stats_path(checkpoint))
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    manifest = dict(
        started=started,
        deadline=deadline,
        source=str(args.checkpoint.resolve()),
        policy_sha256=sha256(checkpoint),
        normalization_sha256=sha256(_stats_path(checkpoint)),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        code_sha256={p.name: sha256(p) for p in snapshot.glob("*.py")},
        candidate=asdict(CANDIDATE),
        controller_panels=dict(
            sequence_count=50,
            sequence_starts=[12_000_001, 14_000_001],
            family_count=50,
            family_start=16_000_001,
            brake_count=20,
        ),
        hard_turns=dict(
            speeds=[0.2, 0.25, 0.3],
            yaw=[0.4, 0.5, 0.6],
            directions=["left_arc", "right_arc"],
            seeds_per_cell=10,
            seed_start=18_000_001,
        ),
        optimization=dict(
            scales=[0.05, 0.1],
            seeds=[44, 45],
            learning_rate=3e-5,
            fixed_std=True,
            rollouts=16,
            mean_kl=0.01,
            max_frame_kl=0.1,
            attempts=9,
            controls="same initialization and frozen std, one unguarded actor rollout",
        ),
        gates=dict(
            technical="All accepted full-rollout mean KL <= .01, max frame KL <= .1; exact rejected-state restoration; "
            "nonzero actor movement and at least one accepted proposal per actor rollout",
            learning="At a common noise setting, both seeds retain every family within 4pp on development and confirmation, "
            "and improve minimum family success by >=4pp on BOTH panels without reducing mean success",
            controller="Candidate fresh families >=48/50 each; both long-sequence panels >=48/50; "
            "no family retention decline >4pp versus heading-only controller",
            long_run="No automatic long run. A learning candidate must additionally retain corrected-controller "
            "performance on a fresh paired panel and long sequences before being recommended.",
        ),
        scope="Nominal simulation preflight; no claims about untested hardware latency or model mismatch",
    )
    atomic_json(root / "manifest.json", manifest)
    (root / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff"]))
    status(root, "starting")
    try:
        sequence_reports = []
        for index, start in enumerate((12_000_001, 14_000_001)):
            jobs = []
            for correction in (HEADING, CANDIDATE):
                jobs += jobs_for(correction.name, correction, 0, ("sequence",), 50, start)
            sequence_reports.append(evaluate(root, checkpoint, f"sequences_{index}", jobs, deadline))
        jobs = []
        for correction in (HEADING, CANDIDATE):
            jobs += jobs_for(correction.name, correction, 0, FAMILIES, 50, 16_000_001)
            jobs += jobs_for(correction.name, correction, 0, BRAKES, 20, 16_000_001)
        fresh = evaluate(root, checkpoint, "controller_fresh", jobs, deadline)
        controller_gate = all(fresh[CANDIDATE.name]["families"][k]["success_count"] >= 48 for k in FAMILIES)
        controller_gate &= retained(fresh[HEADING.name], fresh[CANDIDATE.name])
        controller_gate &= all(
            r[CANDIDATE.name]["families"]["sequence"]["success_count"] >= 48 for r in sequence_reports
        )
        atomic_json(
            root / "controller_gate.json",
            dict(
                passed=controller_gate,
                paired_families=paired(fresh[HEADING.name], fresh[CANDIDATE.name]),
                paired_sequences=[paired(r[HEADING.name], r[CANDIDATE.name]) for r in sequence_reports],
            ),
        )
        hard_jobs = []
        for speed in (0.2, 0.25, 0.3):
            for yaw in (0.4, 0.5, 0.6):
                for case, sign in (("left_arc", 1), ("right_arc", -1)):
                    for correction in (HEADING, CANDIDATE):
                        label = f"v{speed:g}_w{yaw:g}_{correction.name}"
                        hard_jobs += jobs_for(
                            label,
                            correction,
                            0,
                            (case,),
                            10,
                            18_000_001,
                            dict(schedule=[(8.0, speed, sign * yaw), (4.0, 0.0, 0.0)]),
                        )
        evaluate(root, checkpoint, "hard_turns", hard_jobs, deadline)

        # Identical first-rollout controls isolate the optimizer guard, including Adam state.
        controls = [
            (str(checkpoint), str(root / f"control_seed{seed}"), seed, 0.05, False, 1, deadline) for seed in (44, 45)
        ]
        control_results = train_group(root, controls, "unguarded_controls")
        jobs = [
            (str(checkpoint), str(root / f"guard_scale{scale:g}_seed{seed}"), seed, scale, True, 16, deadline)
            for scale in (0.05, 0.1)
            for seed in (44, 45)
        ]
        audit_results = train_group(root, jobs, "guarded_audits")
        reference = {}
        for stage, count, start in (("quick", 10, 20_000_001), ("development", 50, 1001), ("confirmation", 50, 10051)):
            reference[stage] = evaluate(
                root,
                checkpoint,
                f"reference_{stage}",
                jobs_for("baseline", BASELINE, 0, FAMILIES, count, start),
                deadline,
            )["baseline"]
        for folder, checkpoints in control_results:
            evaluate(
                Path(folder),
                Path(checkpoints["1"]),
                "quick_1",
                jobs_for("policy", BASELINE, 0, FAMILIES, 10, 20_000_001),
                deadline,
            )
        reports = {}
        for folder, checkpoints in audit_results:
            folder = Path(folder)
            status(root, "evaluating_guarded", job=folder.name)
            guards = [json.loads(line) for line in (folder / "guard.jsonl").read_text().splitlines()]
            actors = [g for g in guards if g["enabled"]]
            technical = len(actors) == 16 and all(
                g["accepted"] > 0
                and g["mean_action_change_rms"] > 1e-8
                and g["final_mean_kl"] <= 0.0100001
                and g["final_max_kl"] <= 0.1000001
                and g["restoration_checks"] == g["rejected"]
                for g in actors
            )
            reports[folder.name] = dict(technical_passed=technical, guard=actors, panels={})
            for number in (0, 1, 4, 16):
                evaluate(
                    folder,
                    Path(checkpoints[str(number)]),
                    f"quick_{number}",
                    jobs_for("policy", BASELINE, 0, FAMILIES, 10, 20_000_001),
                    deadline,
                )
            for stage, start in (("development", 1001), ("confirmation", 10051)):
                after = evaluate(
                    folder,
                    Path(checkpoints["16"]),
                    stage,
                    jobs_for("policy", BASELINE, 0, FAMILIES, 50, start),
                    deadline,
                )["policy"]
                before = reference[stage]
                reports[folder.name]["panels"][stage] = dict(
                    paired=paired(before, after),
                    retained=retained(before, after),
                    improvement=after["minimum_pass_rate"] >= before["minimum_pass_rate"] + 0.039999
                    and after["mean_pass_rate"] >= before["mean_pass_rate"] - 1e-9,
                    minimum_pass_rate=after["minimum_pass_rate"],
                    mean_pass_rate=after["mean_pass_rate"],
                )
            reports[folder.name]["learning_passed"] = technical and all(
                p["retained"] and p["improvement"] for p in reports[folder.name]["panels"].values()
            )
            atomic_json(folder / "status.json", dict(state="completed", heartbeat=time.time(), evaluated=True))
            atomic_json(root / "partial_results.json", reports)
        eligible = [
            scale
            for scale in (0.05, 0.1)
            if all(reports[f"guard_scale{scale:g}_seed{seed}"]["learning_passed"] for seed in (44, 45))
        ]
        # Only an empirically improving optimizer setting earns additional integration checks.
        integration = {}
        for scale in eligible:
            for seed in (44, 45):
                name = f"guard_scale{scale:g}_seed{seed}"
                candidate = Path(read(root / name / "checkpoints.json")["16"])
                cases = jobs_for("corrected", CANDIDATE, 0, FAMILIES, 50, 22_000_001)
                cases += jobs_for("corrected", CANDIDATE, 0, ("sequence",), 50, 22_000_001)
                base = evaluate(root, checkpoint, f"integration_baseline_{name}", cases, deadline)["corrected"]
                learned = evaluate(root, candidate, f"integration_{name}", cases, deadline)["corrected"]
                integration[name] = dict(
                    paired=paired(base, learned),
                    passed=retained(base, learned)
                    and learned["families"]["sequence"]["pass_rate"]
                    >= max(0.96, base["families"]["sequence"]["pass_rate"] - 0.04),
                )
        ready = bool(eligible) and all(v["passed"] for v in integration.values())
        result = dict(
            elapsed_seconds=time.time() - started,
            controller_gate=controller_gate,
            audits=reports,
            eligible_noise_scales=eligible,
            integration=integration,
            long_training_recommended=ready,
            long_training_started=False,
            decision=(
                "Review evidence before long run"
                if ready
                else "Do not start long training: preflight learning/integration gate not met"
            ),
        )
        if (
            sha256(args.checkpoint) != manifest["policy_sha256"]
            or sha256(_stats_path(args.checkpoint)) != manifest["normalization_sha256"]
        ):
            raise RuntimeError("Original source files changed")
        atomic_json(root / "result.json", result)
        atomic_json(
            root / "status.json",
            dict(
                state="completed",
                heartbeat=time.time(),
                elapsed_seconds=time.time() - started,
                long_training_recommended=ready,
            ),
        )
        print("[completed] " + result["decision"], flush=True)
    except BaseException as error:
        atomic_json(root / "status.json", dict(state="failed", heartbeat=time.time(), error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minutes", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.minutes <= 180:
        parser.error("Use a positive budget of at most 180 minutes")
    os.environ.setdefault("MUJOCO_GL", "disable")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
