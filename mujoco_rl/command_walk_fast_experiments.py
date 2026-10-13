"""Bounded capability, exploration, braking, and first-update PPO experiments."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .command_walk import FAMILIES
from .command_walk_correction import BASELINE, Correction, evaluate_batch, initialize_worker, paired
from .command_walk_robust_eval import summarize
from .command_walk_train import save_bundle
from .command_walk_v4 import BRAKES, CASES, Spec
from .command_walk_v4_train import DiagnosticPPO, build
from .run_training import actor_parameters
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path

HEADING = Correction("heading_050", heading_gain=0.5)


def jobs_for(label, correction, scale, cases, count, seed_start, options=None):
    return [
        (
            label,
            correction,
            scale,
            case,
            list(
                range(
                    seed_start + CASES.index(case) * 100_000 + i,
                    seed_start + CASES.index(case) * 100_000 + min(i + 10, count),
                )
            ),
            options or {},
        )
        for case in cases
        for i in range(0, count, 10)
    ]


def status(root, stage, **extra):
    atomic_json(root / "status.json", dict(state="running", stage=stage, heartbeat=time.time(), **extra))


def evaluate(root, checkpoint, stage, jobs, deadline, workers=8):
    collected = {}
    for label, _, _, case, *_ in jobs:
        collected.setdefault(label, {}).setdefault(case, [])
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(str(checkpoint), deadline),
    ) as pool:
        pending = [pool.submit(evaluate_batch, job) for job in jobs]
        for index, future in enumerate(as_completed(pending), 1):
            label, case, rows = future.result()
            collected[label][case].extend(rows)
            atomic_json(
                root / "batches" / stage / f"{label}_{case}_{min(r['seed'] for r in rows)}.json", dict(episodes=rows)
            )
            status(root, stage, completed_batches=index, total_batches=len(jobs))
            if index % 10 == 0 or index == len(jobs):
                print(f"[{stage}] {index}/{len(jobs)} batches", flush=True)
    reports = {}
    for label, families in collected.items():
        for rows in families.values():
            rows.sort(key=lambda r: r["seed"])
        reports[label] = summarize(families)
        atomic_json(root / stage / f"{label}.json", reports[label])
        print(
            f"[{stage}/{label}] " + str({k: v["success_count"] for k, v in reports[label]["families"].items()}),
            flush=True,
        )
    return reports


class AuditPPO(DiagnosticPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["audit_hook"]

    def train(self):
        super().train()
        self.audit_hook(self)


class AuditCallback(BaseCallback):
    def __init__(self, root, deadline):
        super().__init__()
        self.root, self.deadline = root, deadline

    def _on_rollout_start(self):
        enabled = self.model.num_timesteps >= self.model.command_walk_state["warmup_until"]
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)
        if enabled:
            lr = self.model.audit_lr
            self.model.learning_rate = lr
            self.model.lr_schedule = lambda _: lr
        status(self.root, "actor_updates" if enabled else "critic_calibration", steps=self.model.num_timesteps)

    def _on_step(self):
        if time.time() > self.deadline:
            raise TimeoutError("PPO audit deadline exceeded")
        return True


def train_audit(job):
    checkpoint, root, seed, lr, scale, deadline = job
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with (
        (root / "train.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        model = norm = None
        try:
            # Preserve critic and return statistics; two identical critic-only warmup rollouts.
            model, norm, rng = build(Path(checkpoint), Spec(), seed)
            model.__class__ = AuditPPO
            model.audit_lr = lr
            model._diagnostics_path = root / "updates.jsonl"
            with torch.no_grad():
                model.policy.log_std.add_(float(np.log(scale)))
            initial_std = model.policy.log_std.detach().exp().cpu().numpy().tolist()
            actor_start = model.command_walk_state["warmup_until"]
            checkpoints = {}

            def after_update(updated):
                number = (updated.num_timesteps - actor_start) // (8 * 768)
                if number in (0, 1, 4, 16):
                    checkpoints[str(number)] = str(save_bundle(updated, norm, root, f"audit_{number}").resolve())
                    atomic_json(root / "checkpoints.json", checkpoints)
                status(root, "updated", actor_rollouts=max(0, number), steps=updated.num_timesteps)

            model.audit_hook = after_update
            atomic_json(
                root / "manifest.json",
                dict(
                    seed=seed,
                    learning_rate=lr,
                    noise_scale=scale,
                    initial_std=initial_std,
                    rng_hash=rng,
                    warmup_rollouts=2,
                    actor_rollouts=16,
                    source_sha256=sha256(Path(checkpoint)),
                    correction="none",
                    exploration="independent Gaussian",
                    calibration_learning_rate=3e-5,
                    observation_statistics="frozen",
                ),
            )
            model.learn(total_timesteps=18 * 8 * 768, reset_num_timesteps=False, callback=AuditCallback(root, deadline))
            atomic_json(root / "status.json", dict(state="completed", heartbeat=time.time(), actor_rollouts=16))
            return str(root), checkpoints
        except BaseException as error:
            atomic_json(root / "status.json", dict(state="failed", error=repr(error), heartbeat=time.time()))
            raise
        finally:
            if norm is not None:
                norm.close()


def quality_report(reports):
    """Summarize rewards versus physical success without introducing new reward terms."""
    result = {}
    for label, report in reports.items():
        result[label] = {}
        for case, family in report["families"].items():
            groups = {}
            for success in (False, True):
                rows = [r for r in family["episodes"] if r["success"] == success]
                if rows:
                    groups[str(success)] = dict(
                        count=len(rows),
                        mean_reward=float(np.mean([r["mean_reward"] for r in rows])),
                        components={
                            k: float(np.mean([r["reward_components"][k] for r in rows]))
                            for k in rows[0]["reward_components"]
                        },
                    )
            result[label][case] = groups
    return result


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started, deadline = time.time(), time.time() + args.minutes * 60
    (root / "baseline").mkdir()
    checkpoint = root / "baseline" / "policy.zip"
    shutil.copy2(args.checkpoint, checkpoint)
    shutil.copy2(_stats_path(args.checkpoint), _stats_path(checkpoint))
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, snapshot / p.name)
    manifest = dict(
        started=started,
        deadline=deadline,
        source=str(args.checkpoint.resolve()),
        policy_sha256=sha256(checkpoint),
        normalization_sha256=sha256(_stats_path(checkpoint)),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        code_sha256={p.name: sha256(p) for p in snapshot.glob("*.py")},
        map=dict(
            speeds=[0.1, 0.2, 0.3],
            yaw_magnitudes=[0.2, 0.4, 0.6],
            seeds_per_cell=10,
            reductions=[0, 0.02, 0.04],
            seed_start=8_000_001,
        ),
        exploration=dict(scales=[0.05, 0.1, 0.25], correlation_seconds=0.1, seeds_per_family=20, seed_start=9_000_001),
        selection="Speed reduction: paired grid minimum then mean success then lower violation; prefer smaller change on tie. "
        "Noise: largest independent scale retaining every family within 5pp; fallback .05 diagnostic only.",
        ppo=dict(
            seeds=[44, 45],
            rates=[3e-5, 3e-6],
            actor_rollouts=[1, 4, 16],
            warmup_rollouts=2,
            quick_seed=11_000_001,
            development_seed=1001,
            confirmation_seed=10051,
        ),
        scope="Nominal simulation diagnostics; no automatic release or long training; original scoring unchanged",
    )
    atomic_json(root / "manifest.json", manifest)
    (root / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff"]))
    try:
        map_jobs = []
        for speed in (0.1, 0.2, 0.3):
            for yaw in (0.2, 0.4, 0.6):
                for reduction in (0.0, 0.02, 0.04):
                    correction = Correction(f"reduce_{reduction:g}", heading_gain=0.5, turn_speed_reduction=reduction)
                    label = f"v{speed:g}_w{yaw:g}_reduce{reduction:g}"
                    for case, sign in (("left_arc", 1), ("right_arc", -1)):
                        map_jobs += jobs_for(
                            label,
                            correction,
                            0,
                            (case,),
                            10,
                            8_000_001,
                            dict(schedule=[(8.0, speed, sign * yaw), (4.0, 0.0, 0.0)]),
                        )
        mapping = evaluate(root, checkpoint, "capability", map_jobs, deadline)
        scores = {}
        for reduction in (0.0, 0.02, 0.04):
            families = [
                f
                for label, report in mapping.items()
                if label.endswith(f"reduce{reduction:g}")
                for f in report["families"].values()
            ]
            scores[reduction] = (
                min(f["pass_rate"] for f in families),
                float(np.mean([f["pass_rate"] for f in families])),
                -float(np.mean([f["mean_violation"] for f in families])),
                -reduction,
            )
        reduction = max(scores, key=scores.get)
        speed_candidate = Correction("selected_speed", heading_gain=0.5, turn_speed_reduction=reduction)
        atomic_json(root / "speed_selection.json", dict(configuration=asdict(speed_candidate), scores=scores))

        noise_jobs = jobs_for("deterministic", BASELINE, 0, FAMILIES, 20, 9_000_001)
        for scale in (0.05, 0.1, 0.25):
            for kind, rho in (("independent", 0.0), ("correlated", float(np.exp(-0.02 / 0.1)))):
                noise_jobs += jobs_for(
                    f"{kind}_{scale:g}", BASELINE, scale, FAMILIES, 20, 9_000_001, dict(noise_rho=rho)
                )
        noise = evaluate(root, checkpoint, "exploration", noise_jobs, deadline)
        eligible = [
            s
            for s in (0.05, 0.1, 0.25)
            if all(
                noise[f"independent_{s:g}"]["families"][f]["pass_rate"] + 0.050001
                >= noise["deterministic"]["families"][f]["pass_rate"]
                for f in FAMILIES
            )
        ]
        scale = max(eligible) if eligible else 0.05
        atomic_json(
            root / "noise_selection.json",
            dict(
                scale=scale,
                passed=bool(eligible),
                correlated_noise="Evaluation only; not used in PPO because that requires a matching likelihood model",
            ),
        )

        brake_jobs = []
        brake_arms = [HEADING] + [
            Correction(f"deadband_{deg}", heading_gain=0.5, stop_heading_deadband=float(np.deg2rad(deg)))
            for deg in (2, 5)
        ]
        for c in brake_arms:
            brake_jobs += jobs_for(c.name, c, 0, (*BRAKES, "mixed", "sequence"), 20, 10_000_001)
        brakes = evaluate(root, checkpoint, "braking", brake_jobs, deadline)
        atomic_json(root / "reward_quality.json", quality_report({**noise, **brakes}))

        # Lock speed candidate before standard evaluation; do not tune on confirmation.
        for stage, seed_start in (("development", 1001), ("confirmation", 10051)):
            jobs = jobs_for("heading", HEADING, 0, FAMILIES, 50, seed_start)
            if reduction:
                jobs += jobs_for("speed_candidate", speed_candidate, 0, FAMILIES, 50, seed_start)
            reports = evaluate(root, checkpoint, stage, jobs, deadline)
            if reduction:
                atomic_json(root / stage / "paired.json", paired(reports["heading"], reports["speed_candidate"]))

        status(root, "ppo_audit")
        training_jobs = [
            (str(checkpoint), str(root / f"ppo_lr{lr:g}_seed{seed}"), seed, lr, scale, deadline)
            for lr in (3e-5, 3e-6)
            for seed in (44, 45)
        ]
        audits = []
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            pending = {pool.submit(train_audit, job) for job in training_jobs}
            while pending:
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    audits.append(future.result())
                    print(f"[ppo] completed {audits[-1][0]}", flush=True)
                progress = {}
                for job in training_jobs:
                    path = Path(job[1]) / "status.json"
                    if path.exists():
                        progress[Path(job[1]).name] = json.loads(path.read_text())
                status(root, "ppo_audit", completed_jobs=len(audits), total_jobs=4, jobs=progress)
                print(f"[ppo] {len(audits)}/4 jobs complete; " + str(progress), flush=True)
        reference_jobs = jobs_for("baseline", BASELINE, 0, FAMILIES, 10, 11_000_001)
        evaluate(root, checkpoint, "ppo_reference_quick", reference_jobs, deadline)
        for stage, start in (("development", 1001), ("confirmation", 10051)):
            evaluate(
                root,
                checkpoint,
                f"ppo_reference_{stage}",
                jobs_for("baseline", BASELINE, 0, FAMILIES, 50, start),
                deadline,
            )
        for folder, checkpoints in audits:
            audit_root = Path(folder)
            status(root, "ppo_evaluation", job=audit_root.name)
            for number in (0, 1, 4, 16):
                evaluate(
                    audit_root,
                    Path(checkpoints[str(number)]),
                    f"quick_{number}",
                    jobs_for("policy", BASELINE, 0, FAMILIES, 10, 11_000_001),
                    deadline,
                )
            for stage, start in (("development", 1001), ("confirmation", 10051)):
                evaluate(
                    audit_root,
                    Path(checkpoints["16"]),
                    stage,
                    jobs_for("policy", BASELINE, 0, FAMILIES, 50, start),
                    deadline,
                )
            atomic_json(audit_root / "status.json", dict(state="completed", heartbeat=time.time(), evaluated=True))
        summary = dict(
            speed_reduction=reduction, noise_scale=scale, noise_retention_passed=bool(eligible), qualified=False, ppo={}
        )
        for folder, _ in audits:
            audit_root = Path(folder)
            summary["ppo"][audit_root.name] = {}
            for stage in ("development", "confirmation"):
                before = json.loads((root / f"ppo_reference_{stage}" / "baseline.json").read_text())
                after = json.loads((audit_root / stage / "policy.json").read_text())
                comparison = paired(before, after)
                summary["ppo"][audit_root.name][stage] = dict(
                    paired=comparison,
                    retained=all(v["corrected"] + 0.040001 >= v["baseline"] for v in comparison.values()),
                    minimum_pass_rate=after["minimum_pass_rate"],
                    mean_pass_rate=after["mean_pass_rate"],
                )
        atomic_json(root / "result.json", summary)
        if (
            sha256(args.checkpoint) != manifest["policy_sha256"]
            or sha256(_stats_path(args.checkpoint)) != manifest["normalization_sha256"]
        ):
            raise RuntimeError("Original baseline files changed")
        atomic_json(
            root / "status.json",
            dict(state="completed", heartbeat=time.time(), elapsed_seconds=time.time() - started, qualified=False),
        )
        print("[completed] all diagnostics saved; no policy promoted", flush=True)
    except BaseException as error:
        atomic_json(root / "status.json", dict(state="failed", heartbeat=time.time(), error=repr(error)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--minutes", type=float, default=90)
    args = p.parse_args()
    if not 0 < args.minutes <= 120:
        p.error("Budget must be between 0 and 120 minutes")
    os.environ.setdefault("MUJOCO_GL", "disable")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
