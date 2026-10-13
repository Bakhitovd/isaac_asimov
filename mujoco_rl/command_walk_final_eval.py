"""Parallel physical evaluations and complete moderate-walking inference bundles."""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_correction import CommandCorrection, fingerprint
from .command_walk_final import CASES, CORRECTION, QUALITY, FinalEnv, contract, validate_command
from .command_walk_robust_eval import load_pair, summarize
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path

_MODEL = _NORM = _DEADLINE = None


def initialize(checkpoint, deadline):
    global _MODEL, _NORM, _DEADLINE
    torch.set_num_threads(1)
    _MODEL, _NORM = load_pair(Path(checkpoint))
    _MODEL.policy.set_training_mode(False)
    _DEADLINE = deadline


def batch(job):
    label, case, seeds, seconds, fixed = job
    model, norm = _MODEL, _NORM
    before = fingerprint(model, norm)
    envs = [FinalEnv(case=case, seconds=seconds, fixed=fixed) for _ in seeds]
    records = []
    try:
        obs = [e.reset(seed=s)[0] for e, s in zip(envs, seeds)]
        active = list(range(len(envs)))
        while active:
            if time.time() >= _DEADLINE:
                raise TimeoutError("Evaluation budget reached")
            actions, _ = model.predict(
                norm.normalize_obs(np.array([obs[i][: model.observation_space.shape[0]] for i in active])),
                deterministic=True,
            )
            remaining = []
            for i, action in zip(active, actions):
                obs[i], _, done, truncated, info = envs[i].step(action)
                if done or truncated:
                    records.append(
                        {k: info[k] for k in ("success", "failures", "metrics", "violation", "seconds", "quality")}
                        | dict(seed=seeds[i], schedule=envs[i].schedule)
                    )
                else:
                    remaining.append(i)
            active = remaining
        if before != fingerprint(model, norm):
            raise RuntimeError("Evaluation mutated model or statistics")
        return label, records
    finally:
        for e in envs:
            e.close()


def panel(checkpoint, output, cases=FAMILIES, count=50, seed_start=1001, seconds=60, deadline=None, grid=False):
    output = Path(output)
    if output.exists():
        cached = json.loads(output.read_text())
        if (
            cached["checkpoint_sha256"] != sha256(Path(checkpoint))
            or cached["seed_start"] != seed_start
            or cached["seconds"] != seconds
            or any(f["count"] != count for f in cached["families"].values())
        ):
            raise ValueError(f"Cached evaluation configuration mismatch: {output}")
        return cached
    deadline = deadline or time.time() + 3600
    jobs = []
    if grid:
        for vi, v in enumerate((0.1, 0.2, 0.3)):
            for wi, w in enumerate((0.15, 0.3, 0.5)):
                for case in ("left_arc", "right_arc"):
                    label = f"{case}_v{v:g}_w{w:g}"
                    start = seed_start + (vi * 3 + wi) * 1000 + CASES.index(case) * 100_000
                    for offset in range(0, count, 10):
                        jobs.append(
                            (label, case, list(range(start + offset, start + min(count, offset + 10))), seconds, (v, w))
                        )
    else:
        for case in cases:
            start = seed_start + CASES.index(case) * 100_000
            for offset in range(0, count, 10):
                jobs.append((case, case, list(range(start + offset, start + min(count, offset + 10))), seconds, None))
    collected = {job[0]: [] for job in jobs}
    with ProcessPoolExecutor(
        max_workers=8, mp_context=mp.get_context("spawn"), initializer=initialize, initargs=(str(checkpoint), deadline)
    ) as pool:
        futures = [pool.submit(batch, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            label, rows = future.result()
            collected[label].extend(rows)
            atomic_json(
                output.with_suffix(".status.json"),
                dict(state="running", heartbeat=time.time(), batches=index, total=len(jobs)),
            )
            if index % 10 == 0 or index == len(jobs):
                print(f"[eval {output.parent.name}/{output.stem}] {index}/{len(jobs)}", flush=True)
    for rows in collected.values():
        rows.sort(key=lambda r: r["seed"])
    report = dict(
        **summarize(collected),
        seed_start=seed_start,
        seconds=seconds,
        contract=contract(),
        checkpoint_sha256=sha256(Path(checkpoint)),
    )
    atomic_json(output, report)
    atomic_json(output.with_suffix(".status.json"), dict(state="completed", heartbeat=time.time()))
    print(
        f"[result {output.parent.name}/{output.stem}] "
        + str({k: v["success_count"] for k, v in report["families"].items()}),
        flush=True,
    )
    return report


def retain(before, after, cases=FAMILIES):
    return all(after["families"][k]["pass_rate"] + 0.040001 >= before["families"][k]["pass_rate"] for k in cases)


def passes(report, threshold=0.96):
    return all(v["pass_rate"] + 1e-9 >= threshold for v in report["families"].values())


def rank(report):
    return (report["minimum_pass_rate"], report["mean_pass_rate"], -report["mean_violation"])


def quality_gate(before, after):
    results = {}
    for case, b in before["families"].items():
        a = after["families"][case]
        left = {e["seed"]: e for e in b["episodes"] if e["success"]}
        right = {e["seed"]: e for e in a["episodes"] if e["success"]}
        seeds = sorted(left.keys() & right.keys())
        if not seeds:
            results[case] = dict(passed=False, reason="no_paired_successes")
            continue
        metrics = {
            k: dict(
                before=float(np.mean([left[s]["quality"][k] for s in seeds])),
                after=float(np.mean([right[s]["quality"][k] for s in seeds])),
            )
            for k in QUALITY
        }
        results[case] = dict(
            passed=all(v["after"] <= v["before"] * 1.1 + 1e-12 for v in metrics.values()),
            paired_count=len(seeds),
            metrics=metrics,
        )
    return dict(passed=all(v["passed"] for v in results.values()), families=results)


def bundle(checkpoint, output, qualified, reports):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "policy.zip"
    if Path(checkpoint).resolve() != target.resolve():
        shutil.copy2(checkpoint, target)
        shutil.copy2(_stats_path(Path(checkpoint)), _stats_path(target))
    atomic_json(
        output / "controller.json",
        dict(
            **contract(),
            qualified=qualified,
            policy_sha256=sha256(target),
            normalization_sha256=sha256(_stats_path(target)),
            reports=reports,
        ),
    )
    return target


class Controller:
    """Raw sensor-observation API; callers must NOT apply command correction themselves.

    Requested commands are checked separately from ramped observation channels.
    Call reset at episode start, then act exactly once per 20 ms policy tick.
    """

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "controller.json").read_text())
        if any(self.metadata[k] != v for k, v in contract().items()):
            raise ValueError("Unsupported controller contract")
        checkpoint = directory / "policy.zip"
        if (
            sha256(checkpoint) != self.metadata["policy_sha256"]
            or sha256(_stats_path(checkpoint)) != self.metadata["normalization_sha256"]
        ):
            raise ValueError("Controller bundle integrity mismatch")
        self.model, self.norm = load_pair(checkpoint)
        self.reset()

    def reset(self):
        self.corrector = CommandCorrection(CORRECTION)

    def act(self, raw_observation, requested_v, requested_w):
        validate_command(requested_v, requested_w)
        raw_observation = np.asarray(raw_observation, dtype=np.float32)
        if raw_observation.shape != (87,) or not np.isfinite(raw_observation).all():
            raise ValueError("Expected 87 finite raw observation values")
        observation = self.corrector.apply(raw_observation)[0]
        action, _ = self.model.predict(
            self.norm.normalize_obs(observation[None, : self.model.observation_space.shape[0]]), deterministic=True
        )
        return action[0]

    def close(self):
        self.norm.close()
