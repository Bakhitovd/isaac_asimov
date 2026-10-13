"""Frozen-policy command correction and paired action-noise experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_robust_eval import load_pair, summarize
from .command_walk_v4 import CASES, MotionEnv
from .full_body_env import POLICY_DT
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path


@dataclass(frozen=True)
class Correction:
    name: str
    heading_gain: float = 0.0
    yaw_gain: float = 0.0
    speed_gain: float = 0.0
    yaw_limit: float = 0.15
    speed_limit: float = 0.04
    yaw_slew: float = 0.6
    speed_slew: float = 0.15
    filter_seconds: float = 0.1
    turn_speed_reduction: float = 0.0
    stop_heading_deadband: float = 0.0


BASELINE = Correction("baseline")
CONFIGURATIONS = (
    BASELINE,
    Correction("heading_025", heading_gain=0.25),
    Correction("heading_050", heading_gain=0.5),
    Correction("heading_100", heading_gain=1.0),
    Correction("heading_050_damped", heading_gain=0.5, yaw_gain=0.15),
    Correction("heading_050_speed", heading_gain=0.5, speed_gain=0.25),
)
TUNING_SEED, NOISE_SEED = 6_000_001, 7_000_001
_MODEL = _NORM = _DEADLINE = None


class CommandCorrection:
    """Modify only a copy of policy command inputs; the environment owns the goal."""

    def __init__(self, config):
        self.config = config
        self.delta = np.zeros(2)
        self.filtered = None

    def apply(self, observation):
        c = self.config
        # Raw observation channels: forward command 6, yaw command 78,
        # body forward velocity * .5 at 79, body gyro z * .25 at 2.
        nominal = observation[[6, 78]].astype(float)
        measured = observation[[79, 2]].astype(float) * [2.0, 4.0]
        if self.filtered is None:
            self.filtered = measured.copy()
        self.filtered += POLICY_DT / (c.filter_seconds + POLICY_DT) * (measured - self.filtered)
        heading_error = float(np.arctan2(observation[83], observation[84]))
        if abs(nominal[0]) < 1e-6 and abs(nominal[1]) < 1e-6:
            heading_error = np.sign(heading_error) * max(0.0, abs(heading_error) - c.stop_heading_deadband)
        speed_reduction = c.turn_speed_reduction if abs(nominal[1]) > 0.1 and nominal[0] > 0.05 else 0.0
        desired = np.array(
            [
                c.speed_gain * (nominal[0] - self.filtered[0]) - speed_reduction,
                c.heading_gain * heading_error + c.yaw_gain * (nominal[1] - self.filtered[1]),
            ]
        )
        limits = np.array([c.speed_limit, c.yaw_limit])
        rates = np.array([c.speed_slew, c.yaw_slew]) * POLICY_DT
        bounded = np.clip(desired, -limits, limits)
        self.delta += np.clip(bounded - self.delta, -rates, rates)
        command = np.clip(nominal + self.delta, [0.0, -0.6], [0.3, 0.6])
        modified = observation.copy()
        modified[[6, 78]] = command
        return modified, command - nominal, bool(np.any(np.abs(desired) > limits))


def fingerprint(model, norm):
    h = hashlib.sha256()
    for key, value in sorted(model.policy.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    for stats in (norm.obs_rms, norm.ret_rms):
        for name in ("mean", "var", "count"):
            h.update(np.asarray(getattr(stats, name)).tobytes())
    return h.hexdigest()


def initialize_worker(checkpoint, deadline):
    global _MODEL, _NORM, _DEADLINE
    torch.set_num_threads(1)
    _MODEL, _NORM = load_pair(Path(checkpoint))
    _MODEL.policy.set_training_mode(False)
    _DEADLINE = deadline


def evaluate_batch(job):
    label, config, noise_scale, case, seeds = job[:5]
    options = job[5] if len(job) > 5 else {}
    model, norm = _MODEL, _NORM
    before = fingerprint(model, norm)
    envs = [MotionEnv(case=case) for _ in seeds]
    corrections = [CommandCorrection(config) for _ in seeds]
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, 771])) for seed in seeds]
    totals = [dict(steps=0, reward=0.0, correction_squared=np.zeros(2), limited=0, noise_squared=0.0) for _ in seeds]
    records = []
    try:
        observations = [env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)]
        if "schedule" in options:
            for i, env in enumerate(envs):
                env.schedule = options["schedule"]
                env.duration = sum(s[0] for s in env.schedule)
                env.command[:] = env.requested[:] = 0
                env._prepare_command()
                observations[i] = env._observe(False)
        noise_states = [None for _ in seeds]
        active = list(range(len(envs)))
        while active:
            if time.time() >= _DEADLINE:
                raise TimeoutError("Experiment wall-time budget reached")
            corrected, changes, limits = [], [], []
            for i in active:
                obs, delta, limited = corrections[i].apply(observations[i])
                if config == BASELINE and not np.array_equal(obs, observations[i]):
                    raise RuntimeError("Zero correction changed the original observation")
                corrected.append(obs[: model.observation_space.shape[0]])
                changes.append(delta)
                limits.append(limited)
            normalized = norm.normalize_obs(np.array(corrected))
            with torch.no_grad():
                distribution = model.policy.get_distribution(
                    torch.as_tensor(normalized, device=model.device)
                ).distribution
                means = distribution.mean.cpu().numpy()
                std = distribution.stddev.cpu().numpy()
            innovations = [rngs[i].standard_normal(model.action_space.shape) for i in active]
            rho = float(options.get("noise_rho", 0.0))
            for i, innovation in zip(active, innovations):
                noise_states[i] = (
                    innovation if noise_states[i] is None else rho * noise_states[i] + np.sqrt(1 - rho**2) * innovation
                )
            noise = np.array([noise_states[i] for i in active]) * std * noise_scale
            actions = np.clip(means + noise, model.action_space.low, model.action_space.high)
            remaining = []
            for position, (i, action) in enumerate(zip(active, actions)):
                env, total = envs[i], totals[i]
                original_command = env.command.copy()
                expected_heading = env.reference_heading + original_command[1] * POLICY_DT
                observations[i], reward, done, truncated, info = env.step(action)
                if abs(env.reference_heading - expected_heading) > 1e-10 or not np.array_equal(
                    original_command, [info["command_v"], info["command_w"]]
                ):
                    raise RuntimeError("Correction changed the requested trajectory or scoring command")
                total["steps"] += 1
                total["reward"] += reward
                total["correction_squared"] += changes[position] ** 2
                total["limited"] += limits[position]
                total["noise_squared"] += float(np.mean(noise[position] ** 2))
                if done or truncated:
                    records.append(
                        {k: info[k] for k in ("success", "failures", "metrics", "violation", "seconds")}
                        | dict(
                            seed=seeds[i],
                            schedule=env.schedule,
                            mean_reward=total["reward"] / total["steps"],
                            reward_components={k: v / total["steps"] for k, v in info["reward_totals"].items()},
                            correction_rms=np.sqrt(total["correction_squared"] / total["steps"]).tolist(),
                            correction_limit_fraction=total["limited"] / total["steps"],
                            action_noise_rms=float(np.sqrt(total["noise_squared"] / total["steps"])),
                        )
                    )
                else:
                    remaining.append(i)
            active = remaining
        if fingerprint(model, norm) != before:
            raise RuntimeError("Frozen policy or normalization changed during evaluation")
        return label, case, records
    finally:
        for env in envs:
            env.close()


def paired(before, after):
    result = {}
    for case, family in before["families"].items():
        left = {e["seed"]: e for e in family["episodes"]}
        right = {e["seed"]: e for e in after["families"][case]["episodes"]}
        if left.keys() != right.keys() or any(left[s]["schedule"] != right[s]["schedule"] for s in left):
            raise RuntimeError("Paired comparison has different cases or requested schedules")
        result[case] = dict(
            baseline=family["pass_rate"],
            corrected=after["families"][case]["pass_rate"],
            improved=sum(not left[s]["success"] and right[s]["success"] for s in left),
            worsened=sum(left[s]["success"] and not right[s]["success"] for s in left),
            count=len(left),
        )
    return result


def run_stage(pool, root, stage, arms, cases, count, seed_start, started):
    pending = {}
    collected = {label: {case: [] for case in cases} for label, _, _ in arms}
    for label, correction, noise in arms:
        for case in cases:
            for offset in range(0, count, 10):
                seeds = [seed_start + CASES.index(case) * 100_000 + i for i in range(offset, min(count, offset + 10))]
                job = label, correction, noise, case, seeds
                pending[pool.submit(evaluate_batch, job)] = (label, case, offset)
    total = len(pending)
    last_print = 0
    for number, future in enumerate(as_completed(pending), 1):
        label, case, rows = future.result()
        collected[label][case].extend(rows)
        atomic_json(
            root / "batches" / stage / f"{label}_{case}_{min(e['seed'] for e in rows)}.json", {"episodes": rows}
        )
        atomic_json(
            root / "status.json",
            dict(
                state="running",
                stage=stage,
                heartbeat=time.time(),
                completed_batches=number,
                total_batches=total,
                elapsed_seconds=time.time() - started,
            ),
        )
        if time.time() - last_print >= 10 or number == total:
            print(f"[{stage}] {number}/{total} batches; elapsed={(time.time()-started)/60:.1f} min", flush=True)
            last_print = time.time()
    reports = {}
    for label, correction, noise in arms:
        for rows in collected[label].values():
            rows.sort(key=lambda row: row["seed"])
        reports[label] = dict(
            **summarize(collected[label]),
            correction=asdict(correction),
            noise_scale=noise,
            seed_start=seed_start,
            count_per_family=count,
            physics="nominal",
        )
        atomic_json(root / stage / f"{label}.json", reports[label])
        print(
            f"[{stage}/{label}] "
            + ", ".join(f"{case}={f['success_count']}/{f['count']}" for case, f in reports[label]["families"].items()),
            flush=True,
        )
    return reports


def experiment(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.minutes * 60
    source = args.checkpoint.resolve()
    baseline_dir = root / "baseline"
    baseline_dir.mkdir()
    checkpoint = baseline_dir / "policy.zip"
    shutil.copy2(source, checkpoint)
    shutil.copy2(_stats_path(source), _stats_path(checkpoint))
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, snapshot / p.name)
    manifest = dict(
        source=str(source),
        policy_sha256=sha256(checkpoint),
        normalization_sha256=sha256(_stats_path(checkpoint)),
        code_sha256={p.name: sha256(p) for p in snapshot.glob("*.py")},
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started=started,
        deadline=deadline,
        workers=args.workers,
        configurations=[asdict(c) for c in CONFIGURATIONS],
        noise_scales=[0.0, 0.25, 0.5, 1.0],
        panels=dict(
            tuning=[TUNING_SEED, 20],
            noise=[NOISE_SEED, 20],
            development=[1001, 50],
            confirmation=[10051, 50],
            sequences=[1001, 20],
        ),
        selection="Nonzero candidate: retention within 10pp on tuning, then minimum pass rate, mean pass rate, lower violation",
        benefit_gate="Every development family retains within 4pp, minimum improves >=4pp, mean does not decline",
        scope="Frozen-policy diagnostic; no training, publication, or hardware qualification",
    )
    atomic_json(root / "manifest.json", manifest)
    (root / "working_tree.diff").write_bytes(subprocess.check_output(["git", "diff"]))
    atomic_json(root / "status.json", dict(state="running", stage="starting", heartbeat=time.time()))
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(checkpoint), deadline),
        ) as pool:
            tuning = run_stage(
                pool, root, "tuning", [(c.name, c, 0.0) for c in CONFIGURATIONS], FAMILIES, 20, TUNING_SEED, started
            )

            def rank(c):
                r = tuning[c.name]
                retained = all(
                    r["families"][f]["pass_rate"] + 0.100001 >= tuning["baseline"]["families"][f]["pass_rate"]
                    for f in FAMILIES
                )
                return retained, r["minimum_pass_rate"], r["mean_pass_rate"], -r["mean_violation"]

            selected = max(CONFIGURATIONS[1:], key=rank)
            atomic_json(
                root / "selection.json",
                dict(configuration=asdict(selected), rank=rank(selected), selected_before_development=True),
            )
            print(f"[selected] {selected.name}; locked before development", flush=True)
            noise = run_stage(
                pool,
                root,
                "noise",
                [(f"scale_{s:g}", BASELINE, s) for s in (0.0, 0.25, 0.5, 1.0)],
                FAMILIES,
                20,
                NOISE_SEED,
                started,
            )
            arms = [("baseline", BASELINE, 0.0), ("corrected", selected, 0.0)]
            development = run_stage(pool, root, "development", arms, FAMILIES, 50, 1001, started)
            baseline = development["baseline"]
            if args.expected_baseline:
                expected = json.loads(args.expected_baseline.read_text())["initial"]["families"]
                counts = lambda families: {k: (v["success_count"], v["count"]) for k, v in families.items()}
                if counts(baseline["families"]) != counts(expected):
                    raise RuntimeError("Baseline does not reproduce archived development counts")
            confirmation = run_stage(pool, root, "confirmation", arms, FAMILIES, 50, 10051, started)
            sequences = run_stage(pool, root, "sequences", arms, ("sequence",), 20, 1001, started)
        candidate = development["corrected"]
        comparisons = {
            name: paired(r["baseline"], r["corrected"])
            for name, r in (("development", development), ("confirmation", confirmation), ("sequences", sequences))
        }
        benefit = (
            all(r["corrected"] + 0.040001 >= r["baseline"] for r in comparisons["development"].values())
            and candidate["minimum_pass_rate"] >= baseline["minimum_pass_rate"] + 0.039999
            and candidate["mean_pass_rate"] >= baseline["mean_pass_rate"] - 1e-9
        )
        result = dict(
            selected=asdict(selected),
            development_benefit_gate=benefit,
            comparisons=comparisons,
            noise={
                label: {
                    f: dict(
                        pass_rate=r["pass_rate"],
                        failures=r["failure_counts"],
                        metrics=r["metrics"],
                        mean_violation=r["mean_violation"],
                    )
                    for f, r in report["families"].items()
                }
                for label, report in noise.items()
            },
            elapsed_seconds=time.time() - started,
            policy_unchanged=sha256(source) == manifest["policy_sha256"],
            normalization_unchanged=sha256(_stats_path(source)) == manifest["normalization_sha256"],
            qualified=False,
            decision="Review paired evidence before any learned correction or motor-policy training",
        )
        atomic_json(root / "result.json", result)
        atomic_json(
            root / "status.json",
            dict(state="completed", heartbeat=time.time(), elapsed_seconds=time.time() - started, benefit_gate=benefit),
        )
        print(
            f"[completed] development benefit={benefit}; no weights updated; {(time.time()-started)/60:.1f} min",
            flush=True,
        )
    except BaseException as error:
        atomic_json(root / "status.json", dict(state="failed", heartbeat=time.time(), error=repr(error)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--expected-baseline", type=Path)
    args = p.parse_args()
    if not 1 <= args.workers <= 16 or not 0 < args.minutes <= 90:
        p.error("Use 1-16 workers and a positive budget of at most 90 minutes")
    os.environ.setdefault("MUJOCO_GL", "disable")
    torch.set_num_threads(1)
    experiment(args)


if __name__ == "__main__":
    main()
