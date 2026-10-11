"""Balanced specialist distillation with bounded student-state data aggregation."""

import argparse
import json
import pickle
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .command_walk import FAMILIES
from .command_walk_eval import compact, rank
from .command_walk_robust_eval import load_pair
from .command_walk_train import save_bundle
from .command_walk_v4 import MotionEnv, Spec, panel, passes, teacher_for
from .command_walk_v4_train import build
from .run_training import actor_parameters
from .skill_campaign import atomic_json, sha256


class Teachers:
    def __init__(self, paths):
        self.pairs = {}
        try:
            for key, path in paths.items():
                self.pairs[key] = load_pair(Path(path))
        except Exception:
            self.close()
            raise

    def mean(self, key, raw):
        model, norm = self.pairs[key]
        obs = norm.normalize_obs(raw[None, : model.observation_space.shape[0]])
        with torch.no_grad():
            return model.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution.mean.numpy()[0]

    def close(self):
        for _, norm in self.pairs.values():
            norm.close()


def collect(teachers, student, norm, round_index, progress, per_family=8):
    groups, records = {}, []
    for fi, family in enumerate(FAMILIES):
        xs, ys, accepted = [], [], 0
        for attempt in range(100 if round_index == 0 else per_family):
            progress()
            seed = 4_000_001 + round_index * 1_000_000 + fi * 100_000 + attempt
            env = MotionEnv(case=family)
            observations, labels = [], []
            interventions = 0
            try:
                raw, _ = env.reset(seed=seed)
                while True:
                    if env.step_count % 100 == 0:
                        progress()
                    key = teacher_for(env.requested)
                    target = teachers.mean(key, raw)
                    if round_index == 0:
                        action = target
                    else:
                        action, _ = student.predict(norm.normalize_obs(raw[None]), deterministic=True)
                        action = action[0]
                    # Only intervene for impending falls; all interventions are counted.
                    gravity = env.data.site_xmat[env.imu_site_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
                    if round_index and gravity[2] > -0.85:
                        action = target
                        interventions += 1
                    observations.append(raw.copy())
                    labels.append(target.copy())
                    raw, _, done, truncated, info = env.step(np.clip(action, -1, 1))
                    if done or truncated:
                        break
                records.append(
                    dict(
                        family=family,
                        seed=seed,
                        success=info["success"],
                        failures=info["failures"],
                        interventions=interventions,
                        frames=len(observations),
                    )
                )
                # Failed teacher rollouts are excluded. Student-state labels are retained
                # with failure/intervention evidence, not advertised as successful demonstrations.
                if round_index or info["success"]:
                    indices = np.linspace(0, len(observations) - 1, 128, dtype=int)
                    xs.extend(np.asarray(observations)[indices])
                    ys.extend(np.asarray(labels)[indices])
                    accepted += 1
            finally:
                env.close()
            if accepted == per_family:
                break
        if accepted != per_family:
            raise RuntimeError(f"Insufficient successful teacher episodes for {family}")
        groups[family] = (np.asarray(xs), np.asarray(ys))
    return groups, records


def distill(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = json.loads(args.teachers.read_text())
    if set(paths) != {"baseline", "arcs", "pivots", "braking"}:
        raise ValueError("All four teacher roles are required")
    teachers = Teachers(paths)
    model = norm = None
    optimizer = None
    dataset = {k: ([], []) for k in FAMILIES}
    best, next_round, next_epoch, collected_round = None, 0, 0, -1
    rng = np.random.default_rng(44)
    cancelled = False
    last_heartbeat = 0.0

    def stop(*_):
        nonlocal cancelled
        cancelled = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def progress():
        nonlocal last_heartbeat
        if cancelled or time.time() >= args.deadline:
            raise TimeoutError("interrupted" if cancelled else "time_budget")
        if time.time() - last_heartbeat > 10:
            last_heartbeat = time.time()
            atomic_json(args.output / "status.json", dict(state="running", phase="distilling", heartbeat=time.time()))

    try:
        source = Path(paths["baseline"])
        if args.resume_from:
            source = Path(json.loads((args.resume_from / "latest.json").read_text())["checkpoint"])
        model, norm, _ = build(source, Spec(args.initialization), 44, resume=bool(args.resume_from))
        optimizer = torch.optim.Adam(actor_parameters(model)[:-1], lr=1e-4)
        initial_std = model.policy.log_std.detach().clone()
        if args.resume_from:
            with (args.resume_from / "distill_resume.pkl").open("rb") as f:
                saved = pickle.load(f)
            if saved["teachers"] != paths:
                raise ValueError("Distillation teachers changed on resume")
            optimizer.load_state_dict(saved["optimizer"])
            dataset, best = saved["dataset"], saved["best"]
            next_round, next_epoch, collected_round = saved["next_round"], saved["next_epoch"], saved["collected_round"]
            rng.bit_generator.state = saved["rng"]
        atomic_json(
            args.output / "provenance.json",
            dict(teachers=paths, teacher_hashes={k: sha256(Path(v)) for k, v in paths.items()}, deadline=args.deadline),
        )
        progress()
        for round_index in range(next_round, 4):
            next_round = round_index
            if collected_round < round_index:
                groups, records = collect(teachers, model, norm, round_index, progress, args.episodes)
                atomic_json(args.output / f"round_{round_index}_episodes.json", records)
                for k, (x, y) in groups.items():
                    dataset[k][0].extend(x)
                    dataset[k][1].extend(y)
                collected_round, next_epoch = round_index, 0
            raw = np.concatenate([np.asarray(dataset[k][0]) for k in FAMILIES])
            target = np.concatenate([np.asarray(dataset[k][1]) for k in FAMILIES])
            np.savez_compressed(args.output / f"round_{round_index}_data.npz", observations=raw, means=target)
            x = torch.as_tensor(norm.normalize_obs(raw), dtype=torch.float32)
            y = torch.as_tensor(target, dtype=torch.float32)
            # Every family has equal samples, including after each aggregation round.
            for epoch in range(next_epoch, 20):
                next_epoch = epoch
                for start in range(0, len(x), 256):
                    progress()
                    indices = rng.integers(len(x), size=256)
                    prediction = model.policy.get_distribution(x[indices]).distribution.mean
                    loss = torch.nn.functional.mse_loss(prediction, y[indices])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite distillation loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters(model)[:-1], 1.0, error_if_nonfinite=True)
                    optimizer.step()
                with (args.output / "loss.jsonl").open("a") as f:
                    f.write(json.dumps(dict(round=round_index, epoch=epoch, loss=float(loss.detach()))) + "\n")
                next_epoch = epoch + 1
            assert torch.equal(model.policy.log_std, initial_std)
            checkpoint = save_bundle(model, norm, args.output, f"distill_{round_index}")
            torch.save(optimizer.state_dict(), checkpoint.parent / "distill_optimizer.pt")
            manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
            manifest["distill_optimizer.pt"] = sha256(checkpoint.parent / "distill_optimizer.pt")
            atomic_json(checkpoint.parent / "manifest.json", manifest)
            report = panel(model, norm, count=args.cases, progress=lambda *_: progress())
            atomic_json(args.output / f"round_{round_index}_evaluation.json", report)
            if best is None or rank(report) > tuple(best["rank"]):
                best = dict(checkpoint=str(checkpoint.resolve()), rank=rank(report), report=compact(report))
                atomic_json(args.output / "best.json", best)
            if passes(report, FAMILIES, 0.90, 50):
                break
            next_round, next_epoch = round_index + 1, 0
        atomic_json(args.output / "report.json", dict(best=best, passed=passes(best["report"], FAMILIES, 0.90, 50)))
        atomic_json(args.output / "status.json", dict(state="completed", heartbeat=time.time()))
    except TimeoutError as error:
        atomic_json(args.output / "status.json", dict(state="stopped", reason=str(error), heartbeat=time.time()))
    except Exception as error:
        atomic_json(args.output / "status.json", dict(state="failed", reason=repr(error), heartbeat=time.time()))
        raise
    finally:
        if model is not None:
            save_bundle(model, norm, args.output, "resume")
            if optimizer is not None:
                with (args.output / "distill_resume.pkl").open("wb") as f:
                    pickle.dump(
                        dict(
                            optimizer=optimizer.state_dict(),
                            dataset=dataset,
                            best=best,
                            next_round=next_round,
                            next_epoch=next_epoch,
                            collected_round=collected_round,
                            rng=rng.bit_generator.state,
                            teachers=paths,
                        ),
                        f,
                    )
        if norm:
            norm.close()
        teachers.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("teachers", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--initialization", default="preserved")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--resume-from", type=Path)
    args = p.parse_args()
    if args.episodes < 1 or args.cases < 1 or not np.isfinite(args.deadline):
        p.error("Invalid distillation budget")
    distill(args)
