"""Search short simulated leg motions that bootstrap balance recovery."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .recovery_bootstrap import demonstration_action
from .skill_env import SkillEnv


_environment: SkillEnv | None = None
_tilt = 20.0
_pose = "front"
_randomized_seeds = 0
_joint_ids = [0, 3, 4, 6, 9, 10]
_feedback = False
_tilt_width = 0.0
_rollout_seed_start = 70_000
_tilt_samples_per_seed = 1
POSE_JOINTS = {"front": [0, 3, 4, 6, 9, 10], "back": [0, 3, 4, 6, 9, 10],
               "left": [1, 3, 5, 7, 9, 11], "right": [1, 3, 5, 7, 9, 11]}


def _initialize(tilt: float, pose: str, randomized_seeds: int, joint_ids: list[int], feedback: bool,
                tilt_width: float, rollout_seed_start: int = 70_000, tilt_samples_per_seed: int = 1) -> None:
    global _environment, _tilt, _pose, _randomized_seeds, _joint_ids, _feedback, _tilt_width, _rollout_seed_start
    global _tilt_samples_per_seed
    # Larger-angle training permits crouching before standing. Match its
    # termination rules rather than cutting those search trajectories short.
    level = 10 if tilt + tilt_width > 25.0 else 6
    _environment = SkillEnv("recover", randomize=randomized_seeds > 0, recovery_level=level)
    _tilt = tilt
    _pose = pose
    _randomized_seeds = randomized_seeds
    _joint_ids = joint_ids
    _feedback = feedback
    _tilt_width = tilt_width
    _rollout_seed_start = rollout_seed_start
    _tilt_samples_per_seed = tilt_samples_per_seed


def _trial(candidate: np.ndarray) -> tuple[float, float]:
    assert _environment is not None
    env = _environment
    controls = np.zeros((3, 23))
    dimensions = 3 * len(_joint_ids)
    controls[:, _joint_ids] = candidate[:dimensions].reshape(3, len(_joint_ids))
    gains = candidate[dimensions:] if _feedback else None
    scores, successes = [], []
    cases = ((index, offset) for index in range(max(1, _randomized_seeds))
             for offset in (np.linspace(-_tilt_width, _tilt_width, _tilt_samples_per_seed)
                            if _tilt_samples_per_seed > 1 else
                            [_tilt_width * (2.0 * index / max(1, _randomized_seeds - 1) - 1.0)]))
    for index, offset in cases:
        observation, _ = env.reset(seed=_rollout_seed_start + 4 * index,
                                    options={"recovery_pose": _pose, "recovery_tilt_deg": _tilt + offset})
        score = 0.0
        for step in range(600):
            observation, reward, terminated, truncated, info = env.step(
                demonstration_action(controls, step, True, gains, observation, _pose))
            score += reward * 0.995 ** step
            if terminated or truncated:
                break
        scores.append(score)
        successes.append(info["success"])
    rate = float(np.mean(successes))
    return float(np.mean(scores)) + 100.0 * rate, rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tilt", type=float, default=20.0)
    parser.add_argument("--pose", choices=tuple(POSE_JOINTS), default="front")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rollout-seed-start", type=int, default=70_000,
                        help="Training physics seeds, separate from development and holdout seeds")
    parser.add_argument("--std", type=float, default=0.30)
    parser.add_argument("--feedback-std", type=float, default=1.0,
                        help="Feedback gain exploration; reduce when refining an existing controller")
    parser.add_argument("--randomized-seeds", type=int, default=0)
    parser.add_argument("--required-success-rate", type=float, default=0.75)
    parser.add_argument("--initial-controls", type=Path)
    parser.add_argument("--arms", action="store_true", help="Also search shoulder and elbow motions")
    parser.add_argument("--all-joints", action="store_true",
                        help="Search all 23 joints, including pitch and yaw in side recovery")
    parser.add_argument("--save-best", action="store_true",
                        help="Save the best candidate even below the search criterion, for diagnosis and refinement")
    parser.add_argument("--feedback", action="store_true", help="Search tilt and gyro corrections as well")
    parser.add_argument("--velocity-feedback", action="store_true",
                        help="Also search hip and ankle corrections from observed body velocity")
    parser.add_argument("--tilt-width", type=float, default=0.0)
    parser.add_argument("--tilt-samples-per-seed", type=int, default=1,
                        help="Test each physics seed at multiple tilts to cover latency and angle combinations")
    args = parser.parse_args()
    if args.velocity_feedback and not args.feedback:
        parser.error("--velocity-feedback requires --feedback")
    if (not 0.0 < args.tilt <= 30.0 or args.workers < 1 or args.population < 16 or args.iterations < 1
            or not 0.0 < args.std <= 1.0 or not 0.0 < args.feedback_std <= 4.0 or args.randomized_seeds < 0
            or not 0.0 < args.required_success_rate <= 1.0 or args.tilt_width < 0.0
            or args.tilt - args.tilt_width <= 0.0 or args.tilt + args.tilt_width > 30.0
            or args.rollout_seed_start < 0 or args.tilt_samples_per_seed < 1):
        parser.error("Invalid starter tilt or search budget")
    rng = np.random.default_rng(args.seed)
    joint_ids = (list(range(23)) if args.all_joints else
                 POSE_JOINTS[args.pose] + ([13, 14, 16, 18, 19, 21] if args.arms else []))
    dimensions = 3 * len(joint_ids)
    feedback_dimensions = 6 if args.velocity_feedback else 4
    size = dimensions + (feedback_dimensions if args.feedback else 0)
    mean, std = np.zeros(size), np.full(size, args.std)
    if args.feedback:
        std[dimensions:] = args.feedback_std
    if args.initial_controls:
        initial = np.load(args.initial_controls, allow_pickle=False)
        if initial.shape == (3, 6):
            expanded = np.zeros((3, 23))
            expanded[:, POSE_JOINTS[args.pose]] = initial
            initial = expanded
        mean[:dimensions] = initial[:, joint_ids].reshape(dimensions)
        metadata_path = args.initial_controls.with_suffix(".json")
        if args.feedback and metadata_path.exists():
            gains = json.loads(metadata_path.read_text(encoding="utf-8")).get("feedback")
            if gains is not None:
                if len(gains) not in {4, feedback_dimensions}:
                    parser.error("Initial feedback dimensions do not match the search mode")
                mean[dimensions:dimensions + len(gains)] = np.asarray(gains)
    elite_count = max(2, args.population // 8)
    best_candidate = None
    with mp.get_context("spawn").Pool(args.workers, initializer=_initialize,
                                      initargs=(args.tilt, args.pose, args.randomized_seeds, joint_ids, args.feedback,
                                                args.tilt_width, args.rollout_seed_start,
                                                args.tilt_samples_per_seed)) as pool:
        for iteration in range(args.iterations):
            candidates = rng.normal(mean, std, (args.population, size))
            candidates[:, :dimensions] = np.clip(candidates[:, :dimensions], -1.0, 1.0)
            if args.feedback:
                candidates[:, dimensions:] = np.clip(candidates[:, dimensions:], -4.0, 4.0)
            candidates[0] = mean
            if best_candidate is not None:
                candidates[1] = best_candidate
            results = pool.map(_trial, candidates)
            order = _candidate_order(results)
            best_candidate = candidates[order[-1]].copy()
            elites = candidates[order[-elite_count:]]
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            std = np.maximum(0.06, 0.25 * std + 0.75 * elites.std(axis=0))
            successes = [index for index in order if results[index][1] >= args.required_success_rate]
            print(f"[search] iteration={iteration}, successes={len(successes)}/{args.population}, "
                  f"best_rate={results[order[-1]][1]:.1%}, best_score={results[order[-1]][0]:.2f}", flush=True)
            if successes or args.save_best:
                index = successes[-1] if successes else order[-1]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                controls = np.zeros((3, 23))
                controls[:, joint_ids] = candidates[index, :dimensions].reshape(3, len(joint_ids))
                np.save(args.output, controls)
                args.output.with_suffix(".json").write_text(json.dumps({
                    "pose": args.pose, "tilt_deg": args.tilt, "seed": args.seed, "iteration": iteration,
                    "population": args.population, "search_score": results[index][0],
                    "search_pass_rate": results[index][1], "randomized_seeds": args.randomized_seeds,
                    "search_criterion_met": results[index][1] >= args.required_success_rate,
                    "recovery_version": 9, "rollout_seed_start": args.rollout_seed_start,
                    "tilt_width_deg": args.tilt_width,
                    "tilt_samples_per_seed": args.tilt_samples_per_seed,
                    "action_search_std": args.std, "feedback_search_std": args.feedback_std,
                    "joint_ids": joint_ids,
                    "feedback": candidates[index, dimensions:].tolist() if args.feedback else None,
                    "velocity_feedback": args.velocity_feedback,
                    "randomized_physics_evaluated": args.randomized_seeds > 0,
                }, indent=2) + "\n", encoding="utf-8")
                if successes:
                    return
    raise RuntimeError("No successful simulated motion found within the search budget")


def _candidate_order(results: list[tuple[float, float]]) -> np.ndarray:
    """Rank completion before reward; a fast partial success must not displace recovery."""
    return np.lexsort(([score for score, _ in results], [rate for _, rate in results]))


if __name__ == "__main__":
    main()
