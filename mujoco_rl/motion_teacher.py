"""Versioned whole-body motion teachers, searched and verified in the same simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .full_body_env import POLICY_DT, STANDING_POSE
from .skill_env import MAX_STEPS, SkillEnv
from .skill_scenarios import reset_case


@dataclass
class MotionTeacher:
    skill: str
    pose: str
    targets: np.ndarray
    durations: np.ndarray
    feedback: np.ndarray
    conditions: list[dict]
    scenario: str = "standard"

    @classmethod
    def load(cls, path: Path) -> "MotionTeacher":
        value = json.loads(path.read_text())
        if value.get("version") != 2:
            raise ValueError("Expected whole-body teacher version 2; legacy controls use recovery_bootstrap")
        teacher = cls(value["skill"], value["pose"], np.asarray(value["targets"], dtype=float),
                      np.asarray(value["durations"], dtype=int), np.asarray(value["feedback"], dtype=float),
                      value["conditions"], value.get("scenario", "standard"))
        teacher.validate()
        return teacher

    def validate(self) -> None:
        n = len(self.durations)
        if (self.skill not in {"recover", "jump"} or self.targets.shape != (n, 23) or n < 2
                or not np.isfinite(self.targets).all() or self.feedback.shape != (4,)
                or not np.isfinite(self.feedback).all() or np.any(np.abs(self.feedback) > 4)
                or np.any(self.durations < 1) or sum(self.durations) > MAX_STEPS[self.skill]
                or len(self.conditions) != n):
            raise ValueError("Invalid motion teacher dimensions, feedback, or duration")
        valid_pose = {"front", "back", "left", "right"} if self.skill == "recover" else {"up", "forward"}
        if self.pose not in valid_pose or self.scenario not in {"standard", "settled_floor", "dynamic_fall"}:
            raise ValueError("Invalid teacher pose/scenario")
        for condition in self.conditions:
            if set(condition) - {"min_height", "min_upright", "both_feet"}:
                raise ValueError("Unknown phase condition")
            if any(not np.isfinite(value) for value in condition.values()):
                raise ValueError("Nonfinite phase condition")

    def document(self) -> dict:
        return {"version": 2, "skill": self.skill, "pose": self.pose, "targets": self.targets.tolist(),
                "durations": self.durations.tolist(), "feedback": self.feedback.tolist(),
                "conditions": self.conditions, "scenario": self.scenario,
                "units": {"targets": "radians", "durations": "20ms control steps"}}


class TeacherController:
    def __init__(self, teacher: MotionTeacher, env: SkillEnv):
        teacher.validate()
        self.teacher, self.env = teacher, env
        self.phase, self.age = 0, 0
        self.origin = env.data.qpos[env.qpos_ids].copy()
        self.action_grid = np.linspace(-1, 1, 2049)
        self.target_grid = np.stack([env._target_from_action(np.full(23, value)) for value in self.action_grid])

    def action(self, observation: np.ndarray) -> np.ndarray:
        teacher, env = self.teacher, self.env
        duration = teacher.durations[self.phase]
        condition = teacher.conditions[self.phase]
        ready = bool(condition) and self.age >= int(duration * 0.6)
        ready &= env.data.qpos[2] >= condition.get("min_height", -np.inf)
        ready &= -float(observation[5]) >= condition.get("min_upright", -np.inf)
        ready &= not condition.get("both_feet", False) or env._contacts()[0].all()
        if self.phase < len(teacher.durations) - 1 and (self.age >= duration or ready):
            self.origin = teacher.targets[self.phase].copy()
            self.phase += 1
            self.age = 0
            duration = teacher.durations[self.phase]
        fraction = min(1.0, (self.age + 1) / max(1, int(duration * 0.65)))
        smooth = fraction * fraction * (3 - 2 * fraction)
        target = self.origin + smooth * (teacher.targets[self.phase] - self.origin)
        action = np.array([np.interp(target[j], self.target_grid[:, j], self.action_grid) for j in range(23)],
                          dtype=np.float32)
        # Feedback uses actor-visible delayed IMU and velocity observations.
        pitch = float(np.arctan2(observation[3], -observation[5]))
        roll = float(np.arctan2(-observation[4], -observation[5]))
        action[[0, 6]] += np.array([1, -1]) * (teacher.feedback[0] * pitch + teacher.feedback[1] * observation[1] * 4)
        action[[1, 7]] += teacher.feedback[2] * roll + teacher.feedback[3] * observation[0] * 4
        self.age += 1
        return np.clip(action, -1, 1)


def template(skill: str, pose: str) -> MotionTeacher:
    targets = np.tile(STANDING_POSE, (6, 1))
    # Support -> fold -> move over support -> extend -> settle -> hold.
    # These are search initializations, never assumed to be successful demonstrations.
    fold = np.array([0.30, 1.0, 0.75, 0.30, 0.08, 0.0])
    if skill == "jump":
        fold = np.array([0.8, 0.9, -0.2, 0.45, 0.1, 0.0])
    for index, weight in enumerate(fold):
        targets[index, [0, 6]] += np.array([-0.75, 0.75]) * weight
        targets[index, [3, 9]] += np.array([0.9, -0.9]) * weight
        targets[index, [4, 10]] += np.array([-0.04, 0.04]) * weight
        targets[index, [13, 18]] += np.array([0.8, -0.8]) * weight
        targets[index, [16, 21]] += np.array([-0.5, 0.5]) * weight
    if pose == "back":
        targets[:3, [13, 18]] *= -1
    if pose in {"left", "right"}:
        sign = 1 if pose == "left" else -1
        targets[:3, [1, 7]] += sign * np.array([[0.4], [0.3], [0.1]])
        targets[:3, 12] += sign * np.array([0.4, 0.3, 0.1])
        targets[:3, [14, 19]] += sign * 0.35
    if pose == "forward":
        targets[2, [0, 6]] += [-0.15, 0.15]
    duration = [45, 60, 60, 80, 75, 180] if skill == "recover" else [35, 20, 10, 35, 40, 180]
    conditions = [{}, {}, {}, {"min_height": 0.55, "min_upright": 0.96, "both_feet": True}, {}, {}]
    return MotionTeacher(skill, pose, targets, np.array(duration), np.zeros(4), conditions,
                         "settled_floor" if skill == "recover" else "standard")


def rollout(teacher: MotionTeacher, seed: int, record: bool = False,
            action_noise: float = 0.0) -> tuple[dict, dict]:
    env = SkillEnv(teacher.skill, randomize=True, recovery_sensor_memory=teacher.skill == "recover",
                   recovery_phase_features=teacher.skill == "recover")
    arrays: dict[str, list] = {key: [] for key in ("observations", "actions", "qpos", "qvel", "target",
                                                  "action_buffer", "sensor_buffer", "last_action", "previous_action",
                                                  "step", "phase", "initial_gravity", "torque")}
    try:
        rng = np.random.default_rng(seed)
        observation, initial = reset_case(env, seed, teacher.scenario,
                                         teacher.pose if teacher.skill == "recover" else None)
        if teacher.skill == "jump":
            env.jump_forward = teacher.pose == "forward"
            env.goal_body[0] = 0.2 if env.jump_forward else 0.0
            observation[81:83] = env.goal_body * 0.5
        controller = TeacherController(teacher, env)
        score = 0.0
        for step in range(MAX_STEPS[teacher.skill]):
            action = controller.action(observation)
            if record:
                values = (observation.copy(), action.copy(), env.data.qpos.copy(), env.data.qvel.copy(),
                          env.target.copy(), np.array(env.action_buffer), np.array(env.sensor_buffer),
                          env.last_action.copy(), env.previous_action.copy(), step, env.phase,
                          env.recovery_initial_gravity.copy(), env.data.ctrl.copy())
                for key, value in zip(arrays, values):
                    arrays[key].append(value)
            executed = np.clip(action + rng.normal(0.0, action_noise, 23), -1, 1) if action_noise else action
            observation, reward, terminated, truncated, info = env.step(executed)
            score += float(reward) * 0.995 ** step
            if terminated or truncated:
                break
        return {**info, "seed": seed, "score": score,
                "success": bool(info["success"] and initial.get("disturbance_produced_fall", True))}, {
                    key: np.asarray(value) for key, value in arrays.items()} if record else {}
    finally:
        env.close()


def _score(task: tuple[dict, list[int]]) -> tuple[float, float]:
    value, seeds = task
    teacher = MotionTeacher(value["skill"], value["pose"], np.asarray(value["targets"]),
                            np.asarray(value["durations"]), np.asarray(value["feedback"]),
                            value["conditions"], value["scenario"])
    rows = [rollout(teacher, seed)[0] for seed in seeds]
    return float(np.mean([row["success"] for row in rows])), float(np.mean([row["score"] for row in rows]))


def search(skill: str, pose: str, output: Path, seed: int, seed_start: int, hours: float,
           workers: int = 8, population: int = 32, iterations: int = 12) -> dict:
    initial = template(skill, pose)
    env = SkillEnv(skill, randomize=False)
    lower, upper = env.joint_ranges.T
    env.close()
    rng = np.random.default_rng(seed)
    # The last hold target remains standing. Every joint in the preceding five phases is searchable.
    mean = np.r_[initial.targets[:5].ravel(), np.log(initial.durations[:5]), initial.feedback]
    spread = np.r_[np.full(115, 0.18), np.full(5, 0.15), np.full(4, 0.15)]
    best, best_score = None, (-1.0, -np.inf)
    deadline = time.monotonic() + hours * 3600
    output.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for iteration in range(iterations):
            if time.monotonic() >= deadline:
                break
            candidates = rng.normal(mean, spread, (population, len(mean)))
            candidates[0] = mean
            if best is not None:
                candidates[1] = best
            documents = []
            for candidate in candidates:
                targets = np.vstack((np.clip(candidate[:115].reshape(5, 23), lower, upper), STANDING_POSE))
                durations = np.clip(np.exp(candidate[115:120]), 5, 100).astype(int)
                cap = MAX_STEPS[skill] - 120
                if durations.sum() > cap:
                    durations = np.maximum(1, (durations * cap / durations.sum()).astype(int))
                durations = np.r_[durations, 120]
                teacher = MotionTeacher(skill, pose, targets, durations, np.clip(candidate[120:], -4, 4),
                                        initial.conditions, initial.scenario)
                documents.append(teacher.document())
            results = pool.map(_score, [(document, list(range(seed_start, seed_start + 4)))
                                        for document in documents])
            order = sorted(range(population), key=lambda index: results[index])
            winner = order[-1]
            if results[winner] > best_score:
                best_score, best = results[winner], candidates[winner].copy()
                output.write_text(json.dumps({**documents[winner], "search_seed": seed,
                                              "training_seed_start": seed_start, "search_rate": best_score[0],
                                              "search_criterion_met": best_score[0] >= 0.8,
                                              "status": "experimental_teacher"}, indent=2) + "\n")
            elite = candidates[order[-max(2, population // 8):]]
            mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
            spread = np.maximum(0.025, 0.25 * spread + 0.75 * elite.std(axis=0))
            print(f"[teacher] {skill}/{pose} generation={iteration} best_success={best_score[0]:.1%} "
                  f"score={best_score[1]:.2f}", flush=True)
            if best_score[0] >= 0.8:
                break
    return {"search_success": best_score[0], "teacher": str(output), "status": "requires_separate_validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("recover", "jump"), required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--action-noise", type=float, default=0)
    args = parser.parse_args()
    if (args.hours <= 0 or args.workers < 1 or args.population < 4 or args.iterations < 1 or args.episodes < 1
            or not 0 <= args.action_noise <= 0.2):
        parser.error("Invalid teacher budget")
    template(args.skill, args.pose).validate()
    if args.verify:
        teacher = MotionTeacher.load(args.verify)
        rows, trajectories = [], []
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            row, trajectory = rollout(teacher, seed, bool(args.dataset), args.action_noise)
            rows.append(row)
            if row["success"] and args.dataset:
                trajectories.append(trajectory)
        report = {"teacher": str(args.verify), "seed_start": args.seed_start, "episodes": rows,
                  "pass_rate": sum(row["success"] for row in rows) / len(rows)}
        report["criterion_met"] = report["pass_rate"] >= 0.8
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.dataset and trajectories:
            starts = np.cumsum([0] + [len(item["actions"]) for item in trajectories[:-1]])
            np.savez_compressed(args.dataset, episode_starts=starts,
                                **{key: np.concatenate([item[key] for item in trajectories]) for key in trajectories[0]})
        print(json.dumps({key: value for key, value in report.items() if key != "episodes"}), flush=True)
    else:
        print(json.dumps(search(args.skill, args.pose, args.output, args.seed, args.seed_start, args.hours,
                                args.workers, args.population, args.iterations)), flush=True)


if __name__ == "__main__":
    main()
