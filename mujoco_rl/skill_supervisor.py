"""Command and recovery supervisor for accepted MuJoCo skill policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .skill_env import SkillEnv


class SkillSupervisor:
    def __init__(self, checkpoint_dir: Path, seed: int = 0):
        self.env = SkillEnv("nav", randomize=True)
        self.observation, _ = self.env.reset(seed=seed)
        self.slots = {}
        for skill in ("nav", "squat", "recover", "run", "jump"):
            checkpoint = checkpoint_dir / skill / "accepted.zip"
            stats = checkpoint.with_name("accepted_vecnormalize.pkl")
            if not checkpoint.is_file() or not stats.is_file():
                self.close()
                raise FileNotFoundError(f"Missing accepted {skill} checkpoint or statistics")
            normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda skill=skill: SkillEnv(skill)]))
            normalizer.training = False
            self.slots[skill] = (PPO.load(str(checkpoint), device="cpu"), normalizer)
        self.active = "nav"
        self.gait = "walk"
        self.pending_goal: np.ndarray | None = None
        self.pending_speed = 0.30
        self.events: list[str] = []
        self.faulted = False

    def command(self, name: str, target_xy: tuple[float, float] | None = None) -> None:
        if self.faulted:
            raise RuntimeError("Recovery failed; reset the simulator before sending another command")
        if name in {"go_to", "run_to"}:
            if target_xy is None:
                raise ValueError("Travel commands require target_xy")
            self.pending_goal = np.asarray(target_xy, dtype=float).copy()
            self.pending_speed = 0.60 if name == "run_to" else 0.30
            self.gait = "run" if name == "run_to" else "walk"
            self.active = "nav"
            self.env.switch_skill("nav", self.pending_goal, self.pending_speed)
        elif name == "stand":
            self.pending_goal = None
            self.gait = "walk"
            self.active = "nav"
            self.env.switch_skill("nav", self.env.data.qpos[:2])
        elif name in {"squat", "jump_up", "jump_forward"}:
            self.pending_goal = None
            self.active = "jump" if name.startswith("jump") else "squat"
            self.env.switch_skill(self.active)
            if self.active == "jump":
                self.env.jump_forward = name == "jump_forward"
                self.env.goal_body[0] = 0.2 if self.env.jump_forward else 0.0
        else:
            raise ValueError(f"Unknown command {name}")
        self.events.append(f"command:{name}")
        self.observation = self.env._observe()

    def step(self) -> dict:
        if self.faulted:
            raise RuntimeError("Recovery failed; reset the simulator before stepping")
        slot = "run" if self.active == "nav" and self.gait == "run" else self.active
        model, normalizer = self.slots[slot]
        action, _ = model.predict(normalizer.normalize_obs(self.observation[None, :]), deterministic=True)
        self.observation, _, terminated, truncated, info = self.env.step(action[0])
        if info["fallen"] and self.active != "recover":
            self.active = "recover"
            self.env.switch_skill("recover")
            self.observation = self.env._observe()
            self.events.append("fall:recover")
        elif self.active == "recover" and info["success"]:
            self.events.append("recovered")
            if self.pending_goal is not None:
                self.active = "nav"
                self.env.switch_skill("nav", self.pending_goal, self.pending_speed)
            else:
                self.command("stand")
            self.observation = self.env._observe()
        elif info["success"]:
            self.events.append(f"completed:{self.active}")
            self.command("stand")
        elif truncated:
            self.events.append(f"failed:{self.active}")
            if self.active == "recover":
                self.faulted = True
                self.env.data.ctrl[:] = 0.0
            else:
                self.command("stand")
        return {**info, "supervisor_mode": self.active, "events": self.events.copy(),
                "physical_terminated": terminated}

    def inject_fall(self, pose: str = "left") -> None:
        """Simulation-only disturbance for a chained evaluation episode."""
        self.env._set_recovery_pose(pose)
        self.observation = self.env._observe()
        self.events.append(f"injected_fall:{pose}")

    def close(self) -> None:
        for _, normalizer in getattr(self, "slots", {}).values():
            normalizer.close()
        self.env.close()


def evaluate_chains(checkpoint_dir: Path, seed_start: int = 20_001, seeds: int = 50) -> dict:
    episodes = []
    for seed in range(seed_start, seed_start + seeds):
        supervisor = SkillSupervisor(checkpoint_dir, seed=seed)
        try:
            def until(completed: str, limit: int, inject_fall: bool = False) -> bool:
                start = len(supervisor.events)
                for step in range(limit):
                    if inject_fall and step == 100:
                        supervisor.inject_fall(("front", "back", "left", "right")[seed % 4])
                    supervisor.step()
                    new_events = supervisor.events[start:]
                    if completed in new_events:
                        return True
                    if any(event.startswith("failed:") for event in new_events):
                        return False
                return False

            goal = tuple(supervisor.env.data.qpos[:2] + np.array([1.5, 0.5]))
            supervisor.command("go_to", goal)
            reached = until("completed:nav", 1800, inject_fall=True)
            recovered = "recovered" in supervisor.events
            results = {"travel_after_fall": reached and recovered}
            if results["travel_after_fall"]:
                for command, event, limit in (("squat", "completed:squat", 800),
                                               ("jump_up", "completed:jump", 450),
                                               ("jump_forward", "completed:jump", 450)):
                    supervisor.command(command)
                    results[command] = until(event, limit)
                    if not results[command]:
                        break
            if all(results.values()):
                run_goal = tuple(supervisor.env.data.qpos[:2] + np.array([1.5, 0.0]))
                supervisor.command("run_to", run_goal)
                results["run_to"] = until("completed:nav", 1800)
            episodes.append({"seed": seed, "passed": all(results.values()),
                             "results": results, "events": supervisor.events.copy()})
        finally:
            supervisor.close()
    rate = sum(item["passed"] for item in episodes) / seeds
    return {"pass_rate": round(rate, 3), "criterion_met": rate >= 0.80, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=20_001)
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(evaluate_chains(args.checkpoint_dir, args.seed_start, args.seeds), indent=2))


if __name__ == "__main__":
    main()
