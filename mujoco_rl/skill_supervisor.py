"""Command and recovery supervisor for accepted MuJoCo skill policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .skill_env import SkillEnv
from .skill_scenarios import disturb


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
            qualification = checkpoint.parent / "qualification.json"
            if qualification.exists():
                metadata = json.loads(qualification.read_text())
                if (metadata.get("environment_version") != 12
                        or metadata.get("sha256", {}).get("accepted.zip") != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        or metadata.get("normalization_sha256") != hashlib.sha256(stats.read_bytes()).hexdigest()):
                    self.close()
                    raise ValueError(f"Checkpoint bundle integrity/version mismatch: {skill}")
            normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda skill=skill: SkillEnv(skill)]))
            normalizer.training = False
            model = PPO.load(str(checkpoint), device="cpu")
            if getattr(model, "run_contract", None) is not None:
                normalizer.close()
                self.close()
                raise ValueError("Versioned running checkpoints require run_supervisor.RunningSupervisor; "
                                 "the legacy run_to controller uses a different action contract")
            if model.observation_space.shape != (83,) or model.action_space.shape != (23,):
                normalizer.close()
                self.close()
                raise ValueError(f"Incompatible {skill} policy: expected 83 observations and 23 actions")
            if normalizer.obs_rms.mean.shape != (83,):
                normalizer.close()
                self.close()
                raise ValueError(f"Incompatible {skill} normalization")
            self.slots[skill] = (model, normalizer)
        self.env.set_recovery_sensor_memory(getattr(self.slots["recover"][0], "recovery_sensor_memory", False))
        self.env.set_recovery_phase_features(getattr(self.slots["recover"][0], "recovery_phase_features", False))
        self.active = "nav"
        self.gait = "walk"
        self.pending_goal: np.ndarray | None = None
        self.pending_speed = 0.30
        self.events: list[str] = []
        self.faulted = False
        self.running_segment = False
        self.disturbance_steps = 0
        self.disturbance_pose = "left"

    def command(self, name: str, target_xy: tuple[float, float] | None = None) -> None:
        if self.faulted:
            raise RuntimeError("Recovery failed; reset the simulator before sending another command")
        if name in {"go_to", "run_to"}:
            if target_xy is None:
                raise ValueError("Travel commands require target_xy")
            if np.asarray(target_xy).shape != (2,) or not np.isfinite(target_xy).all():
                raise ValueError("Travel target must contain two finite coordinates")
            self.pending_goal = np.asarray(target_xy, dtype=float).copy()
            self.pending_speed = 0.60 if name == "run_to" else 0.30
            self.gait = "run" if name == "run_to" else "walk"
            self.active = "nav"
            self.env.switch_skill("nav", self.pending_goal, 0.30)
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
        self.running_segment = False
        self.observation = self.env._observe(advance_sensors=False)

    def step(self) -> dict:
        if self.faulted:
            raise RuntimeError("Recovery failed; reset the simulator before stepping")
        if self.active == "nav" and self.gait == "run":
            distance = float(np.linalg.norm(self.env.goal_body))
            angle = abs(float(np.arctan2(self.env.goal_body[1], self.env.goal_body[0])))
            if self.running_segment:
                self.running_segment = distance > 0.6 and angle <= np.deg2rad(20)
            else:
                self.running_segment = distance > 0.8 and angle < np.deg2rad(15)
            speed = 0.6 if self.running_segment else 0.3
            if speed != self.env.nav_max_speed:
                self.env.set_navigation_speed(speed)
            else:
                self.env._goal_and_command()
            # Update command channels without advancing the sensor delay buffer.
            self.observation[6], self.observation[78] = self.env.command
        slot = "run" if self.active == "nav" and self.running_segment else self.active
        model, normalizer = self.slots[slot]
        action, _ = model.predict(normalizer.normalize_obs(self.observation[None, :]), deterministic=True)
        if self.disturbance_steps:
            disturb(self.env, self.disturbance_pose)
            self.disturbance_steps -= 1
        else:
            self.env.data.xfrc_applied[:] = 0.0
        try:
            self.observation, _, terminated, truncated, info = self.env.step(action[0])
        except (ValueError, RuntimeError):
            if np.isfinite(self.env.data.qpos).all() and np.isfinite(self.env.data.qvel).all():
                raise
            self.faulted = True
            self.env.data.ctrl[:] = 0.0
            self.events.append(f"failed:{self.active}:nonfinite")
            return {"success": False, "termination_reason": "nonfinite", "supervisor_mode": self.active,
                    "events": self.events.copy(), "physical_terminated": True}
        if info.get("termination_reason") == "nonfinite":
            self.events.append(f"failed:{self.active}:nonfinite")
            self.faulted = True
            self.env.data.ctrl[:] = 0.0
        elif info["fallen"] and self.active != "recover":
            self.active = "recover"
            self.running_segment = False
            self.env.switch_skill("recover")
            self.observation = self.env._observe(advance_sensors=False)
            self.events.append("fall:recover")
        elif self.active == "recover" and info["success"]:
            self.events.append("recovered")
            if self.pending_goal is not None:
                self.active = "nav"
                self.env.switch_skill("nav", self.pending_goal, 0.30)
            else:
                self.command("stand")
            self.observation = self.env._observe(advance_sensors=False)
        elif info["success"]:
            self.events.append(f"completed:{self.active}")
            self.command("stand")
        elif truncated or (terminated and self.active == "recover"):
            self.events.append(f"failed:{self.active}")
            if self.active == "recover":
                self.faulted = True
                self.env.data.ctrl[:] = 0.0
            else:
                self.command("stand")
        return {**info, "supervisor_mode": self.active, "events": self.events.copy(),
                "physical_terminated": terminated}

    def inject_fall(self, pose: str = "left", mode: str = "prepared") -> None:
        """Simulation-only disturbance for a chained evaluation episode."""
        if pose not in {"front", "back", "left", "right"} or mode not in {"prepared", "dynamic"}:
            raise ValueError("Invalid fall pose or injection mode")
        if mode == "dynamic":
            self.disturbance_pose, self.disturbance_steps = pose, 20
            self.events.append(f"injected_dynamic_fall:{pose}")
            return
        self.env._set_recovery_pose(pose, 90.0)
        self.observation = self.env._observe(advance_sensors=False)
        self.events.append(f"injected_fall:{pose}")

    def close(self) -> None:
        for _, normalizer in getattr(self, "slots", {}).values():
            normalizer.close()
        self.env.close()


def evaluate_chains(checkpoint_dir: Path, seed_start: int = 20_001, seeds: int = 50,
                    fall_mode: str = "dynamic") -> dict:
    episodes = []
    for seed in range(seed_start, seed_start + seeds):
        supervisor = SkillSupervisor(checkpoint_dir, seed=seed)
        try:
            def until(completed: str, limit: int, inject_fall: bool = False) -> bool:
                start = len(supervisor.events)
                for step in range(limit):
                    if inject_fall and step == 100:
                        supervisor.inject_fall(("front", "back", "left", "right")[seed % 4], fall_mode)
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
    return {"pass_rate": round(rate, 3), "criterion_met": rate >= 0.80,
            "fall_mode": fall_mode, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=20_001)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--fall-mode", choices=("prepared", "dynamic"), default="dynamic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_chains(args.checkpoint_dir, args.seed_start, args.seeds, args.fall_mode)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
