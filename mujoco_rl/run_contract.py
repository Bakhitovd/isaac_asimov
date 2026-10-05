"""Versioned running environment and evaluation contract; legacy tasks stay opt-in."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import mujoco

from .full_body_env import STANDING_POSE
from .skill_env import SkillEnv


CONTRACT = "asimov_running_v1"
STAGES = ("track_030", "track_040", "track_050", "track_060", "gait", "brake", "random_half", "random_full")
DEFAULT_CONFIG = {
    "contract": CONTRACT, "action_scale": 0.25, "workers": 8, "rollout_steps": 768,
    "batch_size": 512, "epochs": 4, "learning_rate": 3e-5, "clip_range": 0.1,
    "target_kl": 0.015, "gamma": 0.995, "max_grad_norm": 1.0, "warmup_rollouts": 2,
    "initial_std": 0.15, "evaluation_interval": 100000, "development_episodes": 20,
    "block_steps": 1000000, "maximum_steps": 10000000, "maximum_hours": 72,
    "qualification_hours": 6, "seeds": [44, 45], "exploration_settings": [0.08, 0.15],
    "rewards": {"tracking": 3.0, "upright": 1.0, "flight": 0.2, "slip": -4.0,
                "lateral": -1.5, "action_rate": -0.15, "action_accel": -0.05,
                "effort": -0.03, "fall": -100.0},
}


def config_from(value: dict | Path | None = None) -> dict:
    supplied = json.loads(value.read_text()) if isinstance(value, Path) else (value or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown running settings: {sorted(unknown)}")
    result = {**deepcopy(DEFAULT_CONFIG), **deepcopy(supplied)}
    if result["contract"] != CONTRACT or result["action_scale"] != 0.25:
        raise ValueError("Incompatible running contract/action mapping")
    if result["rewards"] != DEFAULT_CONFIG["rewards"]:
        raise ValueError("Reward changes require a new contract version")
    for name in ("workers", "rollout_steps", "batch_size", "epochs", "evaluation_interval",
                 "development_episodes", "block_steps", "maximum_steps", "warmup_rollouts"):
        if not isinstance(result[name], int) or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "clip_range", "target_kl", "gamma", "max_grad_norm", "initial_std",
                 "maximum_hours", "qualification_hours"):
        if not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if result["qualification_hours"] >= result["maximum_hours"] or result["workers"] > 8:
        raise ValueError("Invalid campaign resource budget")
    if result["batch_size"] > result["workers"] * result["rollout_steps"]:
        raise ValueError("Batch exceeds rollout")
    return result


class RunningEnv(SkillEnv):
    """Stage changes take effect on reset; no rehearsal or teacher dependency."""

    def __init__(self, config: dict | None = None, stage: int = 0, render_mode=None):
        self.run_contract = config_from(config)
        self.pending_stage = int(stage)
        self.stage = int(stage)
        if stage not in range(len(STAGES)):
            raise ValueError("Invalid running stage")
        super().__init__("run", render_mode=render_mode, randomize=stage >= 6)
        self.speed_errors = []
        self.reward_totals = {}
        self.action_clips = []
        self.standing_override = False

    def set_run_stage(self, stage: int):
        if stage not in range(len(STAGES)):
            raise ValueError("Invalid running stage")
        self.pending_stage = int(stage)

    def reset(self, *, seed=None, options=None):
        self.stage = self.pending_stage
        self.randomize = self.stage >= 6
        self.randomization_strength = 0.5 if self.stage == 6 else 1.0
        self.speed_errors, self.action_clips, self.reward_totals = [], [], {}
        self.standing_override = False
        obs, info = super().reset(seed=seed, options=options)
        self.command[:] = (self._run_command(), 0.0)
        obs[6] = self.command[0]
        return obs, {**info, "run_stage": STAGES[self.stage]}

    def _target_from_action(self, action):
        return np.clip(STANDING_POSE + self.run_contract["action_scale"] * action,
                       self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _run_command(self):
        if self.standing_override:
            return 0.0
        speed = (0.3, 0.4, 0.5, 0.6)[min(self.stage, 3)]
        if self.stage >= 5 and self.step_count >= 500:
            return 0.0
        return min(speed, 0.2 + 0.004 * self.step_count) if self.step_count < 100 else speed

    def _run_flight_reward(self):
        return super()._run_flight_reward() if self.stage >= 4 else 0.0

    def step(self, action):
        self.action_clips.append(float(np.mean(np.abs(action) >= 0.99)))
        obs, reward, terminated, truncated, info = super().step(action)
        velocity = self.data.sensor("imu_lin_vel").data
        if 100 <= self.step_count < (500 if self.stage >= 5 else 600):
            self.speed_errors.append(abs(float(velocity[0]) - self.command[0]))
        error = float(np.mean(self.speed_errors)) if self.speed_errors else None
        if self.stage < 5:
            self.success = bool(self.step_count >= 600 and not info["fallen"]
                                and error is not None and error <= 0.10
                                and abs(self.data.qpos[1] - self.initial_xy[1]) < 0.5
                                and (self.stage < 4 or self.run_flights >= 5))
            terminated = bool(info["fallen"] or info["termination_reason"] == "nonfinite" or self.success)
            info["success"] = self.success
            info["termination_reason"] = ("success" if self.success else info["termination_reason"])
        gravity = self.data.site_xmat[self.imu_site_id].reshape(3, 3).T @ np.array([0., 0., -1.])
        parts = {"tracking": 3 * math.exp(-((float(velocity[0]) - self.command[0]) / 0.2) ** 2),
                 "upright": math.exp(-float(gravity[0] ** 2 + gravity[1] ** 2) / 0.1),
                 "flight": self._run_flight_reward(), "slip": -4 * info["slip_m_s"],
                 "lateral": -1.5 * abs(float(velocity[1])), "action_rate": -0.15 * info["action_rate"],
                 "action_accel": -0.05 * info["action_accel"], "effort": -0.03 * info["effort"],
                 "fall": -100.0 if info["fallen"] or info["termination_reason"] == "nonfinite" else 0.0}
        for key, value in parts.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.0) + value
        info.update(run_stage=STAGES[self.stage], run_stage_index=self.stage, speed_error_m_s=error,
                    action_clipping_fraction=float(np.mean(self.action_clips)),
                    reward_components=parts, reward_totals=dict(self.reward_totals))
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera=self.camera)
        # Software-rendered diagnostics do not need expensive shadow maps.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return self.renderer.render().copy()


def environment_for(model, skill: str, *, stage: int | None = None, render_mode=None, **legacy):
    contract = getattr(model, "run_contract", None)
    if contract is not None:
        if skill != "run":
            raise ValueError("A specialized running checkpoint must be evaluated as run")
        return RunningEnv(contract, stage=7 if stage is None else stage, render_mode=render_mode)
    return SkillEnv(skill, render_mode=render_mode, **legacy)


def summarize(episodes: list[dict]) -> dict:
    errors = [e["speed_error_m_s"] for e in episodes if e.get("speed_error_m_s") is not None]
    return {"pass_rate": float(np.mean([e["success"] for e in episodes])),
            "survival_seconds": float(np.mean([e["seconds"] for e in episodes])),
            "speed_error_m_s": float(np.mean(errors)) if errors else None,
            "flight_events": float(np.mean([e["flight_events"] for e in episodes])),
            "action_clipping_fraction": float(np.mean([e.get("action_clipping_fraction", 0) for e in episodes]))}


def made_progress(before: dict, after: dict) -> bool:
    if after["stage"] != before["stage"]:
        return after["stage"] > before["stage"]
    b, a = before["summary"], after["summary"]
    speed_b, speed_a = b["speed_error_m_s"], a["speed_error_m_s"]
    return bool(a["pass_rate"] - b["pass_rate"] >= 0.10 - 1e-9
                or a["survival_seconds"] >= 1.1 * b["survival_seconds"]
                or (speed_a is not None and speed_b is not None and speed_b > 0
                    and speed_a <= 0.9 * speed_b and a["survival_seconds"] >= b["survival_seconds"])
                or (speed_a is not None and speed_b is not None and speed_a <= speed_b
                    and a["flight_events"] >= b["flight_events"] + 1
                    and a["survival_seconds"] >= b["survival_seconds"]))
