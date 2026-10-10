"""Command walking v2: independent disturbances and observable position holding."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from gymnasium import spaces

from .command_walk import FAMILIES, CommandWalkEnv, segments_for
from .full_body_env import POLICY_DT
from .skill_env import _yaw

CONTRACT = "asimov_command_walk_v2"


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    variation: float = 0.0
    action_probability: float = 0.0
    sensor_probability: float = 0.0
    two_frame_probability: float = 0.0
    noise_scale: float = 1.0
    gain_scale: float = 1.0
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    push_scale: float = 1.0


PROFILES = (
    PhysicsProfile("nominal"),
    PhysicsProfile("half_without_delays", 0.5),
    PhysicsProfile("action_10", 0.5, 0.10),
    PhysicsProfile("action_25", 0.5, 0.25),
    PhysicsProfile("action_50", 0.5, 0.50),
    PhysicsProfile("sensor_10", 0.5, 0.50, 0.10),
    PhysicsProfile("sensor_25", 0.5, 0.50, 0.25),
    PhysicsProfile("combined_50", 0.5, 0.50, 0.50),
    PhysicsProfile("half_physics", 0.5, 2 / 3, 2 / 3),
    PhysicsProfile("full_variation", 1.0, 2 / 3, 2 / 3),
    PhysicsProfile("two_frames_10", 1.0, 2 / 3, 2 / 3, 0.10),
    PhysicsProfile("two_frames_25", 1.0, 2 / 3, 2 / 3, 0.25),
    PhysicsProfile("full_physics", 1.0, 2 / 3, 2 / 3, 0.50),
)
CHALLENGE_LEVEL = 8
# Time allocation: compensate for mixed episodes being longer than the others.
TIME_WEIGHTS = np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.10, 0.20])
EPISODE_WEIGHTS = TIME_WEIGHTS / np.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 28.0])
EPISODE_WEIGHTS /= EPISODE_WEIGHTS.sum()


def holding_costs(radius, speed, yaw_rate, requested, age):
    """Keep a 5 cm margin; braking gets the same two seconds as the evaluator."""
    if requested[0] > 1e-6:
        return {"hold_drift": 0.0, "hold_speed": 0.0, "stop_speed": 0.0, "stop_yaw": 0.0}
    stopped = abs(requested[1]) < 1e-6
    settle = min(1.0, max(0.0, age / 2.0)) if stopped else 1.0
    return {
        "hold_drift": -2.0 * min((max(0.0, radius - 0.05) / 0.20) ** 2, 4.0) * settle,
        "hold_speed": -0.25 * min((speed / 0.08) ** 2, 4.0),
        "stop_speed": -min((speed / 0.05) ** 2, 4.0) * settle if stopped else 0.0,
        "stop_yaw": -0.5 * min((yaw_rate / 0.10) ** 2, 4.0) * settle if stopped else 0.0,
    }


class RobustCommandEnv(CommandWalkEnv):
    """The first 85 inputs retain v1 semantics; two appended inputs encode hold offset.

    Offset uses integrated simulated velocity with measurement noise, not root position.
    This is an odometry assumption, not a claim that an IMU measures linear velocity.
    Physics, observations, odometry, pushes, and case sampling have separate RNGs.
    """

    def __init__(self, level=0, variant="aligned", training=False, **kwargs):
        if variant not in {"control", "aligned"} or level not in range(len(PROFILES)):
            raise ValueError("Invalid v2 variant or curriculum level")
        self.level = level
        self.variant, self.training_cases = variant, training
        self.profile = PROFILES[0]
        self.episode_rng = np.random.default_rng(0)
        self.noise_rng = np.random.default_rng(1)
        self.push_rng = np.random.default_rng(2)
        self.odom_rng = np.random.default_rng(3)
        self.noise_step, self.noise = -1, np.zeros(85)
        self.odometry = np.zeros(2)
        self.hold_odometry = np.zeros(2)
        self.hold_origin = np.zeros(2)
        self.holding = False
        self.hold_started = 0
        self.resetting = True
        self.applied_level = 0
        super().__init__(stage=1, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (87,), dtype=np.float32)
        self.resetting = False

    def set_level(self, level):
        if level not in range(len(PROFILES)):
            raise ValueError("Invalid curriculum level")
        self.level = level

    def _observe(self, advance_sensors=True):
        # Parent step/reset call this more than once: one noise sample per sensor tick.
        randomized = self.randomize
        self.randomize = False
        try:
            base = super()._observe(advance_sensors)
        finally:
            self.randomize = randomized
        if not self.resetting:
            if self.noise_step != self.step_count:
                self.noise_step = self.step_count
                scales = np.zeros(85)
                scales[:3], scales[3:6] = 0.0025, 0.01
                scales[7:30], scales[30:53], scales[79:81] = 0.005, 0.005, 0.015
                self.noise = self.noise_rng.normal(size=85) * scales * self.profile.variation * self.profile.noise_scale
            base += self.noise.astype(np.float32)
        offset = np.zeros(2)
        if self.holding and self.variant == "aligned" and not self.resetting:
            dx, dy = self.odometry - self.hold_odometry
            yaw = _yaw(self.data.qpos[3:7])
            offset[:] = (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)
        return np.r_[base, np.clip(offset / 0.25, -4.0, 4.0)].astype(np.float32)

    def _prepare_command(self):
        if getattr(self, "skip_prepare", False):
            return
        previous_segment = getattr(self, "segment", 0)
        super()._prepare_command()
        if self.resetting:
            return
        holding = self.requested[0] <= 1e-6
        new_segment = not self.external and self.segment != previous_segment
        if holding and (not self.holding or new_segment):
            self.hold_origin = self.data.qpos[:2].copy()
            self.hold_odometry = self.odometry.copy()
            self.hold_started = self.step_count
        self.holding = holding

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.episode_rng = np.random.default_rng(seed)
            case_seed = int(seed)
        else:
            case_seed = int(self.episode_rng.integers(0, 2**31))
        streams = np.random.SeedSequence([case_seed, 873]).spawn(5)
        physics, selection = (np.random.default_rng(s) for s in streams[:2])
        self.noise_rng, self.push_rng, self.odom_rng = (np.random.default_rng(s) for s in streams[2:])
        # Case generation is always nominal, so profile changes cannot change poses or commands.
        self.resetting = True
        self.pending_stage = 1
        self.holding = False
        super().reset(seed=case_seed, options=options)
        if self.training_cases:
            self.family = str(selection.choice(FAMILIES, p=EPISODE_WEIGHTS))
            speed, turn = float(selection.uniform(0.1, 0.3)), float(selection.uniform(0.15, 0.6))
            self.schedule = segments_for(self.family, speed, turn)
            self.duration = sum(row[0] for row in self.schedule)
            self.command[:] = self.requested[:] = 0.0
        self.applied_level = self.level
        if self.training_cases:
            u = selection.random()
            if u < 0.30:
                self.applied_level = 0
            elif u < 0.50 and self.level > 1:
                self.applied_level = int(selection.integers(1, self.level))
        self.profile = PROFILES[self.applied_level]
        strength = self.profile.variation
        self.model.body_mass[:] = self.base_mass
        self.model.geom_friction[:] = self.foot_friction
        self.model.body_mass[self.torso_id] *= 1.0 + strength * self.profile.mass_scale * physics.uniform(-0.05, 0.05)
        friction = 1.0 + strength * self.profile.friction_scale * physics.uniform(-0.2, 0.2)
        for foot in self.foot_geom_ids:
            self.model.geom_friction[list(foot), 0] *= friction
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        self.kp_scale = 1.0 + strength * self.profile.gain_scale * physics.uniform(-0.2, 0.2)
        self.kd_scale = 1.0 + strength * self.profile.gain_scale * physics.uniform(-0.2, 0.2)
        for name, probability in [
            ("action_lag", self.profile.action_probability),
            ("sensor_lag", self.profile.sensor_probability),
        ]:
            present, extra = physics.random(2)
            setattr(self, name, int(present < probability) * (1 + int(extra < self.profile.two_frame_probability)))
        mujoco.mj_forward(self.model, self.data)
        self.sensor_buffer.clear()
        for _ in range(3):
            self.sensor_buffer.append(self._raw_sensors())
        self.odometry[:] = self.hold_odometry[:] = 0.0
        self.hold_origin = self.data.qpos[:2].copy()
        self.next_robust_push = int(self.push_rng.integers(100, 200))
        self.noise_step = -1
        self.resetting = False
        self.command[:] = self.requested[:] = 0.0
        self._prepare_command()
        return self._observe(False), {"contract": CONTRACT, "profile": asdict(self.profile), "family": self.family}

    def step(self, action):
        if self.external:
            # CommandWalkEnv also prepares external commands; avoid a second slew below.
            self._prepare_command()
        old_external = self.external
        applied_requested = self.requested.copy()
        origin = self.hold_origin.copy()
        age = (self.step_count + 1 - self.hold_started) * POLICY_DT
        # Parent randomization stays disabled: independent noise and pushes are owned here.
        self.randomize = False
        push = self.step_count + 1 == self.next_robust_push and np.any(self.requested)
        if self.step_count + 1 == self.next_robust_push:
            impulse = self.push_rng.uniform(-0.1, 0.1, 2) * self.profile.variation * self.profile.push_scale
            self.next_robust_push += int(self.push_rng.integers(100, 200))
            if push:
                self.data.qvel[:2] += impulse
                mujoco.mj_forward(self.model, self.data)
        # Prevent double preparation for external controls without enabling scripted commands.
        self.skip_prepare = old_external
        obs, reward, terminated, truncated, info = super().step(action)
        self.skip_prepare = False
        self.odometry += (
            self.data.qvel[:2] + self.odom_rng.normal(0.0, 0.03 * self.profile.variation * self.profile.noise_scale, 2)
        ) * POLICY_DT
        if self.holding and self.hold_started == self.step_count:
            self.hold_odometry = self.odometry.copy()
        costs = holding_costs(
            float(np.linalg.norm(self.data.qpos[:2] - origin)), info["speed"], info["yaw_rate"], applied_requested, age
        )
        if self.variant == "control":
            costs = {k: 0.0 for k in costs}
        for k, v in costs.items():
            self.reward_totals[k] = self.reward_totals.get(k, 0.0) + v
        info["reward_components"].update(costs)
        info.update(
            contract=CONTRACT,
            profile=self.profile.name,
            level=self.applied_level,
            reward_totals=self.reward_totals.copy(),
            push_applied=bool(push and self.profile.variation and self.profile.push_scale),
        )
        return self._observe(False), reward + sum(costs.values()), terminated, truncated, info
