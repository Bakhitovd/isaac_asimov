"""Forward walking task using native, CPU MuJoCo."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "asimov"
XML_PATH = ASSET_DIR / "xmls" / "asimov.xml"
REFERENCE_PATH = ASSET_DIR / "assets" / "walking_reference.csv"

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)

# Values from menloresearch/asimov-mjlab at the pinned asset commit.
ARMATURE = np.array([0.0652, 0.1000, 0.0343, 0.0330, 0.0236, 0.0236] * 2)
EFFORT_LIMIT = np.array([120.0, 90.0, 60.0, 75.0, 36.0, 36.0] * 2)
KP = ARMATURE * (2.0 * math.pi * 10.0) ** 2
KD = np.full(12, 5.0)
PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
TARGET_FORWARD_SPEED = 0.35
GAIT_FREQUENCY = 1.25


def _load_reference() -> np.ndarray:
    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        columns = handle.readline().strip().split(",")
    samples = np.loadtxt(REFERENCE_PATH, delimiter=",", skiprows=1, dtype=np.float64)
    samples = samples[:, [columns.index(name) for name in JOINT_NAMES]]
    if samples.shape != (1000, 12):
        raise ValueError(f"Unexpected walking reference shape: {samples.shape}")
    return samples.reshape(25, 40, 12).mean(axis=0)


class AsimovForwardEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a 12-joint policy to move forward from a standing pose."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, reward_stage: str = "target"):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if reward_stage not in ("progress", "target"):
            raise ValueError(f"Unsupported reward stage: {reward_stage}")
        self.render_mode = render_mode
        self.reward_stage = reward_stage
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.model.opt.timestep = PHYSICS_DT
        self.model.opt.iterations = 10
        self.data = mujoco.MjData(self.model)
        self.reference = _load_reference()

        joints = [self.model.joint(name) for name in JOINT_NAMES]
        self.qpos_ids = np.array([joint.qposadr[0] for joint in joints], dtype=np.int32)
        self.qvel_ids = np.array([joint.dofadr[0] for joint in joints], dtype=np.int32)
        self.ctrl_ids = np.array([self.model.actuator(name + "_ctrl").id for name in JOINT_NAMES])
        self.joint_limits = np.array([joint.range.copy() for joint in joints])
        self.imu_site_id = self.model.site("imu_in_pelvis").id
        self.floor_geom_id = self.model.geom("floor").id
        self.foot_geom_ids = np.array([
            self.model.geom("left_ankle_roll_link_collision").id,
            self.model.geom("right_ankle_roll_link_collision").id,
        ])

        # Match Menlo's feet-only collision configuration: visual meshes and
        # other body meshes must not collide with the floor in this first task.
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.model.geom_contype[self.floor_geom_id] = 1
        self.model.geom_conaffinity[self.floor_geom_id] = 1
        self.model.geom_contype[self.foot_geom_ids] = 1
        self.model.geom_conaffinity[self.foot_geom_ids] = 1
        self.model.geom_friction[self.foot_geom_ids, 0] = 0.6

        # A normalized action of 1 reaches at least the reference gait's
        # excursion; physical joint limits and motor torque still apply.
        reference_excursion = 1.15 * np.max(np.abs(self.reference), axis=0)
        self.action_scale = np.maximum(0.3 * EFFORT_LIMIT / KP, reference_excursion)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(53,), dtype=np.float32)
        self.command = np.array([TARGET_FORWARD_SPEED, 0.0, 0.0], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.phase = 0.0
        self.step_count = 0
        self.renderer: mujoco.Renderer | None = None
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.model.body("pelvis_link").id
        self.camera.distance = 2.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -12.0

    def _observe(self) -> np.ndarray:
        angular_velocity = self.data.sensor("imu_ang_vel").data.copy()
        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate((
            angular_velocity * 0.25,
            gravity_body,
            self.data.sensor("imu_lin_vel").data,
            rotation[:2, 0],
            [self.data.qpos[1]],
            self.command,
            self.data.qpos[self.qpos_ids],
            self.data.qvel[self.qvel_ids] * 0.1,
            self.last_action,
            [math.cos(2.0 * math.pi * self.phase), math.sin(2.0 * math.pi * self.phase)],
        )).astype(np.float32)
        if obs.shape != (53,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid policy observation")
        return obs

    def _feet_in_contact(self) -> np.ndarray:
        result = np.zeros(2, dtype=bool)
        for contact in self.data.contact:
            for foot_idx, foot_geom_id in enumerate(self.foot_geom_ids):
                if {contact.geom1, contact.geom2} == {self.floor_geom_id, foot_geom_id}:
                    result[foot_idx] = True
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, 0.75)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos_ids] = self.np_random.uniform(-0.01, 0.01, 12)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.last_action.fill(0.0)
        self.phase = 0.0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observe(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Action must contain 12 finite joint targets")
        target = np.clip(
            action * self.action_scale,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )
        previous_action = self.last_action.copy()
        for _ in range(DECIMATION):
            torque = KP * (target - self.data.qpos[self.qpos_ids]) - KD * self.data.qvel[self.qvel_ids]
            self.data.ctrl[self.ctrl_ids] = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.phase = (self.phase + POLICY_DT * GAIT_FREQUENCY) % 1.0
        self.last_action = action.astype(np.float32)

        velocity = self.data.sensor("imu_lin_vel").data
        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        velocity_world = rotation @ velocity
        angular_velocity = self.data.sensor("imu_ang_vel").data
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        forward_reward = math.exp(-((velocity_world[0] - TARGET_FORWARD_SPEED) ** 2 + velocity_world[1] ** 2) / 0.09)
        progress_cap = 1.0 if self.reward_stage == "progress" else TARGET_FORWARD_SPEED
        progress_reward = float(np.clip(velocity_world[0], -1.0, progress_cap))
        tracking_weight = 2.0 if self.reward_stage == "progress" else 3.0
        progress_weight = 4.0 if self.reward_stage == "progress" else 2.0
        turn_reward = math.exp(-(angular_velocity[2] ** 2) / 0.5)
        upright_reward = math.exp(-(gravity_body[0] ** 2 + gravity_body[1] ** 2) / 0.2)
        ref_index = int(self.phase * len(self.reference)) % len(self.reference)
        imitation_reward = math.exp(-np.mean((self.data.qpos[self.qpos_ids] - self.reference[ref_index]) ** 2) / 0.25)
        feet_in_contact = self._feet_in_contact()
        swing_idx = 0 if self.phase < 0.5 else 1
        alternating_reward = float(not feet_in_contact[swing_idx])
        action_rate_cost = float(np.mean((action - previous_action) ** 2))
        effort_cost = float(np.mean((self.data.ctrl[self.ctrl_ids] / EFFORT_LIMIT) ** 2))
        reward = (
            tracking_weight * forward_reward + progress_weight * progress_reward
            + 0.5 * turn_reward + upright_reward
            + 0.25 * imitation_reward + 0.25 * alternating_reward
            - 2.0 * abs(float(velocity_world[1]))
            - 0.5 * abs(float(self.data.qpos[1]))
            - 0.5 * (1.0 - float(rotation[0, 0]))
            - 0.1 * action_rate_cost - 0.02 * effort_cost
        )

        tilt = -gravity_body[2]
        fallen = bool(self.data.qpos[2] < 0.45 or tilt < math.cos(math.radians(70.0)))
        nonfinite = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        terminated = fallen or nonfinite
        truncated = self.step_count >= 1000
        if terminated:
            reward -= 5.0
        info = {
            "forward_speed": float(velocity_world[0]),
            "forward_distance": float(self.data.qpos[0]),
            "lateral_distance": float(self.data.qpos[1]),
            "fallen": fallen,
            "feet_in_contact": feet_in_contact.copy(),
            "reward_forward": float(forward_reward),
            "reward_imitation": float(imitation_reward),
        }
        return self._observe(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
