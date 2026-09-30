"""Full-body Asimov standing and slow walking in CPU MuJoCo."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "third_party" / "asimov-1" / "sim-model"
XML_PATH = MODEL_DIR / "xmls" / "asimov_1.xml"
URDF_PATH = MODEL_DIR / "urdf" / "asimov_1.urdf"
PHYSICS_DT = 0.005
POLICY_DT = 0.020
ACTION_SCALE = 0.25
EPISODE_STEPS = 500
STAND_HEIGHT = 0.61

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint",
)
MIRROR_JOINT_IDS = np.array([
    6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12,
    18, 19, 20, 21, 22, 13, 14, 15, 16, 17,
])

# The pinned Isaac Asimov actuator configuration, used as provisional hardware estimates.
# Values are (position stiffness, damping, torque limit, dry friction).
ACTUATOR_GROUPS = {
    "hip_pitch": (150.0, 5.0, 45.0, 0.70),
    "hip_roll": (150.0, 5.0, 45.0, 0.20),
    "hip_yaw": (150.0, 5.0, 28.0, 0.70),
    "knee": (150.0, 5.0, 45.0, 0.70),
    "ankle_pitch": (110.0, 5.0, 40.0, 0.40),
    "ankle_roll": (110.0, 5.0, 17.0, 0.40),
    "waist_yaw": (65.0, 5.0, 40.0, 0.70),
    "shoulder_pitch": (57.0, 5.0, 30.0, 0.20),
    "shoulder_roll": (86.0, 5.0, 25.0, 0.70),
    "shoulder_yaw": (96.0, 5.0, 20.0, 0.70),
    "elbow": (40.0, 2.0, 12.0, 0.40),
    "wrist_yaw": (40.0, 2.0, 12.0, 0.40),
}

STANDING_POSE = np.array([
    -0.15, 0.0, 0.0, 0.45, -0.30, 0.0,
    0.15, 0.0, 0.0, -0.45, 0.30, 0.0,
    0.0,
    0.25, 0.05, 0.0, -0.40, 0.0,
    -0.25, -0.05, 0.0, 0.40, 0.0,
], dtype=np.float64)


def _group(name: str) -> str:
    short = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return short


def _urdf_limits() -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(URDF_PATH).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    effort = np.empty(len(JOINT_NAMES))
    velocity = np.empty(len(JOINT_NAMES))
    for index, name in enumerate(JOINT_NAMES):
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"Missing URDF limit: {name}")
        effort[index] = float(limit.get("effort"))
        velocity[index] = float(limit.get("velocity"))
    return effort, velocity


class FullBodyEnv(gym.Env[np.ndarray, np.ndarray]):
    """23-action policy; only simulated, robot-available signals reach the actor."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, task: str = "stand", render_mode: str | None = None, randomize: bool = True):
        super().__init__()
        if task not in {"stand", "walk"}:
            raise ValueError("task must be 'stand' or 'walk'")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if not XML_PATH.is_file() or not URDF_PATH.is_file():
            raise FileNotFoundError("Initialize the pinned third_party/asimov-1 submodule")
        self.task = task
        self.render_mode = render_mode
        self.randomize = randomize

        spec = mujoco.MjSpec.from_file(str(XML_PATH))
        for name in JOINT_NAMES:
            actuator = spec.add_actuator(name=f"{name}_motor")
            actuator.set_to_motor()
            actuator.target = name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        self.model = spec.compile()
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.nj = len(JOINT_NAMES)
        joints = [self.model.joint(name) for name in JOINT_NAMES]
        self.qpos_ids = np.array([joint.qposadr[0] for joint in joints])
        self.qvel_ids = np.array([joint.dofadr[0] for joint in joints])
        self.ctrl_ids = np.array([self.model.actuator(f"{name}_motor").id for name in JOINT_NAMES])
        self.joint_ranges = np.stack([joint.range for joint in joints])
        self.kp = np.array([ACTUATOR_GROUPS[_group(name)][0] for name in JOINT_NAMES])
        self.kd = np.array([ACTUATOR_GROUPS[_group(name)][1] for name in JOINT_NAMES])
        self.friction = np.array([ACTUATOR_GROUPS[_group(name)][3] for name in JOINT_NAMES])
        self.effort, self.speed = _urdf_limits()
        configured_effort = np.array([ACTUATOR_GROUPS[_group(name)][2] for name in JOINT_NAMES])
        if not np.array_equal(self.effort, configured_effort):
            raise ValueError("URDF and actuator configuration disagree on effort limits")
        for dof_id, value in zip(self.qvel_ids, self.friction):
            self.model.dof_frictionloss[dof_id] = value
        self.base_mass = self.model.body_mass.copy()
        self.torso_id = self.model.body("waist_yaw_link").id
        self.imu_site_id = self.model.site("imu_in_pelvis").id
        self.foot_site_ids = [self.model.site(f"{side}_foot").id for side in ("left", "right")]
        self.floor_id = self.model.geom("floor").id
        self.foot_geom_ids = [
            {self.model.geom(f"{side}_foot{i}_collision").id for i in range(1, 5)}
            for side in ("left", "right")
        ]
        self.foot_friction = self.model.geom_friction.copy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nj,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(78,), dtype=np.float32)
        self.last_action = np.zeros(self.nj)
        self.previous_action = np.zeros(self.nj)
        self.target = STANDING_POSE.copy()
        self.action_buffer: deque[np.ndarray] = deque(maxlen=3)
        self.sensor_buffer: deque[np.ndarray] = deque(maxlen=3)
        self.action_lag = 0
        self.sensor_lag = 0
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.step_count = 0
        self.next_push = 150
        self.phase = 0.0
        self.initial_xy = np.zeros(2)
        self.contact_entries = np.zeros(2, dtype=int)
        self.last_contacts = np.zeros(2, dtype=bool)
        self.last_transition_step = np.zeros(2, dtype=int)
        self.renderer: mujoco.Renderer | None = None
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.model.body("pelvis_link").id
        self.camera.distance = 2.1
        self.camera.azimuth = 90.0
        self.camera.elevation = -10.0

    def _raw_sensors(self) -> np.ndarray:
        gyro = self.data.sensor("imu_ang_vel").data.copy()
        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate((gyro, gravity,
                               self.data.qpos[self.qpos_ids] - STANDING_POSE,
                               self.data.qvel[self.qvel_ids]))

    def _observe(self) -> np.ndarray:
        raw = self._raw_sensors()
        self.sensor_buffer.append(raw)
        sensed = self.sensor_buffer[-1 - self.sensor_lag].copy()
        if self.randomize:
            sensed[:3] += self.np_random.normal(0, 0.01, 3)
            sensed[3:6] += self.np_random.normal(0, 0.01, 3)
            sensed[6:6 + self.nj] += self.np_random.normal(0, 0.005, self.nj)
            sensed[6 + self.nj:] += self.np_random.normal(0, 0.05, self.nj)
        obs = np.concatenate((sensed[:3] * 0.25, sensed[3:6],
                              [0.2 if self.task == "walk" else 0.0],
                              sensed[6:6 + self.nj], sensed[6 + self.nj:] * 0.1,
                              self.last_action,
                              [math.sin(self.phase), math.cos(self.phase)]))
        if not np.isfinite(obs).all():
            raise RuntimeError("Nonfinite actor observation")
        return obs.astype(np.float32)

    def _contacts(self) -> tuple[np.ndarray, np.ndarray, int]:
        contacts = np.zeros(2, dtype=bool)
        forces = np.zeros(2)
        self_contacts = 0
        force6 = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            geoms = {contact.geom1, contact.geom2}
            if self.floor_id in geoms:
                for foot in range(2):
                    if geoms & self.foot_geom_ids[foot]:
                        contacts[foot] = True
                        mujoco.mj_contactForce(self.model, self.data, index, force6)
                        forces[foot] += abs(force6[0])
            elif self.model.geom_bodyid[contact.geom1] != self.model.geom_bodyid[contact.geom2]:
                self_contacts += 1
        return contacts, forces, self_contacts

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[:] = self.base_mass
        self.model.geom_friction[:] = self.foot_friction
        if self.randomize:
            self.model.body_mass[self.torso_id] *= self.np_random.uniform(0.95, 1.05)
            friction_scale = self.np_random.uniform(0.8, 1.2)
            for foot in self.foot_geom_ids:
                self.model.geom_friction[list(foot), 0] *= friction_scale
        mujoco.mj_setConst(self.model, self.data)
        self.kp_scale = self.np_random.uniform(0.8, 1.2) if self.randomize else 1.0
        self.kd_scale = self.np_random.uniform(0.8, 1.2) if self.randomize else 1.0
        self.action_lag = int(self.np_random.integers(0, 3)) if self.randomize else 0
        self.sensor_lag = int(self.np_random.integers(0, 3)) if self.randomize else 0
        self.data.qpos[:3] = (0.0, 0.0, 0.639)
        roll, pitch = self.np_random.uniform(-0.035, 0.035, 2) if self.randomize else (0.0, 0.0)
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([roll, pitch, 0.0]), "xyz")
        deviation = self.np_random.uniform(-0.02, 0.02, self.nj) if self.randomize else 0.0
        self.data.qpos[self.qpos_ids] = np.clip(STANDING_POSE + deviation,
                                                self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.target = STANDING_POSE.copy()
        self.last_action.fill(0.0)
        self.previous_action.fill(0.0)
        self.action_buffer.clear()
        self.sensor_buffer.clear()
        for _ in range(3):
            self.action_buffer.append(np.zeros(self.nj))
        self.step_count = 0
        self.next_push = int(self.np_random.integers(100, 200))
        self.phase = 0.0
        self.contact_entries.fill(0)
        self.last_transition_step.fill(0)
        mujoco.mj_forward(self.model, self.data)
        self.last_contacts, _, _ = self._contacts()
        self.initial_xy = self.data.qpos[:2].copy()
        for _ in range(2):
            self.sensor_buffer.append(self._raw_sensors())
        return self._observe(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.nj,) or not np.isfinite(action).all():
            raise ValueError("Expected 23 finite joint targets")
        action = np.clip(action, -1.0, 1.0)
        self.action_buffer.append(action.copy())
        applied = self.action_buffer[-1 - self.action_lag]
        requested = np.clip(STANDING_POSE + ACTION_SCALE * applied,
                            self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        max_change = self.speed * POLICY_DT
        self.target += np.clip(requested - self.target, -max_change, max_change)
        for _ in range(4):
            torque = (self.kp * self.kp_scale * (self.target - self.data.qpos[self.qpos_ids])
                      - self.kd * self.kd_scale * self.data.qvel[self.qvel_ids])
            self.data.ctrl[self.ctrl_ids] = np.clip(torque, -self.effort, self.effort)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if self.randomize and self.step_count == self.next_push:
            self.data.qvel[:2] += self.np_random.uniform(-0.1, 0.1, 2)
            self.next_push += int(self.np_random.integers(100, 200))
        if self.task == "walk":
            self.phase = (self.phase + 2.0 * math.pi * 1.0 * POLICY_DT) % (2.0 * math.pi)

        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        vel_world = self.data.qvel[:3]
        tilt_cost = gravity[0] ** 2 + gravity[1] ** 2
        height_cost = (self.data.qpos[2] - STAND_HEIGHT) ** 2
        contacts, forces, self_contacts = self._contacts()
        changed = contacts != self.last_contacts
        new_contacts = contacts & ~self.last_contacts
        rapid_contacts = int(np.sum(changed & (self.last_transition_step > 0)
                                    & (self.step_count - self.last_transition_step < 6)))
        self.last_transition_step[changed] = self.step_count
        self.contact_entries += new_contacts
        self.last_contacts = contacts
        foot_speed = np.zeros(2)
        for i, site_id in enumerate(self.foot_site_ids):
            velocity6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                     site_id, velocity6, 0)
            foot_speed[i] = np.linalg.norm(velocity6[3:5])
        slip = float(np.sum(contacts * foot_speed))
        foot_heights = self.data.site_xpos[self.foot_site_ids, 2]
        foot_clearance = float(np.sum((~contacts) * np.exp(-((foot_heights - 0.04) / 0.03) ** 2)))
        body_weight = float(np.sum(self.model.body_mass) * 9.81)
        impact = float(np.mean(np.maximum(forces / body_weight - 0.8, 0.0) ** 2))
        effort = float(np.mean((self.data.ctrl[self.ctrl_ids] / self.effort) ** 2))
        overspeed = float(np.mean(np.maximum(np.abs(self.data.qvel[self.qvel_ids]) / self.speed - 1, 0) ** 2))
        action_rate = float(np.mean((action - self.last_action) ** 2))
        action_accel = float(np.mean((action - 2 * self.last_action + self.previous_action) ** 2))
        upright = math.exp(-tilt_cost / 0.10)
        height = math.exp(-height_cost / 0.01)
        if self.task == "stand":
            still = math.exp(-float(np.dot(vel_world[:2], vel_world[:2])) / 0.01)
            reward = (2.0 * upright + 1.0 * height + 1.5 * still
                      + 0.25 * float(np.sum(contacts)) - 1.5 * slip
                      - 0.1 * action_rate - 0.03 * action_accel - 0.03 * effort
                      - 0.1 * overspeed - 0.5 * rapid_contacts
                      - 0.5 * float(np.linalg.norm(self.data.qpos[:2] - self.initial_xy)))
        else:
            tracking = math.exp(-((float(vel_world[0]) - 0.2) / 0.12) ** 2)
            progress = float(np.clip(vel_world[0], -0.5, 0.5))
            lateral = abs(float(vel_world[1]))
            yaw_rate = float(self.data.sensor("imu_ang_vel").data[2])
            flight = float(not contacts.any())
            reward = (3.0 * tracking + 0.5 * progress + 1.0 * upright + 0.5 * height
                      + 0.25 * foot_clearance - 5.0 * lateral - 8.0 * slip
                      - 0.3 * impact - 1.0 * flight
                      - 0.15 * action_rate - 0.05 * action_accel - 0.03 * effort
                      - 0.1 * overspeed - 3.0 * rapid_contacts
                      - 0.75 * float(np.sum(new_contacts)) - 0.1 * self_contacts
                      - 4.0 * abs(float(self.data.qpos[1] - self.initial_xy[1]))
                      - 0.7 * yaw_rate ** 2)
        self.previous_action = self.last_action.copy()
        self.last_action = action.copy()
        tilt = -float(gravity[2])
        fallen = bool(self.data.qpos[2] < 0.42 or tilt < math.cos(math.radians(50)))
        nonfinite = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        terminated = fallen or nonfinite
        if terminated:
            reward -= 10.0
        info = {
            "fallen": fallen,
            "forward_distance_m": float(self.data.qpos[0] - self.initial_xy[0]),
            "lateral_distance_m": float(self.data.qpos[1] - self.initial_xy[1]),
            "base_height_m": float(self.data.qpos[2]),
            "tilt_cos": tilt,
            "slip_m_s": slip,
            "foot_clearance": foot_clearance,
            "impact": impact,
            "effort": effort,
            "action_rate": action_rate,
            "action_accel": action_accel,
            "overspeed": overspeed,
            "rapid_contacts": rapid_contacts,
            "contact_entries": self.contact_entries.tolist(),
            "self_contacts": self_contacts,
            "reward_upright": upright,
            "reward_height": height,
            "reward_tracking": tracking if self.task == "walk" else still,
        }
        return self._observe(), float(reward), terminated, self.step_count >= EPISODE_STEPS, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class MirroredFullBodyEnv(gym.Wrapper):
    """Randomly reflect training episodes across the robot's sagittal plane."""

    def __init__(self, task: str):
        super().__init__(FullBodyEnv(task=task, randomize=True))
        self.mirror = False
        self.mirror_rng = np.random.default_rng()

    @staticmethod
    def mirror_action(action: np.ndarray) -> np.ndarray:
        return -np.asarray(action)[MIRROR_JOINT_IDS]

    @staticmethod
    def mirror_observation(obs: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(obs).copy()
        # Angular velocity is an axial vector; gravity is a polar vector.
        mirrored[0] *= -1
        mirrored[2] *= -1
        mirrored[4] *= -1
        for start in (7, 30, 53):
            mirrored[start:start + 23] = -obs[start + MIRROR_JOINT_IDS]
        return mirrored

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.mirror_rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.mirror = bool(self.mirror_rng.integers(0, 2))
        return (self.mirror_observation(obs) if self.mirror else obs), info

    def step(self, action: np.ndarray):
        physical_action = self.mirror_action(action) if self.mirror else action
        obs, reward, terminated, truncated, info = self.env.step(physical_action)
        return (self.mirror_observation(obs) if self.mirror else obs), reward, terminated, truncated, info
