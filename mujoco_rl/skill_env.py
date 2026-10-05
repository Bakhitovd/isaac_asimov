"""Flat-ground, 23-joint MuJoCo tasks for navigation and dynamic skills."""

from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .full_body_env import ACTION_SCALE, JOINT_NAMES, PHYSICS_DT, POLICY_DT, STAND_HEIGHT, STANDING_POSE, FullBodyEnv


SKILLS = ("nav", "squat", "recover", "run", "jump")
MAX_STEPS = {"nav": 1500, "squat": 750, "recover": 600, "run": 600, "jump": 400}
RECOVERY_POSES = ("front", "back", "left", "right")
JUMP_MIN_FLIGHT_SECONDS = 0.16
ENVIRONMENT_VERSION = 12  # Jump duration is measured in physics substeps, not policy steps.
RECOVERY_TILT_RANGES = ((0.0, 5.0), (3.0, 8.0), (5.0, 10.0), (8.0, 12.0),
                        (10.0, 15.0), (12.0, 20.0), (15.0, 21.0), (16.0, 22.0),
                        (17.0, 23.0), (18.0, 25.0), (20.0, 30.0), (25.0, 35.0),
                        (30.0, 45.0), (40.0, 60.0),
                        (55.0, 75.0), (70.0, 90.0))


def _yaw(quaternion: np.ndarray) -> float:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return math.atan2(float(matrix[3]), float(matrix[0]))


class SkillEnv(FullBodyEnv):
    """Keep the published walking observation as the first 78 channels."""

    def __init__(self, skill: str, render_mode: str | None = None, randomize: bool = True,
                 curriculum_level: float = 1.0, recovery_level: int | None = None,
                 recovery_rehearsal: bool = False, recovery_sensor_memory: bool = False,
                 recovery_phase_features: bool = False):
        if skill not in SKILLS and skill != "run_train":
            raise ValueError(f"Unknown skill: {skill}")
        super().__init__(task="walk", render_mode=render_mode, randomize=randomize,
                         support_contacts=True)
        self.training_skill = skill
        self.skill = "nav" if skill == "run_train" else skill
        self.curriculum_level = float(np.clip(curriculum_level, 0.0, 1.0))
        self.recovery_level = recovery_level
        self.recovery_rehearsal = recovery_rehearsal
        self.recovery_sensor_memory = recovery_sensor_memory
        self.recovery_phase_features = recovery_phase_features
        self.recovery_initial_gravity = np.zeros(2, dtype=np.float32)
        self.episode_recovery_level = recovery_level
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(83,), dtype=np.float32)
        self.command = np.zeros(2)
        self.goal_world = np.zeros(2)
        self.goal_body = np.zeros(2)
        self.nav_max_speed = 0.30
        self.run_training_speed = 0.60
        self.start_com_z = 0.0
        self.apex_rise = 0.0
        self.max_flight_steps = 0
        self.flight_steps = 0
        self.flight_events = 0
        self.run_flights = 0
        self.run_speeds: list[float] = []
        self.stable_steps = 0
        self.low_holds = np.zeros(3, dtype=int)
        self.high_holds = np.zeros(3, dtype=int)
        self.recovery_pose = "front"
        self.recovery_pose_weights = np.full(4, 0.25)
        self.recovery_focus_ranges: dict[str, tuple[float, float]] = {}
        self.jump_forward = False
        self.success = False
        self.initial_com_xy = np.zeros(2)
        self.initial_heading = 0.0
        self.total_slip = 0.0
        self.self_contact_steps = 0
        self.squat_air_steps = 0
        self.squat_single_support_steps = 0
        self.recovery_start_contacts = 0
        self.recovery_tilt_deg = 90.0
        self.max_base_height = 0.0
        self.max_upright_cos = -1.0

    def set_curriculum_level(self, level: float) -> None:
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    def set_run_training_speed(self, speed: float) -> None:
        if not 0.3 <= speed <= 0.6:
            raise ValueError("Run curriculum speed must be in [0.3, 0.6]")
        self.run_training_speed = speed

    def set_navigation_speed(self, speed: float) -> None:
        """Change gait command scale while preserving queued physical targets."""
        targets = [self._target_from_action(action) for action in self.action_buffer]
        last_target = self._target_from_action(self.last_action)
        previous_target = self._target_from_action(self.previous_action)
        self.nav_max_speed = speed
        self._goal_and_command()
        self.action_buffer.clear()
        self.action_buffer.extend(self.action_from_target(target) for target in targets)
        self.last_action = self.action_from_target(last_target)
        self.previous_action = self.action_from_target(previous_target)

    def set_recovery_level(self, level: int) -> None:
        if level not in range(len(RECOVERY_TILT_RANGES)):
            raise ValueError("Invalid recovery curriculum level")
        self.recovery_level = level
        self.recovery_focus_ranges.clear()

    def set_recovery_sensor_memory(self, enabled: bool) -> None:
        self.recovery_sensor_memory = bool(enabled)

    def set_recovery_phase_features(self, enabled: bool) -> None:
        self.recovery_phase_features = bool(enabled)

    def set_recovery_pose_weights(self, weights: list[float]) -> None:
        probabilities = np.asarray(weights, dtype=float)
        if probabilities.shape != (4,) or not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError("Expected four positive recovery pose weights")
        self.recovery_pose_weights = probabilities / probabilities.sum()

    def set_recovery_focus_ranges(self, ranges: dict[str, tuple[float, float]]) -> None:
        if any(pose not in RECOVERY_POSES or len(bounds) != 2 or
               not 0.0 <= bounds[0] < bounds[1] <= 90.0 for pose, bounds in ranges.items()):
            raise ValueError("Invalid recovery focus ranges")
        self.recovery_focus_ranges = dict(ranges)

    def _goal_and_command(self) -> None:
        yaw = _yaw(self.data.qpos[3:7])
        delta = self.goal_world - self.data.qpos[:2]
        self.goal_body[:] = (math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
                             -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1])
        distance = float(np.linalg.norm(delta))
        angle = math.atan2(self.goal_body[1], self.goal_body[0])
        if distance < 0.25:
            self.command[:] = 0.0
        else:
            self.command[0] = min(self.nav_max_speed, 0.6 * distance) * max(0.0, math.cos(angle))
            self.command[1] = float(np.clip(1.5 * angle, -0.6, 0.6))

    def _observe(self, advance_sensors: bool = True) -> np.ndarray:
        base = super()._observe(advance_sensors)
        base[6] = self.command[0]
        velocity = self.data.sensor("imu_lin_vel").data[:2].copy()
        if self.randomize:
            velocity += self.np_random.normal(0.0, 0.03 * self.randomization_strength, 2)
        observation = np.concatenate((base, [self.command[1]], velocity * 0.5,
                                      np.clip(self.goal_body, -3.0, 3.0) * 0.5)).astype(np.float32)
        if self.skill == "recover" and self.recovery_sensor_memory:
            observation[81:83] = self.recovery_initial_gravity
        if self.skill == "recover" and self.recovery_phase_features:
            phase = 0 if self.step_count < 5 else 1 if self.step_count < 15 else 2 if self.step_count < 40 else 3
            observation[6], observation[78] = phase & 1, phase >> 1
        return observation

    def _set_recovery_pose(self, pose: str, tilt_deg: float) -> None:
        angle = math.radians(tilt_deg)
        # Mild curriculum poses lean the torso while keeping the legs nearer
        # vertical. Fade that support out completely before lying poses.
        compensation = angle * float(np.clip((45.0 - tilt_deg) / 30.0, 0.0, 1.0))
        if pose in {"front", "back"}:
            signed = compensation if pose == "front" else -compensation
            self.data.qpos[self.qpos_ids[[0, 6]]] += [-signed, signed]
        else:
            signed = compensation if pose == "left" else -compensation
            self.data.qpos[self.qpos_ids[[1, 7]]] -= signed
        self.data.qpos[self.qpos_ids] = np.clip(self.data.qpos[self.qpos_ids],
                                                self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        angles = {
            "front": (0.0, angle, 0.0),
            "back": (0.0, -angle, 0.0),
            "left": (angle, 0.0, 0.0),
            "right": (-angle, 0.0, 0.0),
        }[pose]
        self.data.qpos[2] = 0.70
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.asarray(angles), "xyz")
        self.data.qvel[:] = 0.0
        for _ in range(220):
            mujoco.mj_forward(self.model, self.data)
            floor_contacts = [contact for contact in self.data.contact
                              if self.floor_id in (contact.geom1, contact.geom2)]
            if floor_contacts:
                penetration = max(-float(contact.dist) for contact in floor_contacts)
                self.data.qpos[2] += max(0.0, penetration - 0.001)
                break
            self.data.qpos[2] -= 0.005
        else:
            raise RuntimeError(f"Could not place {pose} recovery pose on floor")
        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.recovery_start_contacts = sum(self.floor_id in (contact.geom1, contact.geom2)
                                           for contact in self.data.contact)
        if not self.recovery_start_contacts:
            raise RuntimeError(f"{pose} recovery pose has no floor contact")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.recovery_initial_gravity.fill(0.0)
        self.command = np.zeros(2)
        self.goal_body = np.zeros(2)
        self.goal_world = np.zeros(2)
        self.nav_max_speed = 0.20 + 0.10 * self.curriculum_level if self.training_skill == "nav" else 0.30
        super().reset(seed=seed, options=options)
        self.skill = ("nav" if self.np_random.random() < 0.3 else "run") if self.training_skill == "run_train" else self.training_skill
        self.stable_steps = 0
        self.low_holds.fill(0)
        self.high_holds.fill(0)
        self.apex_rise = 0.0
        self.max_flight_steps = 0
        self.flight_steps = 0
        self.flight_events = 0
        self.run_flights = 0
        self.run_speeds = []
        self.success = False
        self.total_slip = 0.0
        self.self_contact_steps = 0
        self.squat_air_steps = 0
        self.squat_single_support_steps = 0
        self.recovery_start_contacts = 0
        self.recovery_reward_totals = {}
        if self.skill == "nav":
            distance = self.np_random.uniform(0.5, 1.5 + self.curriculum_level)
            angle_limit = 0.35 + (math.pi - 0.35) * self.curriculum_level
            angle = self.np_random.uniform(-angle_limit, angle_limit)
            self.goal_world = self.data.qpos[:2] + distance * np.array([math.cos(angle), math.sin(angle)])
            self._goal_and_command()
        elif self.skill == "recover":
            self.recovery_pose = RECOVERY_POSES[(seed % 4) if seed is not None else
                                              int(self.np_random.choice(4, p=self.recovery_pose_weights))]
            self.episode_recovery_level = self.recovery_level
            if self.recovery_rehearsal and self.recovery_level and self.np_random.random() < 0.30:
                self.episode_recovery_level = (self.recovery_level - 1 if self.np_random.random() < 0.60
                                               else int(self.np_random.integers(self.recovery_level)))
            tilt_range = ((90.0, 90.0) if self.episode_recovery_level is None
                          else RECOVERY_TILT_RANGES[self.episode_recovery_level])
            if (self.recovery_rehearsal and self.episode_recovery_level == self.recovery_level
                    and self.recovery_pose in self.recovery_focus_ranges and self.np_random.random() < 0.80):
                tilt_range = self.recovery_focus_ranges[self.recovery_pose]
            self.recovery_tilt_deg = float(self.np_random.uniform(*tilt_range))
            if options and "recovery_pose" in options:
                if options["recovery_pose"] not in RECOVERY_POSES:
                    raise ValueError("Invalid recovery pose override")
                self.recovery_pose = options["recovery_pose"]
            if options and "recovery_tilt_deg" in options:
                self.recovery_tilt_deg = float(options["recovery_tilt_deg"])
                if not 0.0 <= self.recovery_tilt_deg <= 90.0:
                    raise ValueError("Invalid recovery tilt override")
            self._set_recovery_pose(self.recovery_pose, self.recovery_tilt_deg)
            # Motor targets and delayed commands must match the initial joint
            # pose. Otherwise a lagged first action drives a stale standing
            # command before the policy has a chance to react.
            self.target = self.data.qpos[self.qpos_ids].copy()
            lower, upper = -np.ones(self.nj), np.ones(self.nj)
            for _ in range(32):
                holding = 0.5 * (lower + upper)
                below = self._target_from_action(holding) < self.target
                lower = np.where(below, holding, lower)
                upper = np.where(below, upper, holding)
            holding = 0.5 * (lower + upper)
            self.action_buffer.clear()
            for _ in range(3):
                self.action_buffer.append(holding.copy())
            self.last_action = holding.copy()
            self.previous_action = holding.copy()
        elif self.skill == "jump":
            self.jump_forward = (seed % 2 == 1) if seed is not None else bool(self.np_random.integers(2))
            self.goal_body[0] = 0.2 if self.jump_forward else 0.0
        self.initial_xy = self.data.qpos[:2].copy()
        self.initial_heading = _yaw(self.data.qpos[3:7])
        self.start_com_z = float(self.data.subtree_com[1, 2])
        self.max_base_height = float(self.data.qpos[2])
        self.max_upright_cos = float(self.data.site_xmat[self.imu_site_id].reshape(3, 3)[2, 2])
        self.initial_com_xy = self.data.subtree_com[1, :2].copy()
        self.last_contacts, _, _ = self._contacts()
        self.sensor_buffer.clear()
        for _ in range(3):
            self.sensor_buffer.append(self._raw_sensors())
        observation = self._observe()
        if self.skill == "recover" and self.recovery_sensor_memory:
            self.recovery_initial_gravity[:] = observation[3:5]
            observation[81:83] = self.recovery_initial_gravity
        return observation, {"skill": self.skill}

    def switch_skill(self, skill: str, goal_world: np.ndarray | None = None,
                     nav_max_speed: float = 0.30) -> None:
        """Change the active skill without resetting the physical robot state."""
        if skill not in SKILLS:
            raise ValueError(f"Unknown skill: {skill}")
        queued_targets = [self._target_from_action(action) for action in self.action_buffer]
        last_target = self._target_from_action(self.last_action)
        previous_target = self._target_from_action(self.previous_action)
        self.skill = skill
        self.step_count = 0
        self.success = False
        self.stable_steps = 0
        self.low_holds.fill(0)
        self.high_holds.fill(0)
        self.flight_steps = 0
        self.flight_events = 0
        self.run_flights = 0
        self.run_speeds.clear()
        self.apex_rise = 0.0
        self.max_flight_steps = 0
        self.total_slip = 0.0
        self.self_contact_steps = 0
        self.recovery_reward_totals = {}
        self.squat_air_steps = 0
        self.squat_single_support_steps = 0
        self.max_base_height = float(self.data.qpos[2])
        self.max_upright_cos = float(self.data.site_xmat[self.imu_site_id].reshape(3, 3)[2, 2])
        self.initial_xy = self.data.qpos[:2].copy()
        self.initial_heading = _yaw(self.data.qpos[3:7])
        self.start_com_z = float(self.data.subtree_com[1, 2])
        self.initial_com_xy = self.data.subtree_com[1, :2].copy()
        self.nav_max_speed = nav_max_speed
        if goal_world is not None:
            self.goal_world = np.asarray(goal_world, dtype=float).copy()
        if skill == "nav":
            self._goal_and_command()
        else:
            self.command[:] = 0.0
            self.phase = 0.0
        if skill == "recover" and self.recovery_sensor_memory:
            self.recovery_initial_gravity[:] = self._observe(advance_sensors=False)[3:5]
        if skill == "recover":
            # Runtime falls must get the full recovery horizon, regardless of
            # which curriculum the policy was trained with.
            self.episode_recovery_level = None
        self.action_buffer.clear()
        self.action_buffer.extend(self.action_from_target(target) for target in queued_targets)
        self.last_action = self.action_from_target(last_target)
        self.previous_action = self.action_from_target(previous_target)

    def action_from_target(self, target: np.ndarray) -> np.ndarray:
        """Invert the current command mapping, clipping unreachable targets."""
        target = np.asarray(target, dtype=float)
        if target.shape != (self.nj,) or not np.isfinite(target).all():
            raise ValueError("Expected 23 finite joint targets")
        lower, upper = -np.ones(self.nj), np.ones(self.nj)
        for _ in range(40):
            middle = (lower + upper) * 0.5
            below = self._target_from_action(middle) < target
            lower, upper = np.where(below, middle, lower), np.where(below, upper, middle)
        return ((lower + upper) * 0.5).astype(np.float32)

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        if self.skill == "recover":
            extent = np.where(action >= 0.0, self.joint_ranges[:, 1] - STANDING_POSE,
                              STANDING_POSE - self.joint_ranges[:, 0])
            local_scale = np.full(self.nj, 0.25)
            extra = np.maximum(0.0, (np.abs(action) - 0.5) * 2.0) ** 2
            delta = local_scale * action + np.sign(action) * extra * (extent - local_scale)
            return np.clip(STANDING_POSE + delta, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        scale = np.full(self.nj, ACTION_SCALE)
        if self.skill in {"squat", "jump"}:
            scale[:12] = 0.70
            scale[12:] = 0.45
        elif self.skill == "run" or (self.skill == "nav" and self.command[0] > 0.30):
            scale[:12] = 0.35
        return np.clip(STANDING_POSE + scale * action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _run_command(self) -> float:
        speed = self.run_training_speed if self.training_skill == "run_train" else 0.6
        return min(speed, 0.2 + 0.004 * self.step_count) if self.step_count < 100 else (
            speed if self.step_count < 500 else 0.0)

    def _run_flight_reward(self) -> float:
        return 0.2 * float(self.flight_steps >= 4)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.nj,) or not np.isfinite(action).all():
            raise ValueError("Expected 23 finite joint commands")
        action = np.clip(action, -1.0, 1.0)
        old_distance = float(np.linalg.norm(self.goal_world - self.data.qpos[:2]))
        old_bearing = abs(math.atan2(self.goal_body[1], self.goal_body[0]))
        old_upright_cos = float(self.data.site_xmat[self.imu_site_id].reshape(3, 3)[2, 2])
        old_com_z = float(self.data.subtree_com[1, 2])
        if self.skill == "nav":
            self._goal_and_command()
        elif self.skill == "run":
            self.command[:] = (self._run_command(), 0.0)
        else:
            self.command[:] = 0.0
        self.action_buffer.append(action.copy())
        applied = self.action_buffer[-1 - self.action_lag]
        requested = self._target_from_action(applied)
        old_target = self.target.copy()
        old_joint_velocity = self.data.qvel[self.qvel_ids].copy()
        self.target += np.clip(requested - self.target, -self.speed * POLICY_DT, self.speed * POLICY_DT)
        for _ in range(4):
            torque = (self.kp * self.kp_scale * (self.target - self.data.qpos[self.qpos_ids])
                      - self.kd * self.kd_scale * self.data.qvel[self.qvel_ids])
            self.data.ctrl[self.ctrl_ids] = np.clip(torque, -self.effort, self.effort)
            mujoco.mj_step(self.model, self.data)
            if self.skill in {"run", "jump"}:
                sub_contacts, _, _ = self._contacts()
                if not sub_contacts.any():
                    self.flight_steps += 1
                    self.max_flight_steps = max(self.max_flight_steps, self.flight_steps)
                elif self.flight_steps:
                    if self.flight_steps >= 4:
                        self.flight_events += 1
                        if self.skill == "run" and 100 <= self.step_count < 500:
                            self.run_flights += 1
                    self.flight_steps = 0
        self.step_count += 1
        if self.randomize and self.step_count == self.next_push and self.skill not in {"recover", "jump"}:
            self.data.qvel[:2] += self.np_random.uniform(-0.1, 0.1, 2) * self.randomization_strength
            self.next_push += int(self.np_random.integers(100, 200))
        if self.skill in {"nav", "run"}:
            frequency = 1.0 + max(0.0, self.command[0] - 0.2) * 1.5
            self.phase = (self.phase + 2 * math.pi * frequency * POLICY_DT) % (2 * math.pi)
        else:
            self.phase = 2 * math.pi * (self.step_count % 250) / 250.0
        if self.skill == "nav":
            self._goal_and_command()
        contacts, forces, self_contacts = self._contacts()
        new_contacts = contacts & ~self.last_contacts
        self.contact_entries += new_contacts
        self.last_contacts = contacts
        foot_speed = np.zeros(2)
        for i, site_id in enumerate(self.foot_site_ids):
            velocity6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity6, 0)
            foot_speed[i] = np.linalg.norm(velocity6[3:5])
        slip = float(np.sum(contacts * foot_speed))
        self.total_slip += slip
        self.self_contact_steps += int(self_contacts > 0)
        if self.skill == "squat" and self.step_count > 10:
            self.squat_air_steps += int(not contacts.any())
            self.squat_single_support_steps += int(contacts.sum() == 1)
        gyro = self.data.sensor("imu_ang_vel").data
        body_velocity = self.data.sensor("imu_lin_vel").data
        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        tilt = -float(gravity[2])
        height = float(self.data.qpos[2])
        self.max_base_height = max(self.max_base_height, height)
        self.max_upright_cos = max(self.max_upright_cos, tilt)
        upright = math.exp(-(gravity[0] ** 2 + gravity[1] ** 2) / 0.10)
        effort = float(np.mean((self.data.ctrl[self.ctrl_ids] / self.effort) ** 2))
        action_rate = float(np.mean((action - self.last_action) ** 2))
        action_accel = float(np.mean((action - 2 * self.last_action + self.previous_action) ** 2))
        impact = float(np.max(forces) / max(1.0, np.sum(self.model.body_mass) * 9.81))
        self.previous_action = self.last_action.copy()
        self.last_action = action.copy()
        fallen = height < 0.42 or tilt < math.cos(math.radians(50))
        nonfinite = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        drift = float(np.linalg.norm(self.data.qpos[:2] - self.initial_xy))
        still = float(np.linalg.norm(self.data.qvel[:2])) < 0.1
        reward = 0.0
        recovery_components = {}
        if self.skill == "nav":
            distance = float(np.linalg.norm(self.goal_world - self.data.qpos[:2]))
            near = distance < 0.25
            self.stable_steps = self.stable_steps + 1 if near and still and tilt > 0.96 else 0
            self.success = (self.stable_steps >= 100 and self.total_slip / self.step_count < 0.05
                            and self.self_contact_steps == 0)
            tracking = math.exp(-((float(body_velocity[0]) - self.command[0]) / 0.18) ** 2)
            turning = math.exp(-((float(gyro[2]) - self.command[1]) / 0.45) ** 2)
            progress = float(np.clip((old_distance - distance) / POLICY_DT, -0.5, 0.5))
            bearing = abs(math.atan2(self.goal_body[1], self.goal_body[0]))
            heading_progress = float(np.clip((old_bearing - bearing) / POLICY_DT, -0.6, 0.6))
            waiting = float(not near and abs(progress) < 0.02 and abs(float(gyro[2])) < 0.05)
            reward = (5.0 * progress + 2.0 * heading_progress + tracking
                      + turning + upright + 2.0 * float(near)
                      - 0.1 * distance - 0.3 * waiting - 4.0 * slip
                      - 0.2 * abs(float(body_velocity[1])))
            if self.success:
                reward += 100.0
        elif self.skill == "squat":
            within_cycle = self.step_count % 250
            cycle = min(self.step_count // 250, 2)
            profile = min(1.0, within_cycle / 50.0) if within_cycle < 100 else max(0.0, 1.0 - (within_cycle - 100) / 50.0)
            target_height = STAND_HEIGHT - 0.10 * profile
            height_error = abs(height - target_height)
            if 50 <= within_cycle < 100 and height_error < 0.03:
                self.low_holds[cycle] += 1
            if 150 <= within_cycle < 250 and height_error < 0.03:
                self.high_holds[cycle] += 1
            self.success = (self.step_count >= 750 and np.all(self.low_holds >= 40)
                            and np.all(self.high_holds >= 50) and drift < 0.15 and not fallen
                            and contacts.all() and self.squat_air_steps <= 5
                            and self.squat_single_support_steps <= 50
                            and self.self_contact_steps == 0)
            reward = (3.0 * math.exp(-(height_error / 0.05) ** 2) + upright
                      + 0.4 * float(np.sum(contacts)) - 3.0 * drift - 3.0 * slip)
        elif self.skill == "recover":
            stable = height > 0.55 and tilt > math.cos(math.radians(15)) and still and contacts.all()
            self.stable_steps = self.stable_steps + 1 if stable else 0
            self.success = self.stable_steps >= 100
            supported_height = float(np.clip((height - 0.10) / 0.50, 0.0, 1.0))
            standing_quality = supported_height * math.exp(-max(0.0, 1.0 - tilt) / 0.01)
            reward = (standing_quality + 8.0 * (tilt - old_upright_cos)
                      + 8.0 * (float(self.data.subtree_com[1, 2]) - old_com_z)
                      - 0.02 + 0.5 * float(stable))
            recovery_components = {
                "standing": standing_quality, "upright_progress": 8.0 * (tilt - old_upright_cos),
                "height_progress": 8.0 * (float(self.data.subtree_com[1, 2]) - old_com_z),
                "time": -0.02, "stable": 0.5 * float(stable), "completion": 500.0 if self.success else 0.0,
            }
            if self.success:
                # gamma=0.995: the maximum discounted standing reward is
                # 1.5 / (1 - gamma) = 300. Completion must beat delaying it.
                reward += 500.0
        elif self.skill == "run":
            if 100 <= self.step_count < 500:
                self.run_speeds.append(float(body_velocity[0]))
            tracking = math.exp(-((float(body_velocity[0]) - self.command[0]) / 0.20) ** 2)
            self.success = (self.step_count >= 600 and len(self.run_speeds) >= 350
                            and 0.5 <= np.mean(self.run_speeds) <= 0.7
                            and self.run_flights >= 5 and height > 0.55 and tilt > 0.96 and still
                            and contacts.all() and abs(float(self.data.qpos[1] - self.initial_xy[1])) < 0.5
                            and self.total_slip / self.step_count < 0.05 and self.self_contact_steps == 0)
            reward = (3.0 * tracking + upright + self._run_flight_reward()
                      - 4.0 * slip - 1.5 * abs(float(body_velocity[1])))
        else:
            com = self.data.subtree_com[1]
            self.apex_rise = max(self.apex_rise, float(com[2]) - self.start_com_z)
            forward = (com[0] - self.initial_com_xy[0]) * math.cos(self.initial_heading) + (com[1] - self.initial_com_xy[1]) * math.sin(self.initial_heading)
            landed = self.step_count > 100 and contacts.all() and height > 0.52 and tilt > 0.96 and still
            self.stable_steps = self.stable_steps + 1 if landed else 0
            lateral = -(com[0] - self.initial_com_xy[0]) * math.sin(self.initial_heading) + (com[1] - self.initial_com_xy[1]) * math.cos(self.initial_heading)
            travel_ok = (forward >= 0.20 and abs(lateral) < 0.15 if self.jump_forward
                         else np.linalg.norm(com[:2] - self.initial_com_xy) < 0.15)
            self.success = (self.apex_rise >= 0.05 and self.max_flight_steps * PHYSICS_DT >= JUMP_MIN_FLIGHT_SECONDS
                            and travel_ok and self.stable_steps >= 100 and self.self_contact_steps == 0)
            reward = (2.0 * upright + 2.0 * min(self.apex_rise / 0.05, 1.0)
                      + 0.5 * float(self.max_flight_steps * PHYSICS_DT >= JUMP_MIN_FLIGHT_SECONDS)
                      + 0.02 * self.stable_steps)
            if self.jump_forward:
                reward += min(max(forward, 0.0) / 0.20, 1.0)
            else:
                reward -= 2.0 * abs(forward)
            if self.success:
                reward += 30.0
        reward -= 0.15 * action_rate + 0.05 * action_accel + 0.03 * effort
        if self.skill != "recover" and (fallen or nonfinite):
            reward -= 100.0
        recovery_failed = (self.skill == "recover" and self.episode_recovery_level is not None
                           and RECOVERY_TILT_RANGES[self.episode_recovery_level][1] <= 25.0
                           and (height < 0.30 or tilt < 0.5))
        if recovery_failed:
            reward -= 20.0
        terminated = bool(self.success or nonfinite or recovery_failed or (fallen and self.skill != "recover"))
        truncated = self.step_count >= MAX_STEPS[self.skill]
        if self.skill == "recover":
            recovery_components.update(action_rate=-0.15 * action_rate, action_accel=-0.05 * action_accel,
                                       effort=-0.03 * effort, failure=-20.0 if recovery_failed else 0.0)
            for key, value in recovery_components.items():
                self.recovery_reward_totals[key] = self.recovery_reward_totals.get(key, 0.0) + value
        info = {
            "skill": self.skill, "success": bool(self.success), "fallen": bool(fallen),
            "seconds": self.step_count * POLICY_DT, "distance_to_goal_m": float(np.linalg.norm(self.goal_world - self.data.qpos[:2])) if self.skill == "nav" else None,
            "drift_m": drift, "base_height_m": height, "slip_m_s": slip, "action_rate": action_rate,
            "effort": effort, "impact_weight": impact, "self_contacts": self_contacts,
            "action_accel": action_accel,
            "target_velocity_rms": float(np.sqrt(np.mean(((self.target - old_target) / POLICY_DT) ** 2))),
            "joint_acceleration_rms": float(np.sqrt(np.mean(
                ((self.data.qvel[self.qvel_ids] - old_joint_velocity) / POLICY_DT) ** 2))),
            "torque_saturation_fraction": float(np.mean(np.abs(self.data.ctrl[self.ctrl_ids]) >= self.effort * 0.99)),
            "self_contact_steps": self.self_contact_steps, "squat_air_steps": self.squat_air_steps,
            "squat_single_support_steps": self.squat_single_support_steps,
            "flight_events": self.run_flights if self.skill == "run" else self.flight_events,
            "max_flight_s": self.max_flight_steps * 0.005, "apex_rise_m": self.apex_rise,
            "mean_run_speed_m_s": float(np.mean(self.run_speeds)) if self.run_speeds else None,
            "low_holds": self.low_holds.tolist(), "high_holds": self.high_holds.tolist(),
            "recovery_pose": self.recovery_pose if self.skill == "recover" else None,
            "recovery_tilt_deg": self.recovery_tilt_deg if self.skill == "recover" else None,
            "episode_recovery_level": self.episode_recovery_level if self.skill == "recover" else None,
            "recovery_start_contacts": self.recovery_start_contacts if self.skill == "recover" else None,
            "max_base_height_m": self.max_base_height,
            "max_upright_cos": self.max_upright_cos,
            "stable_steps": self.stable_steps,
            "jump_variant": "forward" if self.jump_forward else "up" if self.skill == "jump" else None,
        }
        info["termination_reason"] = ("nonfinite" if nonfinite else "success" if self.success else
                                      "recovery_fall" if recovery_failed else "fall" if terminated else
                                      "time_limit" if truncated else None)
        if self.skill == "recover" and (terminated or truncated):
            info["recovery_reward_totals"] = dict(self.recovery_reward_totals)
        return self._observe(), float(reward), terminated, truncated, info
