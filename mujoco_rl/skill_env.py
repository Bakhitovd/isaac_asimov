"""Flat-ground, 23-joint MuJoCo tasks for navigation and dynamic skills."""

from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .full_body_env import ACTION_SCALE, JOINT_NAMES, POLICY_DT, STAND_HEIGHT, STANDING_POSE, FullBodyEnv


SKILLS = ("nav", "squat", "recover", "run", "jump")
MAX_STEPS = {"nav": 1500, "squat": 750, "recover": 600, "run": 600, "jump": 400}
RECOVERY_POSES = ("front", "back", "left", "right")


def _yaw(quaternion: np.ndarray) -> float:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return math.atan2(float(matrix[3]), float(matrix[0]))


class SkillEnv(FullBodyEnv):
    """Keep the published walking observation as the first 78 channels."""

    def __init__(self, skill: str, render_mode: str | None = None, randomize: bool = True):
        if skill not in SKILLS and skill != "run_train":
            raise ValueError(f"Unknown skill: {skill}")
        super().__init__(task="walk", render_mode=render_mode, randomize=randomize,
                         support_contacts=True)
        self.training_skill = skill
        self.skill = "nav" if skill == "run_train" else skill
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(83,), dtype=np.float32)
        self.command = np.zeros(2)
        self.goal_world = np.zeros(2)
        self.goal_body = np.zeros(2)
        self.nav_max_speed = 0.30
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
        self.jump_forward = False
        self.success = False
        self.initial_com_xy = np.zeros(2)
        self.initial_heading = 0.0
        self.total_slip = 0.0
        self.self_contact_steps = 0
        self.squat_air_steps = 0

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

    def _observe(self) -> np.ndarray:
        base = super()._observe()
        base[6] = self.command[0]
        velocity = self.data.sensor("imu_lin_vel").data[:2].copy()
        if self.randomize:
            velocity += self.np_random.normal(0.0, 0.03, 2)
        return np.concatenate((base, [self.command[1]], velocity * 0.5,
                               np.clip(self.goal_body, -3.0, 3.0) * 0.5)).astype(np.float32)

    def _set_recovery_pose(self, pose: str) -> None:
        angles = {
            "front": (0.0, math.pi / 2, 0.0),
            "back": (0.0, -math.pi / 2, 0.0),
            "left": (math.pi / 2, 0.0, 0.0),
            "right": (-math.pi / 2, 0.0, 0.0),
        }[pose]
        self.data.qpos[2] = 0.25
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.asarray(angles), "xyz")
        self.data.qvel[:] = 0.0
        for _ in range(8):
            mujoco.mj_forward(self.model, self.data)
            penetration = max((-float(contact.dist) for contact in self.data.contact
                               if self.floor_id in (contact.geom1, contact.geom2)), default=0.0)
            if penetration < 0.002:
                break
            self.data.qpos[2] += penetration + 0.003
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.command = np.zeros(2)
        self.goal_body = np.zeros(2)
        self.goal_world = np.zeros(2)
        self.nav_max_speed = 0.30
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
        if self.skill == "nav":
            distance = self.np_random.uniform(0.5, 2.5)
            angle = self.np_random.uniform(-math.pi, math.pi)
            self.goal_world = self.data.qpos[:2] + distance * np.array([math.cos(angle), math.sin(angle)])
            self._goal_and_command()
        elif self.skill == "recover":
            self.recovery_pose = RECOVERY_POSES[(seed % 4) if seed is not None else int(self.np_random.integers(4))]
            self._set_recovery_pose(self.recovery_pose)
        elif self.skill == "jump":
            self.jump_forward = (seed % 2 == 1) if seed is not None else bool(self.np_random.integers(2))
            self.goal_body[0] = 0.2 if self.jump_forward else 0.0
        self.initial_xy = self.data.qpos[:2].copy()
        self.initial_heading = _yaw(self.data.qpos[3:7])
        self.start_com_z = float(self.data.subtree_com[1, 2])
        self.initial_com_xy = self.data.subtree_com[1, :2].copy()
        self.last_contacts, _, _ = self._contacts()
        self.sensor_buffer.clear()
        for _ in range(3):
            self.sensor_buffer.append(self._raw_sensors())
        return self._observe(), {"skill": self.skill}

    def switch_skill(self, skill: str, goal_world: np.ndarray | None = None,
                     nav_max_speed: float = 0.30) -> None:
        """Change the active skill without resetting the physical robot state."""
        if skill not in SKILLS:
            raise ValueError(f"Unknown skill: {skill}")
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
        self.squat_air_steps = 0
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

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        if self.skill == "recover":
            return self.joint_ranges[:, 0] + (action + 1.0) * 0.5 * np.diff(self.joint_ranges, axis=1)[:, 0]
        scale = np.full(self.nj, ACTION_SCALE)
        if self.skill in {"squat", "jump"}:
            scale[:12] = 0.70
            scale[12:] = 0.45
        elif self.skill == "run" or (self.skill == "nav" and self.command[0] > 0.30):
            scale[:12] = 0.35
        return np.clip(STANDING_POSE + scale * action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.nj,) or not np.isfinite(action).all():
            raise ValueError("Expected 23 finite joint commands")
        action = np.clip(action, -1.0, 1.0)
        old_distance = float(np.linalg.norm(self.goal_world - self.data.qpos[:2]))
        old_bearing = abs(math.atan2(self.goal_body[1], self.goal_body[0]))
        if self.skill == "nav":
            self._goal_and_command()
        elif self.skill == "run":
            self.command[:] = (min(0.6, 0.2 + 0.004 * self.step_count) if self.step_count < 100
                               else 0.6 if self.step_count < 500 else 0.0, 0.0)
        else:
            self.command[:] = 0.0
        self.action_buffer.append(action.copy())
        applied = self.action_buffer[-1 - self.action_lag]
        requested = self._target_from_action(applied)
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
            self.data.qvel[:2] += self.np_random.uniform(-0.1, 0.1, 2)
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
        if self.skill == "squat" and not contacts.all():
            self.squat_air_steps += 1
        gyro = self.data.sensor("imu_ang_vel").data
        body_velocity = self.data.sensor("imu_lin_vel").data
        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        tilt = -float(gravity[2])
        height = float(self.data.qpos[2])
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
                            and self.self_contact_steps == 0)
            reward = (3.0 * math.exp(-(height_error / 0.05) ** 2) + upright
                      + 0.4 * float(np.sum(contacts)) - 3.0 * drift - 3.0 * slip)
        elif self.skill == "recover":
            stable = height > 0.55 and tilt > math.cos(math.radians(15)) and still and contacts.all()
            self.stable_steps = self.stable_steps + 1 if stable else 0
            self.success = self.stable_steps >= 100
            reward = (3.0 * upright + 2.0 * np.clip((height - 0.2) / 0.4, 0.0, 1.0)
                      + 0.02 * self.stable_steps - 0.3 * drift)
            if self.success:
                reward += 30.0
        elif self.skill == "run":
            if 100 <= self.step_count < 500:
                self.run_speeds.append(float(body_velocity[0]))
            tracking = math.exp(-((float(body_velocity[0]) - self.command[0]) / 0.20) ** 2)
            self.success = (self.step_count >= 600 and len(self.run_speeds) >= 350
                            and 0.5 <= np.mean(self.run_speeds) <= 0.7
                            and self.run_flights >= 5 and height > 0.55 and tilt > 0.96 and still
                            and contacts.all() and abs(float(self.data.qpos[1] - self.initial_xy[1])) < 0.5
                            and self.total_slip / self.step_count < 0.05 and self.self_contact_steps == 0)
            reward = (3.0 * tracking + upright + 0.2 * float(self.flight_steps >= 4)
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
            self.success = (self.apex_rise >= 0.05 and self.max_flight_steps >= 8
                            and travel_ok and self.stable_steps >= 100 and self.self_contact_steps == 0)
            reward = (2.0 * upright + 2.0 * min(self.apex_rise / 0.05, 1.0)
                      + 0.5 * float(self.max_flight_steps >= 8) + 0.02 * self.stable_steps)
            if self.jump_forward:
                reward += min(max(forward, 0.0) / 0.20, 1.0)
            else:
                reward -= 2.0 * abs(forward)
            if self.success:
                reward += 30.0
        reward -= 0.15 * action_rate + 0.05 * action_accel + 0.03 * effort
        if self.skill != "recover" and (fallen or nonfinite):
            reward -= 100.0
        terminated = bool(self.success or nonfinite or (fallen and self.skill != "recover"))
        truncated = self.step_count >= MAX_STEPS[self.skill]
        info = {
            "skill": self.skill, "success": bool(self.success), "fallen": bool(fallen),
            "seconds": self.step_count * POLICY_DT, "distance_to_goal_m": float(np.linalg.norm(self.goal_world - self.data.qpos[:2])) if self.skill == "nav" else None,
            "drift_m": drift, "base_height_m": height, "slip_m_s": slip, "action_rate": action_rate,
            "effort": effort, "impact_weight": impact, "self_contacts": self_contacts,
            "self_contact_steps": self.self_contact_steps, "squat_air_steps": self.squat_air_steps,
            "flight_events": self.run_flights if self.skill == "run" else self.flight_events,
            "max_flight_s": self.max_flight_steps * 0.005, "apex_rise_m": self.apex_rise,
            "mean_run_speed_m_s": float(np.mean(self.run_speeds)) if self.run_speeds else None,
            "low_holds": self.low_holds.tolist(), "high_holds": self.high_holds.tolist(),
            "recovery_pose": self.recovery_pose if self.skill == "recover" else None,
            "jump_variant": "forward" if self.jump_forward else "up" if self.skill == "jump" else None,
        }
        return self._observe(), float(reward), terminated, truncated, info
