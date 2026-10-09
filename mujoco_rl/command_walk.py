"""Versioned continuous speed/yaw control and trajectory-based acceptance."""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
from gymnasium import spaces

from .full_body_env import POLICY_DT, STANDING_POSE
from .skill_env import SkillEnv, _yaw

CONTRACT = "asimov_command_walk_v1"
FAMILIES = ("forward", "left_arc", "right_arc", "left_pivot", "right_pivot", "stop", "mixed")
STAGES = ("gentle", "full_commands", "half_physics", "full_physics")
LIMITS = {"speed_error": .05, "yaw_error": .10, "heading_rms": math.radians(10),
          "heading_max": math.radians(20), "lateral_max": .25, "pivot_drift": .25,
          "stop_speed": .05, "stop_yaw": .10, "slip": .05}
DEFAULTS = {"contract": CONTRACT, "workers": 8, "rollout_steps": 768, "batch_size": 512,
            "epochs": 4, "learning_rate": 3e-5, "clip_range": .1, "target_kl": .015,
            "gamma": .995, "initial_std": .15, "maximum_steps": 10_000_000,
            "evaluation_interval": 100_000, "development_episodes": 50,
            "block_steps": 1_000_000, "maximum_hours": 72}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def segments_for(family, speed, turn):
    if family not in FAMILIES:
        raise ValueError(f"Unknown command family: {family}")
    commands = {"forward": (speed, 0.), "left_arc": (speed, turn),
                "right_arc": (speed, -turn), "left_pivot": (0., turn),
                "right_pivot": (0., -turn)}
    if family == "mixed":
        return [(4., *commands[k]) for k in ("forward", "left_arc", "forward", "right_arc",
                                             "left_pivot", "right_pivot")] + [(4., 0., 0.)]
    if family == "stop":
        return [(4., speed, 0.), (8., 0., 0.)]
    return [(8., *commands[family]), (4., 0., 0.)]


def assess_trace(trace, complete=True):
    """Evaluate full trajectories; final-position coincidence cannot pass a loop."""
    if not trace:
        return {"success": False, "failures": ["empty"], "metrics": {}, "violation": 100.}
    failures, metrics, violations = [], {}, []
    if not complete:
        failures.append("incomplete")
        violations.append(1.)
    if any(r["fallen"] for r in trace):
        failures.append("fall")
        violations.append(1.)
    if any(r["self_contacts"] for r in trace):
        failures.append("self_collision")
        violations.append(1.)

    def check(name, value, limit):
        metrics[name] = float(value)
        excess = max(0., float(value) / limit - 1.)
        if excess > 1e-6:
            failures.append(name)
        violations.append(min(20., excess))

    check("slip", np.mean([r["slip"] for r in trace]), LIMITS["slip"])
    for segment in sorted({r["segment"] for r in trace}):
        rows = [r for r in trace if r["segment"] == segment]
        measured = [r for r in rows if r["age"] >= 2.]
        prefix = f"segment{segment}/"
        if not measured:
            failures.append(prefix + "too_short")
            violations.append(1.)
            continue
        for key, values in (("speed_error", [abs(r["vx"] - r["command_v"]) for r in measured]),
                            ("yaw_error", [abs(r["yaw_rate"] - r["command_w"]) for r in measured])):
            check(prefix + key, np.mean(values), LIMITS[key])
        heading = np.array([r["heading_error"] for r in measured])
        check(prefix + "heading_rms", np.sqrt(np.mean(heading ** 2)), LIMITS["heading_rms"])
        check(prefix + "heading_max", np.max(np.abs(heading)), LIMITS["heading_max"])
        v, w = rows[0]["requested_v"], rows[0]["requested_w"]
        if v > 0 and abs(w) < 1e-8:
            check(prefix + "lateral_max", max(abs(r["lateral"]) for r in rows), LIMITS["lateral_max"])
            expected = sum(r["command_v"] * POLICY_DT for r in rows)
            actual = rows[-1]["forward"]
            check(prefix + "progress_error", abs(actual - expected), .05 * len(rows) * POLICY_DT + .1)
            if actual <= 0:
                failures.append(prefix + "no_forward_progress")
                violations.append(1.)
        if v == 0 and abs(w) > 0:
            check(prefix + "pivot_drift", max(r["displacement"] for r in rows), LIMITS["pivot_drift"])
        if abs(w) > 0:
            rotation = rows[-1]["rotation"]
            if rotation * w <= 0:
                failures.append(prefix + "wrong_turn_direction")
                violations.append(1.)
        if v == 0 and w == 0:
            check(prefix + "stop_speed", max(r["speed"] for r in measured), LIMITS["stop_speed"])
            check(prefix + "stop_yaw", max(abs(r["yaw_rate"]) for r in measured), LIMITS["stop_yaw"])
    return {"success": not failures, "failures": failures, "metrics": metrics,
            "violation": float(np.mean(violations)), "seconds": len(trace) * POLICY_DT}


class CommandWalkEnv(SkillEnv):
    """One policy for walking, steering, pivoting, and stopping; all distances in metres."""

    def __init__(self, stage=0, family=None, render_mode=None, external=False, clock=time.monotonic):
        if stage not in range(len(STAGES)) or (family is not None and family not in FAMILIES):
            raise ValueError("Invalid stage or command family")
        self.pending_stage = self.stage = stage
        self.family_override, self.external, self.clock = family, external, clock
        self.reference_heading = self.actual_heading = 0.
        self.requested = np.zeros(2)
        self.last_command_time = -math.inf
        self.trace = []
        super().__init__("nav", randomize=stage >= 2, render_mode=render_mode)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(85,), dtype=np.float32)

    def set_stage(self, stage):
        if stage not in range(len(STAGES)):
            raise ValueError("Invalid curriculum stage")
        self.pending_stage = stage

    def _goal_and_command(self):
        # SkillEnv calls this before/after physics. External commands own this task.
        pass

    def _target_from_action(self, action):
        return np.clip(STANDING_POSE + .25 * action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _observe(self, advance_sensors=True):
        base = super()._observe(advance_sensors)
        base[81:83] = 0.
        error = self.reference_heading - _yaw(self.data.qpos[3:7])
        return np.r_[base, math.sin(error), math.cos(error)].astype(np.float32)

    def set_command(self, forward_m_s, yaw_rad_s):
        values = np.array([forward_m_s, yaw_rad_s], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Commands must be finite")
        self.requested[:] = np.clip(values, [0., -.6], [.3, .6])
        self.last_command_time = self.clock()

    def _prepare_command(self):
        if self.external:
            if self.clock() - self.last_command_time >= .5:
                self.requested[:] = 0.
        else:
            elapsed, offset, index = self.step_count * POLICY_DT, 0., 0
            for index, (duration, v, w) in enumerate(self.schedule):
                if elapsed < offset + duration - 1e-8:
                    break
                offset += duration
            else:
                v = w = 0.
            if index != self.segment:
                self.segment = index
                self.segment_origin = self.data.qpos[:2].copy()
                self.segment_heading = self.reference_heading
                self.segment_actual_heading = self.actual_heading
            self.segment_start_time = offset
            self.set_command(v, w)
        self.command += np.clip(self.requested - self.command,
                                -np.array([.3, 1.2]) * POLICY_DT, np.array([.3, 1.2]) * POLICY_DT)

    def reset(self, *, seed=None, options=None):
        self.stage = self.pending_stage
        self.randomize = self.stage >= 2
        self.randomization_strength = .5 if self.stage == 2 else 1.
        self.reference_heading = self.actual_heading = 0.
        super().reset(seed=seed, options=options)
        # Starting-state diversity is independent of physics randomization.
        roll, pitch = self.np_random.uniform(-.035, .035, 2)
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([roll, pitch, yaw]), "xyz")
        self.data.qpos[self.qpos_ids] = np.clip(STANDING_POSE + self.np_random.uniform(-.02, .02, self.nj),
                                               self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self.data.qvel[self.qvel_ids] = self.np_random.uniform(-.02, .02, self.nj)
        mujoco.mj_forward(self.model, self.data)
        self.sensor_buffer.clear()
        for _ in range(3):
            self.sensor_buffer.append(self._raw_sensors())
        self.goal_body[:] = self.goal_world[:] = 0.
        self.command[:] = self.requested[:] = 0.
        self.reference_heading = self.actual_heading = self.previous_yaw = _yaw(self.data.qpos[3:7])
        self.segment_actual_heading = self.actual_heading
        self.segment_heading = self.reference_heading
        self.segment_origin = self.data.qpos[:2].copy()
        self.segment, self.segment_start_time = 0, 0.
        self.family = self.family_override or str(self.np_random.choice(FAMILIES))
        self.schedule = segments_for(self.family, float(self.np_random.uniform(.1, .2 if self.stage == 0 else .3)),
                                     float(self.np_random.uniform(.15, .3 if self.stage == 0 else .6)))
        self.duration = sum(s[0] for s in self.schedule)
        self.trace, self.reward_totals = [], {}
        self.last_command_time = -math.inf
        self._prepare_command()
        return self._observe(False), {"contract": CONTRACT, "family": self.family, "stage": self.stage}

    def step(self, action):
        if self.external:
            self._prepare_command()
        applied_command = self.command.copy()
        self.reference_heading += applied_command[1] * POLICY_DT
        push_due = self.randomize and self.step_count + 1 == self.next_push
        # Braking is measured after the motion command ends. Do not inject new
        # velocity impulses during a hold that requires every sample below .05 m/s.
        if push_due and not np.any(self.requested):
            self.next_push += int(self.np_random.integers(100, 200))
            push_due = False
        _, _, _, _, old_info = super().step(action)
        yaw = _yaw(self.data.qpos[3:7])
        delta = wrap(yaw - self.previous_yaw)
        self.actual_heading += delta
        self.previous_yaw = yaw
        yaw_rate = delta / POLICY_DT
        heading_error = self.reference_heading - self.actual_heading
        velocity = self.data.sensor("imu_lin_vel").data
        gravity = self.data.site_xmat[self.imu_site_id].reshape(3, 3).T @ np.array([0., 0., -1.])
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        fallen = bool(old_info["fallen"] or not finite)
        parts = {"speed": 3 * math.exp(-((velocity[0] - applied_command[0]) / .15) ** 2),
                 "turn": 2 * math.exp(-((yaw_rate - applied_command[1]) / .3) ** 2),
                 "heading": math.exp(-(heading_error / .25) ** 2),
                 "upright": math.exp(-float(gravity[:2] @ gravity[:2]) / .1),
                 "slip": -4 * old_info["slip_m_s"], "lateral": -1.5 * abs(float(velocity[1])),
                 "action_rate": -.15 * old_info["action_rate"], "action_accel": -.05 * old_info["action_accel"],
                 "effort": -.03 * old_info["effort"], "fall": -100. if fallen else 0.}
        for key, value in parts.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.) + float(value)
        displacement = self.data.qpos[:2] - self.segment_origin
        c, s = math.cos(self.segment_heading), math.sin(self.segment_heading)
        row = {"segment": self.segment, "age": self.step_count * POLICY_DT - self.segment_start_time,
               "command_v": float(applied_command[0]), "command_w": float(applied_command[1]),
               "requested_v": float(self.requested[0]), "requested_w": float(self.requested[1]),
               "vx": float(velocity[0]), "yaw_rate": yaw_rate, "heading_error": heading_error,
               "lateral": float(-s * displacement[0] + c * displacement[1]),
               "forward": float(c * displacement[0] + s * displacement[1]),
               "displacement": float(np.linalg.norm(displacement)),
               "rotation": self.actual_heading - self.segment_actual_heading,
               "speed": float(np.linalg.norm(self.data.qvel[:2])), "fallen": fallen,
               "self_contacts": int(old_info["self_contacts"]), "slip": float(old_info["slip_m_s"]),
               "x": float(self.data.qpos[0]), "y": float(self.data.qpos[1]),
               "reference_heading": self.reference_heading, "actual_heading": self.actual_heading}
        row['push_applied'] = bool(push_due)
        self.trace.append(row)
        complete = not self.external and self.step_count * POLICY_DT >= self.duration - 1e-8
        info = {**row, "family": self.family, "stage": self.stage, "contract": CONTRACT,
                "reward_components": parts, "reward_totals": self.reward_totals.copy(), "success": False}
        if fallen or complete:
            info.update(assess_trace(self.trace, complete))
        if not self.external:
            self._prepare_command()
        return self._observe(False), float(sum(parts.values())), fallen, complete and not fallen, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=368, width=480)
        self.renderer.update_scene(self.data, camera=self.camera)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return self.renderer.render().copy()
