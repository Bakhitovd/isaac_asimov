"""Versioned moderate-turn walking contract, corrected observations, and curriculum."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .command_walk import FAMILIES, segments_for
from .command_walk_correction import CommandCorrection, Correction
from .command_walk_v4 import BRAKES, MotionEnv
from .full_body_env import POLICY_DT

VERSION = "asimov_moderate_walk_v1"
CORRECTION = Correction("moderate_speed_002", heading_gain=0.5, turn_speed_reduction=0.02)
CASES = (*FAMILIES, *BRAKES, "left_exit", "right_exit", "sequence")
TARGETS = ("left_exit", "right_exit", *BRAKES)
QUALITY = ("action_rate", "action_accel", "effort")


def contract():
    return dict(
        version=VERSION,
        forward_range=[0.1, 0.3],
        yaw_range=[-0.5, 0.5],
        minimum_turn=0.15,
        stop_and_pivot_speed=0.0,
        correction=asdict(CORRECTION),
        policy_dt=POLICY_DT,
        observations=87,
        actions=23,
        physics="nominal",
    )


def validate_command(v, w):
    if not np.isfinite([v, w]).all() or not (v == 0 or 0.1 <= v <= 0.3) or not (w == 0 or 0.15 <= abs(w) <= 0.5):
        raise ValueError("Supported requests: v=0 or 0.1..0.3 m/s; yaw=0 or +/-0.15..0.5 rad/s")


def penalty(error, limit):
    return min(4.0, (abs(float(error)) / limit) ** 2)


def extra_costs(info):
    v, w = info["requested_v"], info["requested_w"]
    settle = min(1.0, max(0.0, info["age"] / 2.0))
    return dict(
        final_heading=-0.5 * penalty(info["heading_error"], np.deg2rad(10)) * settle,
        final_yaw=-0.5 * penalty(info["yaw_rate"] - info["command_w"], 0.1) * settle if w else 0.0,
        final_lateral=-penalty(info["lateral"], 0.25) if v and not w else 0.0,
        final_pivot=-0.5 * penalty(info["displacement"], 0.25) if not v and w else 0.0,
        final_stop=(
            -(penalty(info["speed"], 0.05) + 0.5 * penalty(info["yaw_rate"], 0.1)) * settle if not v and not w else 0.0
        ),
    )


def schedule_for(case, rng, seconds=60, fixed=None):
    v, w = fixed if fixed is not None else (float(rng.uniform(0.1, 0.3)), float(rng.uniform(0.15, 0.5)))
    if case in FAMILIES:
        return segments_for(case, v, w)
    if case in BRAKES:
        movement = segments_for(case.removeprefix("brake_"), v, w)[0][1:]
        return [(4.0, *movement), (4.0, 0.0, 0.0), (4.0, v, 0.0), (4.0, 0.0, 0.0)]
    if case in ("left_exit", "right_exit"):
        w = float(rng.uniform(0.35, 0.5)) if fixed is None else w
        return [(4.0, v, 0.0), (4.0, v, w if case == "left_exit" else -w), (6.0, v, 0.0), (4.0, 0.0, 0.0)]
    if case != "sequence" or seconds not in (20, 40, 60):
        raise ValueError("Unknown case or sequence duration")
    names = ["forward", "left_arc", "forward", "right_arc", "left_pivot", "right_pivot", "stop", "forward"]
    if seconds == 20:
        names = ["forward", "left_arc", "forward", "right_arc", "stop"]
    elif seconds == 60:
        names += list(rng.choice(FAMILIES[:6], 4))
    if rng.random() < 0.5:
        names = [n.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right") for n in names]
    remaining = float(seconds)
    result = []
    for index, name in enumerate(names):
        others = len(names) - index - 1
        duration = (
            remaining
            if not others
            else float(rng.uniform(max(3.0, remaining - 8 * others), min(8.0, remaining - 3 * others)))
        )
        remaining -= duration
        result.append((duration, *segments_for(name, v, w)[0][1:]))
    return result


def sampling(phase, seconds):
    if phase == "sequence":
        names = [*FAMILIES, "sequence"]
        fractions = [0.4 / 7] * 7 + [0.6]
    else:
        names = [*FAMILIES, "left_exit", "right_exit", *BRAKES]
        fractions = [0.4 / 7] * 7 + [0.3, 0.15] + [0.15 / 5] * 5
    durations = [
        28 if n == "mixed" else seconds if n == "sequence" else 18 if n.endswith("exit") else 16 if n in BRAKES else 12
        for n in names
    ]
    probabilities = np.asarray(fractions) / durations
    return names, probabilities / probabilities.sum(), np.asarray(fractions), np.asarray(durations)


class FinalEnv(MotionEnv):
    def __init__(
        self,
        training=False,
        case="forward",
        reward_kind="original",
        phase="motion",
        seconds=60,
        fixed=None,
        render_mode=None,
    ):
        self.final_training = training
        self.final_case = case
        self.reward_kind = reward_kind
        self.phase, self.sequence_seconds = phase, seconds
        self.fixed = fixed
        self.command_rng = np.random.default_rng(0)
        self.corrector = CommandCorrection(CORRECTION)
        self.correction_calls = 0
        self.quality_totals = {k: 0.0 for k in QUALITY}
        super().__init__(case=case if case in FAMILIES else "forward", render_mode=render_mode)

    def set_phase(self, phase, seconds=60):
        if phase not in ("motion", "sequence") or seconds not in (20, 40, 60):
            raise ValueError("Invalid curriculum")
        self.phase, self.sequence_seconds = phase, seconds

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self.command_rng = np.random.default_rng(np.random.SeedSequence([seed, 5138]))
        case = self.final_case
        if self.final_training:
            names, p, _, _ = sampling(self.phase, self.sequence_seconds)
            case = str(self.command_rng.choice(names, p=p))
        self.family = case
        self.schedule = schedule_for(case, self.command_rng, self.sequence_seconds, self.fixed)
        for _, v, w in self.schedule:
            validate_command(v, w)
        self.duration = sum(s[0] for s in self.schedule)
        self.command[:] = self.requested[:] = 0
        self._prepare_command()
        self.corrector = CommandCorrection(CORRECTION)
        self.correction_calls = 0
        self.quality_totals = {k: 0.0 for k in QUALITY}
        return self.correct(self._observe(False)), info

    def correct(self, observation):
        self.correction_calls += 1
        return self.corrector.apply(observation)[0]

    def step(self, action):
        command = self.command.copy()
        expected = self.reference_heading + command[1] * POLICY_DT
        observation, reward, done, truncated, info = super().step(action)
        if abs(self.reference_heading - expected) > 1e-10 or not np.array_equal(
            command, [info["command_v"], info["command_w"]]
        ):
            raise RuntimeError("Requested trajectory changed")
        for key, coefficient in (("action_rate", -0.15), ("action_accel", -0.05), ("effort", -0.03)):
            self.quality_totals[key] += info["reward_components"][key] / coefficient
        if self.reward_kind == "aligned":
            costs = extra_costs(info)
            reward += sum(costs.values())
            info["reward_components"].update(costs)
        info["quality"] = {k: v / self.step_count for k, v in self.quality_totals.items()}
        corrected = self.correct(observation)
        if self.correction_calls != self.step_count + 1:
            raise RuntimeError("Command correction was not applied exactly once per observation")
        return corrected, reward, done, truncated, info
