"""Nominal specialist tasks, deterministic evaluations, and V4 behavioral gates."""

from dataclasses import asdict, dataclass

import numpy as np

from .command_walk import FAMILIES, segments_for
from .command_walk_robust import RobustCommandEnv
from .command_walk_robust_eval import summarize

BRAKES = tuple("brake_" + f for f in FAMILIES[:5])
CASES = (*FAMILIES, *BRAKES, "sequence")
TARGETS = {
    "all": FAMILIES,
    "arcs": ("left_arc", "right_arc"),
    "pivots": ("left_pivot", "right_pivot"),
    "braking": BRAKES,
    "transitions": FAMILIES,
}
INITIALIZATIONS = {"current": (False, 2), "preserved": (True, 2), "calibrated": (False, 20)}


@dataclass(frozen=True)
class Spec:
    initialization: str = "preserved"
    task: str = "all"
    heading_penalty: bool = False
    workers: int = 8
    rollout_steps: int = 768
    version: int = 4

    def __post_init__(self):
        if (
            self.initialization not in INITIALIZATIONS
            or self.task not in TARGETS
            or self.workers not in range(1, 9)
            or self.rollout_steps < 2
            or self.version != 4
        ):
            raise ValueError("Invalid V4 configuration")


def sequence_schedule(rng, speed, turn, long=False):
    names = ["forward", "left_arc", "forward", "right_arc", "left_pivot", "right_pivot", "stop", "forward"]
    if rng.random() < 0.5:
        names = [n.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right") for n in names]
    if long:
        names += list(rng.choice(FAMILIES[:6], 4))
    durations = []
    remaining = 60.0
    for i in range(len(names)):
        if long:
            others = len(names) - i - 1
            lo, hi = max(3.0, remaining - 8.0 * others), min(8.0, remaining - 3.0 * others)
            duration = remaining if not others else float(rng.uniform(lo, hi))
            remaining -= duration
        else:
            duration = float(rng.uniform(3, 8))
        durations.append(duration)
    return [
        (duration, *(segments_for(name, speed, turn)[0][1:] if name != "stop" else (0.0, 0.0)))
        for duration, name in zip(durations, names)
    ]


class MotionEnv(RobustCommandEnv):
    def __init__(self, spec=None, training=False, case=None, gentle=False, render_mode=None):
        self.spec = spec or Spec()
        self.v4_training, self.case, self.gentle = training, case, gentle
        self.motion_rng = np.random.default_rng(0)
        self.episode_index = 0
        parent_case = case if case in FAMILIES else "forward"
        super().__init__(
            level=0,
            variant="control",
            training=training and self.spec.task == "all",
            family=None if training else parent_case,
            render_mode=render_mode,
        )

    def set_gentle(self, value):
        self.gentle = bool(value)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self.motion_rng = np.random.default_rng(np.random.SeedSequence([seed, 4004]))
            self.episode_index = int(seed) % 2
        rng = self.motion_rng
        case = self.case
        if self.v4_training:
            if self.spec.task == "all":
                return obs, info
            rehearsal = 0.30 if self.spec.task == "transitions" else 0.20
            target_seconds = 44.0 if self.spec.task == "transitions" else 16.0 if self.spec.task == "braking" else 12.0
            # Convert target simulation-time allocation into episode probabilities.
            rehearsal_probability = (rehearsal / 12.0) / (rehearsal / 12.0 + (1 - rehearsal) / target_seconds)
            if rng.random() < rehearsal_probability:
                case = (
                    str(rng.choice(FAMILIES[:6]))
                    if self.spec.task == "transitions"
                    else str(rng.choice(["forward", "stop"]))
                )
            elif self.spec.task == "transitions":
                case = "sequence"
            else:
                targets = TARGETS[self.spec.task]
                case = targets[self.episode_index % len(targets)]
                self.episode_index += 1
        if case in FAMILIES and not self.v4_training and not self.gentle:
            return obs, info  # Exact existing nominal evaluation cases.
        speed, turn = float(rng.uniform(0.1, 0.2 if self.gentle else 0.3)), float(
            rng.uniform(0.15, 0.3 if self.gentle else 0.6)
        )
        if case == "sequence":
            schedule = sequence_schedule(rng, speed, turn, long=not self.v4_training)
        elif case in BRAKES:
            movement = segments_for(case.removeprefix("brake_"), speed, turn)[0][1:]
            schedule = [(4.0, *movement), (4.0, 0.0, 0.0), (4.0, speed, 0.0), (4.0, 0.0, 0.0)]
        else:
            schedule = segments_for(case, speed, turn)
        self.family, self.schedule = case, schedule
        self.duration = sum(s[0] for s in schedule)
        self.command[:] = self.requested[:] = 0
        self._prepare_command()
        return self._observe(False), info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if self.spec.heading_penalty:
            segment = info["segment"]
            turning = abs(self.schedule[segment][2]) > 0
            after_turn = (
                segment > 0 and abs(self.schedule[segment - 1][2]) > 0 and self.schedule[segment][1:] == (0.0, 0.0)
            )
            cost = -0.5 * min((info["heading_error"] / np.deg2rad(10)) ** 2, 4) if turning or after_turn else 0.0
            reward += cost
            info["reward_components"]["v4_heading"] = cost
        return obs, reward, done, truncated, info


def panel(model, norm, cases=FAMILIES, count=50, seed_start=1001, gentle=False, progress=None):
    if count < 1 or not cases or any(case not in CASES for case in cases):
        raise ValueError("Invalid evaluation cases/count")
    episodes = {case: [] for case in cases}
    size = model.observation_space.shape[0]
    for case in cases:
        for offset in range(0, count, 10):
            envs = [MotionEnv(case=case, gentle=gentle) for _ in range(min(10, count - offset))]
            seeds = [seed_start + CASES.index(case) * 100_000 + offset + i for i in range(len(envs))]
            try:
                obs = [env.reset(seed=s)[0] for env, s in zip(envs, seeds)]
                active = list(range(len(envs)))
                while active:
                    actions, _ = model.predict(
                        norm.normalize_obs(np.array([obs[i][:size] for i in active])), deterministic=True
                    )
                    remaining = []
                    for i, action in zip(active, actions):
                        env = envs[i]
                        if progress and env.step_count % 100 == 0:
                            progress(case, offset + i, env.step_count)
                        obs[i], _, done, truncated, info = env.step(action)
                        if done or truncated:
                            episodes[case].append(
                                {k: info[k] for k in ["success", "failures", "metrics", "violation", "seconds"]}
                                | dict(seed=seeds[i], schedule=env.schedule)
                            )
                        else:
                            remaining.append(i)
                    active = remaining
            finally:
                for env in envs:
                    env.close()
    return dict(
        **summarize(episodes),
        seed_start=seed_start,
        gentle=gentle,
        physics="nominal",
        training_steps=model.num_timesteps,
    )


def retains(reference, candidate, tolerance=0.04, cases=None):
    return all(
        candidate["families"][k]["pass_rate"] + tolerance + 1e-9 >= reference["families"][k]["pass_rate"]
        for k in (cases or reference["families"])
    )


def passes(report, cases, threshold=0.96, count=50):
    return all(
        report["families"][k]["count"] >= count and report["families"][k]["pass_rate"] + 1e-9 >= threshold
        for k in cases
    )


def specialist_passes(report, task):
    return passes(report, TARGETS[task]) and passes(report, ("forward", "stop"), 0.92)


def select_initialization(results):
    eligible = {
        name: rows
        for name, rows in results.items()
        if len(rows) == 2 and all(retains(r["initial"], r["final"]) for r in rows)
    }
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda name: (
            min(r["final"]["minimum_pass_rate"] for r in eligible[name]),
            np.mean([r["final"]["mean_pass_rate"] for r in eligible[name]]),
            name == "preserved",
        ),
    )


def teacher_for(command):
    v, w = command
    return (
        "braking"
        if abs(v) < 1e-8 and abs(w) < 1e-8
        else "pivots" if abs(v) < 1e-8 else "arcs" if abs(w) > 1e-8 else "baseline"
    )
