"""Reproducible initial states and physical disturbances for skill evaluations."""

from __future__ import annotations

import numpy as np

from .skill_env import RECOVERY_POSES, SkillEnv


def disturb(env: SkillEnv, pose: str, strength: float = 120.0) -> None:
    """Apply an external world-frame force; do not teleport the robot."""
    direction = {"front": (1, 0), "back": (-1, 0), "left": (0, -1), "right": (0, 1)}[pose]
    env.data.xfrc_applied[env.torso_id, :2] = np.asarray(direction) * strength


def reset_case(env: SkillEnv, seed: int, scenario: str = "standard", pose: str | None = None,
               tilt: float | None = None) -> tuple[np.ndarray, dict]:
    pose = pose or RECOVERY_POSES[seed % 4]
    options = {}
    if env.training_skill == "recover":
        options["recovery_pose"] = pose
        if tilt is not None:
            options["recovery_tilt_deg"] = tilt
        if scenario == "dynamic_fall":
            options["recovery_tilt_deg"] = 0.0
    observation, info = env.reset(seed=seed, options=options)
    if scenario == "standard":
        return observation, info
    return prepare_recovery(env, scenario, pose)


def prepare_recovery(env: SkillEnv, scenario: str, pose: str) -> tuple[np.ndarray, dict]:
    """Settle or disturb an already initialized robot without recursively resetting it."""
    if scenario not in {"settled_floor", "dynamic_fall"} or env.skill != "recover":
        raise ValueError("Recovery scenario must be standard, settled_floor, or dynamic_fall")
    env.episode_recovery_level = None
    holding = env.action_from_target(env.data.qpos[env.qpos_ids])
    fell = scenario == "settled_floor"
    # Settling is real physics. Neither velocities nor root position are replaced.
    for step in range(100 if scenario == "dynamic_fall" else 50):
        if scenario == "dynamic_fall" and step < 20:
            disturb(env, pose)
        else:
            env.data.xfrc_applied[:] = 0.0
        observation, _, _, _, info = env.step(holding)
        if scenario == "dynamic_fall" and step >= 19 and info["fallen"]:
            fell = True
            break
    env.data.xfrc_applied[:] = 0.0
    env.switch_skill("recover")
    env.recovery_pose = pose
    observation = env._observe(advance_sensors=False)
    return observation, {**info, "scenario": scenario, "disturbance_produced_fall": fell}
