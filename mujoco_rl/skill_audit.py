"""Audit the simulator contract and export a fixed recovery diagnostic panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .full_body_env import ACTUATOR_GROUPS, JOINT_NAMES, PHYSICS_DT, POLICY_DT, URDF_PATH, XML_PATH
from .skill_env import ENVIRONMENT_VERSION, JUMP_MIN_FLIGHT_SECONDS, RECOVERY_POSES, SkillEnv
from .skill_scenarios import reset_case


def audit_contract() -> dict:
    env = SkillEnv("recover", randomize=False, recovery_level=0)
    try:
        obs, _ = env.reset(seed=0)
        grid = np.linspace(-1, 1, 257)
        mapping = np.stack([env._target_from_action(np.full(23, value)) for value in grid])
        start = env.data.time
        _, _, _, _, info = env.step(np.zeros(23))
        checks = {
            "observation_shape": obs.shape == (83,), "action_shape": env.action_space.shape == (23,),
            "policy_time": bool(np.isclose(env.data.time - start, POLICY_DT)),
            "physics_time": bool(np.isclose(env.model.opt.timestep, PHYSICS_DT)),
            "joint_addresses_unique": len(set(env.qpos_ids)) == 23,
            "targets_within_limits": bool(np.all(mapping >= env.joint_ranges[:, 0] - 1e-9)
                                          and np.all(mapping <= env.joint_ranges[:, 1] + 1e-9)),
            "mapping_monotonic": bool(np.all(np.diff(mapping, axis=0) >= -1e-9)),
            "torque_within_limits": bool(np.all(np.abs(env.data.ctrl[env.ctrl_ids]) <= env.effort)),
            "positive_link_mass": bool(np.all(env.model.body_mass[1:] > 0)),
            "positive_link_inertia": bool(np.all(env.model.body_inertia[1:] > 0)),
            "jump_flight_units": bool(np.isclose(JUMP_MIN_FLIGHT_SECONDS / PHYSICS_DT, 32)),
            "finite_state": bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()),
            "motor_joint_order": all(env.model.actuator_trnid[actuator, 0] == env.model.joint(name).id
                                      for actuator, name in zip(env.ctrl_ids, JOINT_NAMES)),
            "motor_transmission_sign": bool(np.all(env.model.actuator_gear[env.ctrl_ids, 0] == 1)),
            "no_duplicate_passive_pd": bool(np.all(env.model.dof_damping[env.qvel_ids] == 0)
                                             and np.all(env.model.jnt_stiffness[1:] == 0)),
        }
        for skill in ("nav", "squat", "recover", "run", "jump"):
            qpos, qvel, target = env.data.qpos.copy(), env.data.qvel.copy(), env.target.copy()
            env.switch_skill(skill)
            checks[f"{skill}_transition_preserves_state"] = bool(np.array_equal(qpos, env.data.qpos)
                and np.array_equal(qvel, env.data.qvel) and np.array_equal(target, env.target))
        return {"environment_version": ENVIRONMENT_VERSION, "passed": all(checks.values()), "checks": checks,
                "joint_names": JOINT_NAMES, "joint_limits_rad": env.joint_ranges.tolist(),
                "torque_limits_nm": env.effort.tolist(), "speed_limits_rad_s": env.speed.tolist(),
                "total_mass_kg": float(env.model.body_mass.sum()),
                "asset_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (XML_PATH, URDF_PATH)},
                "provisional": {"actuator_gains": ACTUATOR_GROUPS, "motor_sensor_delay_ms": [0, 20, 40],
                                "delayed_sensors": "gyro, gravity, joint positions and velocities",
                                "undelayed_inputs": "body-velocity estimate, simulator navigation localization",
                                "contact_friction": "asset values with training randomization",
                                "wrist_support": "local floor-contact collision overlay",
                                "body_velocity": "MuJoCo IMU-frame velocity; hardware estimator required"},
                "scope": "simulation consistency; hardware response has not been measured"}
    finally:
        env.close()


def diagnostic_panel(checkpoint: Path, output: Path) -> list[dict]:
    model = PPO.load(checkpoint, device="cpu")
    normalizer = VecNormalize.load(str(checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")),
                                   DummyVecEnv([lambda: SkillEnv("recover")]))
    normalizer.training = False
    env = SkillEnv("recover", recovery_sensor_memory=getattr(model, "recovery_sensor_memory", False),
                   recovery_phase_features=getattr(model, "recovery_phase_features", False))
    cases = [{"name": f"{pose}_{tilt}", "pose": pose, "tilt": tilt, "scenario": "standard",
              "seed": 2_100_000 + p * 100 + index}
             for p, pose in enumerate(RECOVERY_POSES) for index, tilt in enumerate((23, 25, 28, 30))]
    cases += [{"name": f"{scenario}_{pose}", "pose": pose, "tilt": 90, "scenario": scenario,
               "seed": 2_110_000 + p + index * 100}
              for index, scenario in enumerate(("settled_floor", "dynamic_fall"))
              for p, pose in enumerate(RECOVERY_POSES)]
    cases += [{"name": f"known_{seed}", "seed": seed, "pose": "right", "tilt": None,
               "scenario": "standard", "level": 9} for seed in (20011, 20035, 20047)]
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for case in cases:
            env.recovery_level = case.get("level")
            observation, initial = reset_case(env, case["seed"], case["scenario"], case["pose"], case["tilt"])
            frames = []
            for step in range(600):
                action, _ = model.predict(normalizer.normalize_obs(observation[None]), deterministic=True)
                frame = {"seconds": step * POLICY_DT, "observation": observation.copy(),
                         "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                         "com": env.data.subtree_com[1].copy(), "action": action[0].copy()}
                observation, reward, terminated, truncated, info = env.step(action[0])
                frame.update(target=env.target.copy(), torque=env.data.ctrl[env.ctrl_ids].copy(),
                             contacts=env.last_contacts.copy(), reward=reward)
                frames.append(frame)
                if terminated or truncated:
                    break
            np.savez_compressed(output / f"{case['name']}.npz",
                                **{key: np.asarray([item[key] for item in frames]) for key in frames[0]})
            rows.append({**case, **info, "disturbance_produced_fall": initial.get("disturbance_produced_fall"),
                         "classification": "unclassified_requires_trace_review"})
            print(f"[diagnostic] {case['name']}: success={info['success']} duration={info['seconds']:.2f}s", flush=True)
    finally:
        env.close()
        normalizer.close()
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit_contract()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "physics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["checks"], indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Simulation audit failed; training must remain stopped")
    if args.checkpoint:
        diagnostic_panel(args.checkpoint, args.output / "traces")


if __name__ == "__main__":
    main()
