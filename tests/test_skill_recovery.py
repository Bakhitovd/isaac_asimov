"""Recovery reset and action contract for the MuJoCo policy."""

import numpy as np
import pytest

from mujoco_rl.full_body_env import STANDING_POSE
from mujoco_rl.skill_env import RECOVERY_TILT_RANGES, SkillEnv


@pytest.mark.parametrize("level", [None, *range(len(RECOVERY_TILT_RANGES))])
@pytest.mark.parametrize("pose_index", range(4))
def test_recovery_reset_is_supported(level, pose_index):
    env = SkillEnv("recover", randomize=False, recovery_level=level)
    try:
        observation, _ = env.reset(seed=10_000 + pose_index)
        floor_contacts = [contact for contact in env.data.contact
                          if env.floor_id in (contact.geom1, contact.geom2)]
        assert observation.shape == (83,)
        assert floor_contacts
        assert min(contact.dist for contact in floor_contacts) >= -0.002
        assert np.allclose(env.data.qvel, 0.0)
        _, reward, _, _, info = env.step(np.zeros(env.nj))
        assert np.isfinite(reward)
        assert info["recovery_start_contacts"] > 0
    finally:
        env.close()


def test_recovery_actions_are_centered_on_standing_pose():
    env = SkillEnv("recover", randomize=False)
    try:
        assert np.allclose(env._target_from_action(np.zeros(env.nj)), STANDING_POSE)
        assert np.allclose(env._target_from_action(np.ones(env.nj)), env.joint_ranges[:, 1])
        assert np.allclose(env._target_from_action(-np.ones(env.nj)), env.joint_ranges[:, 0])
    finally:
        env.close()


def test_neutral_actions_can_hold_easy_recovery_pose():
    env = SkillEnv("recover", randomize=False, recovery_level=0)
    try:
        env.reset(seed=20_000)
        for _ in range(600):
            _, _, terminated, _, info = env.step(np.zeros(env.nj))
            if terminated:
                break
        assert info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("seed", range(20_000, 20_004))
def test_training_exploration_can_complete_easy_recovery(seed):
    env = SkillEnv("recover", randomize=True, recovery_level=0)
    rng = np.random.default_rng(seed)
    try:
        env.reset(seed=seed)
        for _ in range(600):
            _, _, terminated, truncated, info = env.step(rng.normal(0.0, 0.05, env.nj))
            if terminated or truncated:
                break
        assert info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("seed", range(20_000, 20_004))
def test_supported_torso_tilt_recovers_to_upright(seed):
    env = SkillEnv("recover", randomize=False, recovery_level=4)
    try:
        env.reset(seed=seed)
        initial_cos = env.data.site_xmat[env.imu_site_id].reshape(3, 3)[2, 2]
        assert initial_cos < 0.99
        for _ in range(600):
            _, _, terminated, truncated, info = env.step(np.zeros(env.nj))
            if terminated or truncated:
                break
        assert info["success"]
        assert env.data.site_xmat[env.imu_site_id].reshape(3, 3)[2, 2] > 0.99
    finally:
        env.close()


def test_full_lying_reset_receives_no_hip_compensation():
    env = SkillEnv("recover", randomize=False)
    try:
        env.reset(seed=20_000)
        assert env.recovery_tilt_deg == 90.0
        assert np.allclose(env.data.qpos[env.qpos_ids], STANDING_POSE)
    finally:
        env.close()


@pytest.mark.parametrize("level,expected_termination", [(5, True), (6, True), (7, True), (10, False), (None, False)])
def test_collapse_ends_only_starter_balance_episodes(level, expected_termination):
    env = SkillEnv("recover", randomize=False, recovery_level=level)
    try:
        env.reset(seed=20_000)
        env.data.qpos[env.qpos_ids] = STANDING_POSE
        env._set_recovery_pose("front", 90.0)
        _, _, terminated, truncated, info = env.step(np.zeros(env.nj))
        assert terminated == expected_termination
        assert not truncated
        assert not info["success"]
    finally:
        env.close()


def test_recovery_completion_beats_discounted_standing_reward():
    env = SkillEnv("recover", randomize=False, recovery_level=0)
    try:
        env.reset(seed=20_000)
        for _ in range(600):
            _, reward, terminated, truncated, info = env.step(np.zeros(env.nj))
            if terminated or truncated:
                break
        assert info["success"]
        assert reward > 1.5 / (1.0 - 0.995)
    finally:
        env.close()


def test_adaptive_pose_sampling_keeps_seeded_evaluation_balanced():
    env = SkillEnv("recover", randomize=False, recovery_level=0)
    try:
        env.set_recovery_pose_weights([100.0, 1.0, 1.0, 1.0])
        front_count = 0
        env.reset(seed=20_000)
        for _ in range(40):
            env.reset()
            front_count += env.recovery_pose == "front"
        assert front_count >= 30
        for seed, pose in zip(range(20_000, 20_004), ("front", "back", "left", "right")):
            env.reset(seed=seed)
            assert env.recovery_pose == pose
        with pytest.raises(ValueError):
            env.set_recovery_pose_weights([0.0, 1.0, 1.0, 1.0])
    finally:
        env.close()


def test_boundary_focus_does_not_change_evaluation_reset():
    env = SkillEnv("recover", randomize=False, recovery_level=5)
    try:
        before, _ = env.reset(seed=20_000)
        tilt = env.recovery_tilt_deg
        env.set_recovery_focus_ranges({"front": (16.25, 17.75)})
        after, _ = env.reset(seed=20_000)
        assert env.recovery_tilt_deg == tilt
        assert np.array_equal(before, after)
        env.set_recovery_level(6)
        assert not env.recovery_focus_ranges
    finally:
        env.close()


@pytest.mark.parametrize("pose", ("front", "back", "left", "right"))
def test_controlled_recovery_reset_for_randomized_search(pose):
    env = SkillEnv("recover", recovery_level=6)
    try:
        observation, _ = env.reset(seed=20_000, options={"recovery_pose": pose, "recovery_tilt_deg": 23.0})
        assert env.recovery_pose == pose
        assert env.recovery_tilt_deg == 23.0
        assert env.recovery_start_contacts > 0
        assert np.isfinite(observation).all()
        with pytest.raises(ValueError):
            env.reset(options={"recovery_tilt_deg": 120.0})
    finally:
        env.close()


@pytest.mark.parametrize("level", (0, 6, None))
@pytest.mark.parametrize("pose_index", range(4))
def test_recovery_motor_state_matches_initial_pose(level, pose_index):
    env = SkillEnv("recover", recovery_level=level)
    try:
        env.reset(seed=20_000 + pose_index)
        assert np.array_equal(env.target, env.data.qpos[env.qpos_ids])
        for command in env.action_buffer:
            assert np.allclose(env._target_from_action(command), env.target, atol=1e-8)
        assert np.array_equal(env.last_action, env.action_buffer[-1])
    finally:
        env.close()


@pytest.mark.parametrize("pose_index", range(4))
def test_recovery_memory_retains_initial_noisy_imu_reading(pose_index):
    env = SkillEnv("recover", recovery_level=7, recovery_sensor_memory=True)
    try:
        observation, _ = env.reset(seed=20000 + pose_index)
        initial = observation[3:5].copy()
        assert np.array_equal(observation[81:83], initial)
        for _ in range(20):
            observation, _, terminated, truncated, _ = env.step(np.zeros(23))
            assert np.array_equal(observation[81:83], initial)
            if terminated or truncated:
                break
        env.reset(seed=20004 + pose_index)
        assert np.array_equal(env.recovery_initial_gravity, env._observe()[81:83])
    finally:
        env.close()


def test_legacy_recovery_observations_keep_unused_goal_channels_zero():
    env = SkillEnv("recover", recovery_level=7)
    try:
        observation, _ = env.reset(seed=20000)
        assert not observation[81:83].any()
    finally:
        env.close()


def test_recovery_memory_does_not_replace_navigation_goal():
    env = SkillEnv("nav", recovery_sensor_memory=True)
    try:
        observation, _ = env.reset(seed=20000)
        assert np.allclose(observation[81:83], np.clip(env.goal_body, -3, 3) * 0.5)
        env.phase = 2.0
        env.switch_skill("recover")
        assert env.phase == 0.0
        assert np.array_equal(env._observe()[81:83], env.recovery_initial_gravity)
    finally:
        env.close()


@pytest.mark.parametrize("step,bits", [(0, (0, 0)), (4, (0, 0)), (5, (1, 0)), (14, (1, 0)),
                                      (15, (0, 1)), (39, (0, 1)), (40, (1, 1)), (250, (1, 1))])
def test_recovery_phase_features_match_motion_boundaries(step, bits):
    env = SkillEnv("recover", recovery_level=7, recovery_phase_features=True)
    try:
        env.reset(seed=20000)
        env.step_count = step
        observation = env._observe()
        assert (observation[6], observation[78]) == bits
        assert observation.shape == (83,)
    finally:
        env.close()
