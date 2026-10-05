"""Physics demonstration actions and neural actor fitting."""

import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_rl import recovery_bootstrap
from mujoco_rl import recovery_search
from mujoco_rl.recovery_bootstrap import CONTROL_JOINTS, demonstration_action, fit_actor, load_controls
from mujoco_rl.skill_env import SkillEnv


def test_preview_collection_keeps_all_directions_at_every_rehearsal_level(monkeypatch):
    cases = []

    class Environment:
        recovery_tilt_deg = 20.0

        def __init__(self, *args, **kwargs):
            pass

        def set_recovery_level(self, level):
            self.level = level

        def reset(self, seed):
            self.recovery_pose = ("front", "back", "left", "right")[seed % 4]
            cases.append((self.level, self.recovery_pose))
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(83, dtype=np.float32), 0.0, True, False, {"success": True}

        def close(self):
            pass

    monkeypatch.setattr(recovery_bootstrap, "SkillEnv", Environment)
    recovery_bootstrap.collect_demonstrations(np.full((3, 6), 0.5), 8, 160, 0, preview_next_level=True)
    for level in range(10):
        assert {pose for difficulty, pose in cases if difficulty == level} == {"front", "back", "left", "right"}


def test_demonstration_phases_and_neutral_standing():
    controls = np.full((3, 6), 0.25)
    controls[1] = 0.5
    controls[2] = -0.25
    for step, phase in ((0, 0), (4, 0), (5, 1), (14, 1), (15, 2), (39, 2)):
        action = demonstration_action(controls, step, True)
        assert np.array_equal(action[CONTROL_JOINTS], controls[phase])
        assert np.count_nonzero(action) == 6
    assert not demonstration_action(controls, 40, True).any()
    assert not demonstration_action(controls, 0, False).any()


def test_demonstration_supports_all_joint_controls():
    controls = np.zeros((3, 23), dtype=np.float32)
    controls[1, [1, 3, 5, 7, 9, 11]] = 0.25
    assert np.array_equal(demonstration_action(controls, 6, True), controls[1])
    assert not demonstration_action(controls, 40, True).any()


def test_custom_phase_timing_keeps_second_phase_longer():
    controls = np.zeros((3, 23), dtype=np.float32)
    controls[1] = 0.25
    controls[2] = -0.25
    for step, phase in ((4, 0), (5, 1), (19, 1), (20, 2), (39, 2)):
        assert np.array_equal(demonstration_action(controls, step, True, phase_ends=(5, 20, 40)), controls[phase])
    assert not demonstration_action(controls, 40, True, phase_ends=(5, 20, 40)).any()


@pytest.mark.parametrize("ends", [(5, 5, 40), (0, 15, 40), (5, 15, 601), (5, 15), (5, 15.0, 40)])
def test_control_manifest_rejects_invalid_phase_timing(tmp_path, ends):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps([{"pose": "right", "min_tilt_deg": 22, "max_tilt_deg": 25,
                                "controls": np.zeros((3, 23)).tolist(), "phase_ends": ends}]))
    with pytest.raises(ValueError, match="Phase ends"):
        load_controls(path)


@pytest.mark.parametrize("tilt,width,valid", [(28, 2, True), (30, 0, True), (30, 0.1, False), (31, 0, False)])
def test_balance_search_supports_next_curriculum_range(monkeypatch, tmp_path, tilt, width, valid):
    monkeypatch.setattr("sys.argv", ["recovery_search", "--output", str(tmp_path / "candidate.npy"),
                                    "--tilt", str(tilt), "--tilt-width", str(width),
                                    "--initial-controls", str(tmp_path / "missing.npy")])
    if valid:
        # Validation reaches the input load without starting a physics search.
        with pytest.raises(FileNotFoundError):
            recovery_search.main()
    else:
        with pytest.raises(SystemExit) as error:
            recovery_search.main()
        assert error.value.code == 2


def test_feedback_uses_observed_tilt_and_gyro_with_joint_limits():
    controls = np.zeros((3, 23), dtype=np.float32)
    observation = np.zeros(83, dtype=np.float32)
    observation[3] = np.sin(np.deg2rad(10.0))
    observation[5] = -np.cos(np.deg2rad(10.0))
    observation[1] = 0.1
    gains = np.array([2.0, 0.5, 0.5, 0.1])
    action = demonstration_action(controls, 50, True, gains, observation)
    assert np.isclose(action[0], 2.0 * np.deg2rad(10.0) + 0.2)
    assert action[0] == -action[6]
    assert action[4] == -action[10]
    assert not demonstration_action(controls, 50, False, gains, observation).any()
    clipped = demonstration_action(controls, 50, True, np.full(4, 4.0), observation)
    assert np.max(np.abs(clipped)) <= 1.0


@pytest.mark.parametrize("pose,index", [("front", 79), ("right", 80)])
def test_velocity_feedback_uses_body_speed_without_changing_legacy_gains(pose, index):
    controls = np.zeros((3, 23), dtype=np.float32)
    observation = np.zeros(83, dtype=np.float32)
    observation[5] = -1.0
    observation[index] = 0.15
    gains = np.array([0.0, 0.0, 0.0, 0.0, 0.5, -0.25])
    action = demonstration_action(controls, 50, True, gains, observation, pose)
    hip_ids = [1, 7] if pose == "right" else [0, 6]
    ankle_ids = [5, 11] if pose == "right" else [4, 10]
    signs = [1, 1] if pose == "right" else [1, -1]
    assert np.allclose(action[hip_ids], np.asarray(signs) * 0.15)
    assert np.allclose(action[ankle_ids], np.asarray(signs) * -0.075)
    assert not demonstration_action(controls, 50, True, gains[:4], observation, pose).any()


@pytest.mark.parametrize("tilt,width,expected_level", [(24, 1, 6), (25, 0, 6), (28, 2, 10)])
def test_search_uses_matching_large_angle_termination(monkeypatch, tilt, width, expected_level):
    levels = []

    class Environment:
        def __init__(self, *args, **kwargs):
            levels.append(kwargs["recovery_level"])

    with monkeypatch.context() as patch:
        patch.setattr(recovery_search, "SkillEnv", Environment)
        for name in ("_environment", "_tilt", "_pose", "_randomized_seeds", "_joint_ids",
                     "_feedback", "_tilt_width", "_rollout_seed_start", "_tilt_samples_per_seed"):
            patch.setattr(recovery_search, name, getattr(recovery_search, name))
        recovery_search._initialize(tilt, "left", 8, list(range(23)), True, width)
    assert levels == [expected_level]


def test_mixed_replay_keeps_physical_actions_bounded_and_reference_means_intact():
    dataset = {"observations": np.zeros((2, 83), np.float32), "actions": np.full((2, 23), 1.4, np.float32),
               "episode_starts": np.array([0])}
    arrays = recovery_bootstrap._demonstration_arrays(dataset)
    assert np.max(arrays["actions"]) == 1.0
    assert np.allclose(arrays["policy_means"], 1.4)
    assert np.allclose(dataset["actions"], 1.4)


def test_actor_fitting_preserves_critic_and_exploration():
    torch.set_num_threads(1)
    norm = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]))
    try:
        model = PPO("MlpPolicy", norm, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=42)
        critic = model.policy.value_net.weight.detach().clone()
        exploration = model.policy.log_std.detach().clone()
        observations = np.zeros((2, 83), dtype=np.float32)
        observations[0, 0] = 1.0
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[0, 0] = 0.5
        report = fit_actor(model, norm, {"observations": observations, "actions": actions}, 500, 42)
        assert report["final_action_mse"] < 0.004
        assert torch.equal(critic, model.policy.value_net.weight)
        assert torch.equal(exploration, model.policy.log_std)
        first_layer = next(layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear))
        assert torch.count_nonzero(first_layer.weight[:, 53:76]) == 0
        assert report["critical_frames"] > 0
    finally:
        norm.close()


def test_search_metadata_preserves_pose_and_feedback(tmp_path):
    path = tmp_path / "back.npy"
    np.save(path, np.zeros((3, 23)))
    path.with_suffix(".json").write_text(json.dumps({"pose": "back", "feedback": [0.1, 0.2, 0.3, 0.4]}))
    controls = load_controls(path)
    assert controls[0]["pose"] == "back"
    assert controls[0]["feedback"] == [0.1, 0.2, 0.3, 0.4]
    assert controls[0]["controls"].shape == (3, 23)


def test_actor_repair_can_preserve_motor_history():
    torch.set_num_threads(1)
    norm = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]))
    try:
        model = PPO("MlpPolicy", norm, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=42)
        observations = np.zeros((2, 83), dtype=np.float32)
        observations[0, 53] = 0.4
        actions = np.zeros((2, 23), dtype=np.float32)
        actions[0, 0] = 0.5
        report = fit_actor(model, norm, {"observations": observations, "actions": actions}, 1, 42,
                           preserve_motor_history=True)
        first = next(layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear))
        assert torch.count_nonzero(first.weight[:, 53:76]) > 0
        assert report["ignore_previous_action_during_bootstrap"] is False
    finally:
        norm.close()


def test_search_metadata_json_is_not_a_control_manifest(tmp_path):
    path = tmp_path / "search.json"
    path.write_text(json.dumps({"pose": "front", "feedback": [0, 0, 0, 0]}))
    with pytest.raises(ValueError, match="search .npy"):
        load_controls(path)


def test_collection_labels_clean_actions_when_execution_is_perturbed(monkeypatch):
    executed = []

    class Environment:
        recovery_pose = "front"
        recovery_tilt_deg = 20.0

        def __init__(self, *args, **kwargs):
            pass

        def set_recovery_level(self, level):
            pass

        def reset(self, seed):
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            executed.append(action.copy())
            return np.zeros(83, dtype=np.float32), 0.0, True, False, {"success": True}

        def close(self):
            pass

    monkeypatch.setattr(recovery_bootstrap, "SkillEnv", Environment)
    dataset = recovery_bootstrap.collect_demonstrations(np.full((3, 6), 0.5), 6, 8, 123, action_noise=0.03)
    expected = demonstration_action(np.full((3, 6), 0.5), 0, True)
    assert np.all(dataset["actions"] == expected)
    assert not np.allclose(executed[0], expected)
    assert np.max(np.abs(executed)) <= 1.0


@pytest.mark.parametrize("noise", [-0.01, 0.21, float("nan")])
def test_collection_rejects_invalid_action_noise(noise):
    with pytest.raises(ValueError, match="action noise"):
        recovery_bootstrap.collect_demonstrations(np.zeros((3, 6)), 6, 8, 123, action_noise=noise)


def test_learner_states_keep_expert_labels_even_after_failure(monkeypatch):
    from types import SimpleNamespace

    executed = []

    class Environment:
        recovery_pose = "front"
        recovery_tilt_deg = 20.0

        def __init__(self, *args, **kwargs):
            pass

        def set_recovery_level(self, level):
            pass

        def reset(self, seed):
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            executed.append(action.copy())
            return np.zeros(83, dtype=np.float32), 0.0, True, False, {"success": False}

        def close(self):
            pass

    monkeypatch.setattr(recovery_bootstrap, "SkillEnv", Environment)
    learner = SimpleNamespace(predict=lambda *args, **kwargs: (np.full((1, 23), -0.5), None))
    norm = SimpleNamespace(normalize_obs=lambda obs: obs)
    dataset = recovery_bootstrap.collect_demonstrations(np.full((3, 6), 0.5), 8, 8, 123,
                                                        learner=learner, normalizer=norm,
                                                        learner_fraction=0.8, keep_failed=True)
    assert dataset["successes"] == 0
    assert dataset["retained_episodes"] == 8
    assert np.allclose(dataset["actions"][:, CONTROL_JOINTS], 0.5)
    assert np.allclose(np.asarray(executed)[:, CONTROL_JOINTS], -0.3)


def test_learner_state_collection_requires_a_policy():
    with pytest.raises(ValueError, match="Learner-state"):
        recovery_bootstrap.collect_demonstrations(np.zeros((3, 6)), 8, 8, 123, learner_fraction=0.8)


def test_collection_can_target_selected_directions(monkeypatch):
    resets = []

    class Environment:
        recovery_tilt_deg = 20.0

        def __init__(self, *args, **kwargs):
            pass

        def set_recovery_level(self, level):
            pass

        def reset(self, seed, options):
            resets.append((seed, options["recovery_pose"]))
            self.recovery_pose = options["recovery_pose"]
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(83, dtype=np.float32), 0.0, True, False, {"success": True}

        def close(self):
            pass

    monkeypatch.setattr(recovery_bootstrap, "SkillEnv", Environment)
    controls = [{"pose": pose, "min_tilt_deg": 0, "max_tilt_deg": 25,
                 "controls": np.full((3, 23), 0.2)} for pose in ("left", "right")]
    recovery_bootstrap.collect_demonstrations(controls, 8, 8, 600000, collection_poses=("left", "right"))
    assert resets == [(600000 + index, "left" if index % 2 == 0 else "right") for index in range(8)]


@pytest.mark.parametrize("tilts_per_seed", [1, 3])
def test_search_uses_separate_physics_seeds_and_tilt_range(monkeypatch, tilts_per_seed):
    resets = []

    class Environment:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed, options):
            resets.append((seed, options["recovery_tilt_deg"]))
            return np.zeros(83, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(83, dtype=np.float32), 0.0, True, False, {"success": False}

    with monkeypatch.context() as patch:
        patch.setattr(recovery_search, "SkillEnv", Environment)
        for name in ("_environment", "_tilt", "_pose", "_randomized_seeds", "_joint_ids",
                     "_feedback", "_tilt_width", "_rollout_seed_start", "_tilt_samples_per_seed"):
            patch.setattr(recovery_search, name, getattr(recovery_search, name))
        recovery_search._initialize(19.5, "front", 4, CONTROL_JOINTS.tolist(), False, 1.5, 71000, tilts_per_seed)
        assert recovery_search._trial(np.zeros(18)) == (0.0, 0.0)
    if tilts_per_seed == 1:
        assert resets == [(71000, 18.0), (71004, 19.0), (71008, 20.0), (71012, 21.0)]
    else:
        for seed in [71000, 71004, 71008, 71012]:
            assert {tilt for s, tilt in resets if s == seed} == {18.0, 19.5, 21.0}


def test_controller_search_ranks_success_before_reward():
    results = [(500.0, 0.50), (300.0, 0.90), (350.0, 0.90), (600.0, 0.0)]
    assert recovery_search._candidate_order(results).tolist() == [3, 0, 1, 2]


def test_fitting_samples_both_sides_of_command_transitions():
    torch.set_num_threads(1)
    norm = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]))
    try:
        model = PPO("MlpPolicy", norm, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=42)
        observations = np.zeros((5, 83), dtype=np.float32)
        observations[:4, 53] = 0.5
        actions = np.zeros((5, 23), dtype=np.float32)
        actions[:3, 0] = 0.5
        report = fit_actor(model, norm, {"observations": observations, "actions": actions}, 1, 42)
        assert report["critical_frames"] == 2  # final old-command frame and first new-command frame
    finally:
        norm.close()
