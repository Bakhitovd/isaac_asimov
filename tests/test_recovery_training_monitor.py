"""Curriculum progression and regression protection for recovery training."""

import json
import time
from types import SimpleNamespace

import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_rl import skill_train
from mujoco_rl.skill_env import SkillEnv


def test_comparison_reader_tolerates_an_incomplete_live_log_line(tmp_path):
    from mujoco_rl.recovery_compare import read_records

    path = tmp_path / "live.jsonl"
    path.write_text('{"steps": 1}\n{"steps":')
    assert read_records(path) == [{"steps": 1}]
    path.write_text('{"steps": 1}\n{"steps": 2}\n')
    assert read_records(path) == [{"steps": 1}, {"steps": 2}]


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    normalizer = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]))
    rates = {"probe": 0.9, "validation": 0.9, "retention": 1.0, "fresh": 0.9,
             "stress": 0.9, "stress_retention": 1.0}

    def evaluate(model, norm, skill, seed_start=10001, seeds=50, recovery_level=None):
        role = ("validation" if recovery_level == callback.recovery_level else "retention") if seed_start == 30000 else (
            "fresh" if seed_start >= 100000 else "probe")
        if seed_start == 550000:
            role = "stress" if recovery_level == callback.recovery_level else "stress_retention"
        rate = 0.0 if recovery_level is None else rates[role]
        return {"pass_rate": rate, "criterion_met": rate >= 0.8,
                "recovery_level": recovery_level,
                "seed_start": seed_start,
                "episodes": [{"seconds": 12.0} for _ in range(seeds)]}

    monkeypatch.setattr(skill_train, "evaluate", evaluate)
    logger = SimpleNamespace(name_to_value={}, record=lambda *args: None)
    model = SimpleNamespace(get_env=lambda: normalizer, save=lambda *args: None,
                            logger=logger, num_timesteps=0, lr_schedule=lambda _: 3e-5,
                            policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 3e-5}])))
    callback = skill_train.GateCallback("recover", tmp_path, time.monotonic() + 60, 100000,
                                        recovery_level=1)
    callback.model = model
    callback.num_timesteps = 25008
    try:
        yield callback, normalizer, rates
    finally:
        normalizer.close()


def test_advancement_requires_two_successes_and_retention(monitor):
    callback, normalizer, rates = monitor
    rates["retention"] = 0.5
    for _ in range(2):
        callback._probe_recovery(normalizer)
    assert callback.recovery_level == 1
    rates["retention"] = 1.0
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 1
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2


def test_advancement_requires_current_level_on_future_retention_panel(monitor):
    callback, normalizer, rates = monitor
    rates["validation"] = 0.7
    for _ in range(3):
        callback._probe_recovery(normalizer)
    assert callback.recovery_level == 1
    assert callback.curriculum_holdout_attempt == 0
    assert callback.abort_reason is None
    assert not callback.last_probe_record["recovery_current_validation"]["criterion_met"]
    rates["validation"] = 0.9
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2


def test_validation_direction_failures_guide_pose_sampling(monitor, monkeypatch):
    callback, normalizer, _ = monitor
    original = skill_train.evaluate

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        rates = dict.fromkeys(("front", "back", "left", "right"), 1.0)
        if kwargs["seed_start"] == 30000 and kwargs["recovery_level"] == callback.recovery_level:
            rates["right"] = 0.5
            result["criterion_met"] = False
        return {**result, "group_rates": rates}

    monkeypatch.setattr(skill_train, "evaluate", evaluate)
    callback._probe_recovery(normalizer)
    weights = callback.last_probe_record["next_training_pose_weights"]
    assert weights[3] > 5 * weights[0]
    assert callback.probe_pass_streak == 0


def test_best_checkpoint_tracks_validation_improvement_without_probe_gain(monitor):
    callback, normalizer, rates = monitor
    saved = []
    callback.model.save = saved.append
    rates["validation"] = 0.7
    callback._probe_recovery(normalizer)
    rates["validation"] = 0.9
    callback._probe_recovery(normalizer)
    assert len(saved) == 2
    assert callback.best_recovery_quality[0] == 1


def test_stress_panel_blocks_advancement_and_is_retained_after_passing(monitor):
    callback, normalizer, rates = monitor
    callback.recovery_stress_seed_start = 550000
    rates["stress"] = 0.7
    for _ in range(2):
        callback._probe_recovery(normalizer)
    assert callback.recovery_level == 1
    assert not hasattr(callback.model, "recovery_stress_validated_levels")
    rates["stress"] = 0.9
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2
    assert callback.model.recovery_stress_validated_levels == [1]
    rates["stress_retention"] = 0.5
    for _ in range(3):
        callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2
    assert "failed retention" in callback.abort_reason
    assert not callback.last_probe_record["recovery_stress_retention"]["criterion_met"]


def test_stress_direction_guides_training_before_it_passes(monitor, monkeypatch):
    callback, normalizer, _ = monitor
    callback.recovery_stress_seed_start = 550000
    original = skill_train.evaluate

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        groups = dict.fromkeys(("front", "back", "left", "right"), 1.0)
        if kwargs["seed_start"] == 550000:
            groups["left"] = 0.5
            result["criterion_met"] = False
        return {**result, "group_rates": groups}

    monkeypatch.setattr(skill_train, "evaluate", evaluate)
    callback._probe_recovery(normalizer)
    weights = callback.last_probe_record["next_training_pose_weights"]
    assert weights[2] > 5 * weights[0]
    assert callback.probe_pass_streak == 0


def test_repeated_success_regression_stops_run(monitor):
    callback, normalizer, rates = monitor
    callback._probe_recovery(normalizer)
    rates["probe"] = 0.5
    callback._probe_recovery(normalizer)
    assert callback.abort_reason is None
    callback._probe_recovery(normalizer)
    assert "regressed" in callback.abort_reason


@pytest.mark.parametrize("budget", [250000, 1000000])
def test_stalled_curriculum_stops_run(monitor, budget):
    callback, normalizer, rates = monitor
    callback.recovery_stall_steps = budget
    rates["probe"] = 0.5
    callback.num_timesteps = budget - 1
    callback._probe_recovery(normalizer)
    assert callback.abort_reason is None
    callback.num_timesteps = budget
    callback._probe_recovery(normalizer)
    assert "stalled" in callback.abort_reason


def test_curriculum_requires_fresh_seed_confirmation(monitor):
    callback, normalizer, rates = monitor
    rates["fresh"] = 0.5
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 1
    record = callback.last_probe_record
    assert not record["curriculum_fresh_evaluation"]["criterion_met"]
    assert callback.probe_pass_streak == 0
    rates["fresh"] = 0.9
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2
    assert callback.curriculum_holdout_attempt == 2


def test_failed_fresh_panel_guides_sampling_and_must_pass_before_retry(monitor, monkeypatch):
    callback, normalizer, rates = monitor
    original = skill_train.evaluate

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        groups = dict.fromkeys(("front", "back", "left", "right"), 1.0)
        if kwargs["seed_start"] >= 100000:
            groups["back"] = rates["fresh"]
        return {**result, "group_rates": groups}

    monkeypatch.setattr(skill_train, "evaluate", evaluate)
    rates["fresh"] = 0.5
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.model.recovery_failed_fresh_panel["recovery_level"] == 1
    weights = callback.last_probe_record["next_training_pose_weights"]
    assert weights[1] > 5 * weights[0]
    callback._probe_recovery(normalizer)
    assert callback.curriculum_holdout_attempt == 1
    assert not callback.last_probe_record["recovery_retry_validation"]["criterion_met"]
    rates["fresh"] = 0.9
    callback._probe_recovery(normalizer)
    callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2
    assert callback.model.recovery_failed_fresh_panel is None
    assert callback.model.recovery_validated_retry_panels["1"]["recovery_level"] == 1
    rates["fresh"] = 0.5
    for _ in range(3):
        callback._probe_recovery(normalizer)
    assert callback.recovery_level == 2
    assert not callback.last_probe_record["recovery_retry_retention"]["criterion_met"]
    assert "failed retention" in callback.abort_reason


def test_repeated_retention_failure_stops_run(monitor):
    callback, normalizer, rates = monitor
    rates["retention"] = 0.7
    for _ in range(2):
        callback._probe_recovery(normalizer)
    assert callback.abort_reason is None
    callback._probe_recovery(normalizer)
    assert "failed retention three times" in callback.abort_reason


def test_fresh_confirmation_seeds_are_reproducible_and_distinct_across_runs(tmp_path):
    first = skill_train._recovery_fresh_seed_start(tmp_path / "first")
    second = skill_train._recovery_fresh_seed_start(tmp_path / "second")
    assert first == skill_train._recovery_fresh_seed_start(tmp_path / "first")
    assert first != second
    assert min(first, second) >= 1_000_000


def test_full_evaluation_preserves_its_own_timestep(monitor):
    callback, normalizer, rates = monitor
    callback.num_timesteps = 75024
    callback._probe_recovery(normalizer)
    callback.num_timesteps = 100008
    assert callback._on_step()
    record = json.loads((callback.run_dir / "evaluations.jsonl").read_text())
    assert record["training_steps"] == 100008
    assert record["probe_steps"] == 75024


def test_mixed_failure_tilts_do_not_focus_on_successful_middle():
    probe = {"recovery_level": 8, "group_rates": {"back": 0.6}, "episodes": [
        {"recovery_pose": "back", "recovery_tilt_deg": tilt, "passed": passed}
        for tilt, passed in [(17.6, False), (18.5, False), (20.2, True), (20.7, True), (22.9, False)]]}
    assert skill_train._recovery_focus_ranges(probe) == {"back": (17.0, 23.0)}


def test_reward_normalization_migration_preserves_actor_and_observation_stats():
    normalizer = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]), norm_reward=False)
    try:
        model = PPO("MlpPolicy", normalizer, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, device="cpu")
        actor_weight = model.policy.action_net.weight.detach().clone()
        observation_mean = normalizer.obs_rms.mean.copy()
        with torch.no_grad():
            model.policy.value_net.weight.fill_(2.0)
            model.policy.value_net.bias.fill_(2.0)
        for parameter in model.policy.value_net.parameters():
            model.policy.optimizer.state[parameter] = {"old_state": True}
        assert skill_train._normalize_recovery_rewards(model, normalizer)
        assert normalizer.norm_reward
        assert torch.equal(actor_weight, model.policy.action_net.weight)
        assert (observation_mean == normalizer.obs_rms.mean).all()
        for parameter in model.policy.value_net.parameters():
            assert torch.count_nonzero(parameter) == 0
            assert parameter not in model.policy.optimizer.state
        with torch.no_grad():
            model.policy.value_net.bias.fill_(1.0)
        assert not skill_train._normalize_recovery_rewards(model, normalizer)
        assert model.policy.value_net.bias.item() == 1.0
    finally:
        normalizer.close()


def test_boundary_sampling_targets_weak_direction():
    probe = {"recovery_level": 5, "group_rates": {"front": 0.5, "left": 1.0},
             "episodes": [{"recovery_pose": "front", "recovery_tilt_deg": 16.0, "passed": True},
                          {"recovery_pose": "front", "recovery_tilt_deg": 18.0, "passed": False}]}
    assert skill_train._recovery_focus_ranges(probe) == {"front": (16.25, 17.75)}


def test_high_kl_reduces_an_already_low_learning_rate(monitor):
    callback, normalizer, _ = monitor
    callback.model.lr_schedule = lambda _: 1e-5
    callback.kl_samples = [0.05, 0.05]
    callback._probe_recovery(normalizer)
    assert callback.model.lr_schedule(1.0) < 1e-5
    assert callback.model.policy.optimizer.param_groups[0]["lr"] < 1e-5
    assert callback.model.learning_rate == callback.model.lr_schedule(1.0)


def test_critic_warmup_freezes_only_actor(tmp_path):
    norm = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover")]))
    try:
        model = PPO("MlpPolicy", norm, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}})
        callback = skill_train.GateCallback("recover", tmp_path, time.monotonic() + 60, 100000,
                                            recovery_critic_warmup_steps=100)
        callback.model = model
        callback._on_training_start()
        callback._on_rollout_start()
        assert not model.policy.action_net.weight.requires_grad
        assert not model.policy.log_std.requires_grad
        assert model.policy.value_net.weight.requires_grad
        callback.num_timesteps = 128
        callback._on_rollout_start()
        assert model.policy.action_net.weight.requires_grad
        assert model.policy.log_std.requires_grad
    finally:
        norm.close()
