"""Contracts for the running-first campaign; no learning claims from unit tests."""

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

from mujoco_rl.full_body_env import STANDING_POSE
from mujoco_rl.run_contract import RunningEnv, config_from, environment_for, made_progress, STAGES
from mujoco_rl.run_training import FrozenObservationStats, RunningMonitor, RunningPPO, actor_parameters, save_bundle
from mujoco_rl.run_campaign import ranking
from mujoco_rl.skill_env import SkillEnv


def small_config():
    return config_from({"workers": 1, "rollout_steps": 8, "batch_size": 8, "epochs": 1,
                        "development_episodes": 1, "evaluation_interval": 100000})


@pytest.mark.parametrize("value", [{"contract": "bad"}, {"action_scale": .35}, {"typo": 1},
                                   {"initial_std": float("nan")}, {"workers": 0}])
def test_invalid_configuration(value):
    with pytest.raises(ValueError):
        config_from(value)


def test_action_mapping_fixed_across_speeds_and_stages():
    env = RunningEnv()
    try:
        for stage in range(8):
            env.set_run_stage(stage)
            env.reset(seed=2)
            for speed in (0., .3, .6):
                env.command[0] = speed
                action = np.full(23, .3)
                np.testing.assert_allclose(env._target_from_action(action),
                    np.clip(STANDING_POSE + .25 * action, env.joint_ranges[:, 0], env.joint_ranges[:, 1]))
    finally:
        env.close()


def test_legacy_mapping_unchanged_and_contract_dispatch():
    legacy = environment_for(SimpleNamespace(), "run", randomize=False)
    modern = environment_for(SimpleNamespace(run_contract=config_from()), "run")
    try:
        legacy.reset(seed=1)
        expected = STANDING_POSE + np.r_[np.full(12, .035), np.full(11, .025)]
        np.testing.assert_allclose(legacy._target_from_action(np.full(23, .1)), expected)
        assert type(legacy) is SkillEnv
        assert isinstance(modern, RunningEnv) and modern.stage == 7
        with pytest.raises(ValueError):
            environment_for(SimpleNamespace(run_contract={"contract": "unknown"}), "run")
        with pytest.raises(ValueError):
            environment_for(SimpleNamespace(run_contract=config_from()), "nav")
    finally:
        legacy.close()
        modern.close()


def test_stage_changes_only_on_reset_and_no_rehearsal():
    env = RunningEnv()
    try:
        env.reset(seed=0)
        env.set_run_stage(6)
        assert env.stage == 0 and not env.randomize
        for seed in range(5):
            env.reset(seed=seed)
            assert env.skill == "run" and env.stage == 6 and env.randomization_strength == .5
            assert env.action_lag <= 1 and env.sensor_lag <= 1
            assert .9 <= env.kp_scale <= 1.1
            assert .9 <= env.kd_scale <= 1.1
    finally:
        env.close()


def test_stage_commands_and_flight_rewards():
    env = RunningEnv()
    try:
        for stage in range(8):
            env.set_run_stage(stage)
            env.reset(seed=0)
            env.step_count = 250
            assert env._run_command() == (.3, .4, .5, .6)[min(stage, 3)]
            env.step_count = 500
            assert (env._run_command() == 0) == (stage >= 5)
            env.flight_steps = 4
            assert env._run_flight_reward() == (0.2 if stage >= 4 else 0.)
    finally:
        env.close()


def test_reward_components_sum_and_observation_shape():
    env = RunningEnv()
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == (83,)
        for _ in range(100):
            obs, reward, done, truncated, info = env.step(np.zeros(23))
            assert reward == pytest.approx(sum(info["reward_components"].values()))
            assert np.isfinite(obs).all()
            if done or truncated:
                break
    finally:
        env.close()


def test_frozen_observations_and_independent_returns(tmp_path):
    env = VecNormalize(DummyVecEnv([lambda: RunningEnv()]), norm_reward=True)
    env.obs_rms = FrozenObservationStats.from_stats(env.obs_rms)
    before = copy.deepcopy(env.obs_rms)
    env.reset()
    for _ in range(4):
        env.step(np.zeros((1, 23)))
    np.testing.assert_array_equal(before.mean, env.obs_rms.mean)
    np.testing.assert_array_equal(before.var, env.obs_rms.var)
    assert before.count == env.obs_rms.count
    assert env.ret_rms.count > 4
    path = tmp_path / "stats.pkl"
    env.save(path)
    loaded = VecNormalize.load(path, DummyVecEnv([lambda: RunningEnv()]))
    try:
        assert isinstance(loaded.obs_rms, FrozenObservationStats)
        assert loaded.ret_rms.count == env.ret_rms.count
        loaded.training = False
        count = loaded.ret_rms.count
        loaded.reset()
        loaded.step(np.zeros((1, 23)))
        assert loaded.ret_rms.count == count
    finally:
        env.close()
        loaded.close()


def result(stage=0, success=0., survival=4., error=.3, flights=0.):
    return {"stage": stage, "summary": {"pass_rate": success, "survival_seconds": survival,
            "speed_error_m_s": error, "flight_events": flights}}


def test_progress_credits_partial_learning_but_not_noise_or_worse_survival():
    baseline = result()
    assert made_progress(baseline, result(error=.25))
    assert made_progress(baseline, result(survival=4.5))
    assert made_progress(baseline, result(stage=1))
    assert not made_progress(baseline, result(error=.25, survival=3.9))
    assert not made_progress(baseline, result(error=.29, survival=4.01))
    assert not made_progress(baseline, result(flights=1, error=.4))


def test_freeze_two_rollouts_updates_and_resume_bundle(tmp_path):
    torch.set_num_threads(1)
    config = small_config()
    env = VecNormalize(DummyVecEnv([lambda: RunningEnv(config)]))
    env.obs_rms = FrozenObservationStats.from_stats(env.obs_rms)
    model = RunningPPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, seed=4,
                       policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.run_contract = config
    model.run_state = {"stage": 0, "pass_streak": 0, "last_evaluation": 0}
    model._diagnostics_path = tmp_path / "updates.jsonl"
    callback = RunningMonitor(tmp_path, config, time.time() + 3600, 500)
    before = [p.detach().clone() for p in actor_parameters(model)]
    try:
        model.learn(16, callback=callback)
        assert all(torch.equal(b, p) for b, p in zip(before, actor_parameters(model)))
        model.learn(8, callback=callback, reset_num_timesteps=False)
        assert any(not torch.equal(b, p) for b, p in zip(before, actor_parameters(model)))
        path = save_bundle(model, env, tmp_path, "test")
        loaded = RunningPPO.load(path)
        assert loaded.run_contract == config and loaded.num_timesteps == 24
        assert loaded.run_state == model.run_state
        obs = np.zeros((1, 83), dtype=np.float32)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        assert loaded.policy.optimizer.state_dict()["state"]
    finally:
        env.close()


def test_deadline_stops_callback(tmp_path):
    callback = RunningMonitor(tmp_path, small_config(), 0, 1)
    callback.model = SimpleNamespace(num_timesteps=0, _n_updates=0,
        run_state={"stage": 0}, policy=SimpleNamespace(action_net=SimpleNamespace(weight=torch.zeros(1))))
    callback.locals = {"infos": []}
    assert not callback._on_step()
    assert callback.stop_reason == "time_budget"


def test_curriculum_requires_two_passes_and_preserves_ongoing_episode(tmp_path, monkeypatch):
    from stable_baselines3.common.logger import configure
    import mujoco_rl.run_training as training
    config = small_config()
    config["evaluation_interval"] = 8
    env = VecNormalize(DummyVecEnv([lambda: RunningEnv(config)]))
    model = RunningPPO("MlpPolicy", env, n_steps=8, batch_size=8,
                       policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.set_logger(configure(folder=None, format_strings=[]))
    model.run_state = {"stage": 0, "pass_streak": 0, "last_evaluation": 0}
    model.run_contract = config
    monitor = RunningMonitor(tmp_path, config, time.time() + 3600, 500)
    monitor.init_callback(model)
    monkeypatch.setattr(training, "save_bundle", lambda *a: None)
    monkeypatch.setattr(monitor, "videos", lambda *a: None)
    response = {**result(success=.8), "stage_name": STAGES[0], "episodes": []}
    monkeypatch.setattr(training, "evaluate_stage", lambda *a, **kw: copy.deepcopy(response))
    try:
        model.num_timesteps = 8
        monitor._on_rollout_start()
        assert model.run_state["stage"] == 0
        response["summary"]["pass_rate"] = .7
        model.num_timesteps = 16
        monitor._on_rollout_start()
        assert model.run_state["pass_streak"] == 0
        response["summary"]["pass_rate"] = .8
        for step in (24, 32):
            model.num_timesteps = step
            monitor._on_rollout_start()
        assert model.run_state["stage"] == 1
        assert env.get_attr("pending_stage") == [1] and env.get_attr("stage") == [0]
    finally:
        env.close()


def test_flight_events_require_completed_four_substep_airborne_phase(monkeypatch):
    env = RunningEnv(stage=4)
    try:
        env.reset(seed=1)
        env.step_count = 120
        calls = 0
        def contacts():
            nonlocal calls
            calls += 1
            return np.full(2, calls > 4), np.zeros(2), 0
        monkeypatch.setattr(env, "_contacts", contacts)
        env.step(np.zeros(23))
        assert env.run_flights == 0 and env.flight_steps == 4
        env.step(np.zeros(23))
        assert env.run_flights == 1 and env.flight_steps == 0
        env.step(np.zeros(23))
        assert env.run_flights == 1
    finally:
        env.close()


def test_checkpoint_loading_cannot_collapse_independent_trial_seeds(tmp_path):
    from mujoco_rl.run_training import reseed_after_loading
    from mujoco_rl.skill_train import _warm_start
    env = VecNormalize(DummyVecEnv([lambda: RunningEnv()]))
    architecture = {"net_arch": {"pi": [16], "vf": [16]}}
    source = RunningPPO("MlpPolicy", env, seed=7, n_steps=8, batch_size=8, policy_kwargs=architecture)
    path = tmp_path / "source.zip"
    source.save(path)
    env.save(tmp_path / "source_vecnormalize.pkl")
    samples, fingerprints = [], []
    try:
        for seed in (44, 45, 44):
            model = RunningPPO("MlpPolicy", env, seed=seed, n_steps=8, batch_size=8, policy_kwargs=architecture)
            _warm_start(model, env, path)
            fingerprints.append(reseed_after_loading(model, seed))
            samples.append(model.predict(np.zeros((1, 83), dtype=np.float32), deterministic=False)[0])
        assert fingerprints[0] != fingerprints[1] and fingerprints[0] == fingerprints[2]
        assert not np.array_equal(samples[0], samples[1])
        np.testing.assert_array_equal(samples[0], samples[2])
    finally:
        env.close()
