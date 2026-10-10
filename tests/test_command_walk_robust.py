"""Behavioral regressions for v2 migration, disturbance isolation and retention."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_rl.command_walk import CommandWalkEnv
from mujoco_rl.command_walk_robust import PROFILES, RobustCommandEnv, holding_costs
from mujoco_rl.command_walk_robust_eval import evaluate
from mujoco_rl.command_walk_robust_train import RobustMonitor, create_model, new_budget, retention_passed
from mujoco_rl.command_walk_train import CommandMonitor, save_bundle
from mujoco_rl.skill_train import _stats_path

SOURCE = Path("logs/mujoco_rl/command_walk_v1/seed45/checkpoints/007311360_probe/policy.zip")


def test_nominal_dynamics_and_original_observations_preserved():
    a, b = CommandWalkEnv(stage=1, family="left_pivot"), RobustCommandEnv(family="left_pivot")
    try:
        x, _ = a.reset(seed=1001)
        y, _ = b.reset(seed=1001)
        np.testing.assert_array_equal(x, y[:85])
        for _ in range(20):
            x, *_ = a.step(np.zeros(23))
            y, *_ = b.step(np.zeros(23))
            np.testing.assert_array_equal(x, y[:85])
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    finally:
        a.close()
        b.close()


def test_physics_does_not_change_cases_and_noise_is_sampled_once():
    a, b = RobustCommandEnv(level=0, family="mixed"), RobustCommandEnv(level=12, family="mixed")
    try:
        for seed in [3, 4, 5]:
            a.reset(seed=seed)
            b.reset(seed=seed)
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            assert a.schedule == b.schedule
            x = b._observe(False)
            y = b._observe(False)
            np.testing.assert_array_equal(x, y)
        assert PROFILES[2].action_probability == 0.1 and PROFILES[2].sensor_probability == 0
    finally:
        a.close()
        b.close()


def test_anchor_latches_and_external_commands_only_slew_once():
    e = RobustCommandEnv(external=True)
    try:
        e.reset(seed=3)
        e.set_command(0.3, 0.6)
        e.step(np.zeros(23))
        np.testing.assert_allclose(e.command, [0.006, 0.024], atol=1e-7)
        e.set_command(0.0, 0.3)
        e.step(np.zeros(23))
        anchor = e.hold_odometry.copy()
        e.odometry += np.array([0.1, 0.2])
        e.set_command(0.0, 0.3)
        e._prepare_command()
        np.testing.assert_array_equal(anchor, e.hold_odometry)
        assert np.linalg.norm(e._observe(False)[-2:]) > 0.5
        e.set_command(0.2, 0.0)
        e._prepare_command()
        np.testing.assert_array_equal(e._observe(False)[-2:], 0.0)
    finally:
        e.close()


def test_drift_and_stopping_costs_match_failure_modes():
    good = holding_costs(0.02, 0.0, 0.3, [0.0, 0.3], 3.0)
    bad = holding_costs(0.36, 0.08, 0.3, [0.0, 0.3], 3.0)
    assert bad["hold_drift"] < -4 and bad["hold_drift"] < good["hold_drift"]
    assert good["stop_yaw"] == 0  # a requested pivot must not be penalized as a stop
    stopped = holding_costs(0.02, 0.08, 0.3, [0.0, 0.0], 3.0)
    assert stopped["stop_speed"] < 0 and stopped["stop_yaw"] < 0
    assert sum(holding_costs(0.5, 0.2, 0.3, [0.2, 0.3], 3.0).values()) == 0


def test_retention_checks_every_family_and_new_budget_is_additional():
    reference = {
        "families": {
            k: {"pass_rate": 0.98}
            for k in ["forward", "left_arc", "right_arc", "left_pivot", "right_pivot", "stop", "mixed"]
        }
    }
    result = copy.deepcopy(reference)
    result["families"]["left_pivot"]["pass_rate"] = 0.6
    assert not retention_passed(reference, result)
    result["families"]["left_pivot"]["pass_rate"] = 0.9
    assert retention_passed(reference, result)
    assert new_budget(500_000, 6144) == 497664


def test_plateau_does_not_cancel_final_evaluation(tmp_path):
    import time

    for cls, args in [
        (CommandMonitor, (tmp_path, {}, time.time() + 60, 1001)),
        (RobustMonitor, (tmp_path, 0, time.time() + 60)),
    ]:
        m = cls(*args)
        m.stop_requested = True
        m.stop_reason = "behavioral_plateau"
        m.last_heartbeat = time.time()
        if cls is CommandMonitor:
            m.evaluation_progress("forward", 0, 0)
        else:
            m.progress("forward", 0, 0)
    m.stop("interrupted")
    with pytest.raises(TimeoutError, match="interrupted"):
        m.progress()


@pytest.mark.skipif(not SOURCE.exists(), reason="Local campaign checkpoint required for integration checks")
def test_migration_resume_and_batched_evaluation(tmp_path):
    torch.set_num_threads(1)
    old = PPO.load(SOURCE, device="cpu")
    oldnorm = VecNormalize.load(str(_stats_path(SOURCE)), DummyVecEnv([lambda: CommandWalkEnv()]))
    model, norm, resume, rng = create_model(SOURCE, workers=1, rollout_steps=16)
    try:
        obs = np.random.default_rng(3).normal(size=(8, 85)).astype(np.float32)
        expanded = np.c_[obs, np.zeros((8, 2))].astype(np.float32)
        with torch.no_grad():
            x = old.policy.get_distribution(torch.tensor(oldnorm.normalize_obs(obs))).distribution.mean.numpy()
            y = model.policy.get_distribution(torch.tensor(norm.normalize_obs(expanded))).distribution.mean.numpy()
        np.testing.assert_allclose(x, y, atol=1e-6)
        assert model.command_walk_state["warmup_until"] > old.num_timesteps
        assert not resume and norm.obs_rms.mean.shape == (87,)
        a = evaluate(model, norm, 0, count=2, batch=1, families=("forward",))
        b = evaluate(model, norm, 0, count=2, batch=2, families=("forward",))
        assert [e["success"] for e in a["families"]["forward"]["episodes"]] == [
            e["success"] for e in b["families"]["forward"]["episodes"]
        ]
        np.testing.assert_allclose(a["mean_violation"], b["mean_violation"], atol=1e-5)
        model.command_walk_state["level"] = 4
        path = save_bundle(model, norm, tmp_path, "test")
        restored, rnorm, is_resume, rng2 = create_model(path, workers=1, rollout_steps=16, seed=45)
        try:
            assert is_resume and rng2 != rng
            assert restored.command_walk_state["level"] == 4
            assert new_budget(128, 16) == 128
        finally:
            rnorm.close()
    finally:
        norm.close()
        oldnorm.close()


def test_campaign_requires_both_improvement_and_retention():
    from mujoco_rl.command_walk import FAMILIES
    from mujoco_rl.command_walk_robust_campaign import qualifies_for_extension

    nominal = {"families": {k: {"pass_rate": 0.98} for k in FAMILIES}}
    challenge = {"minimum_pass_rate": 0.2, "mean_pass_rate": 0.5, "mean_violation": 0.1}
    reference = {"nominal": nominal, "challenge": challenge}
    improved = {**challenge, "mean_pass_rate": 0.7}
    assert qualifies_for_extension(reference, nominal, improved)
    regressed = copy.deepcopy(nominal)
    regressed["families"]["forward"]["pass_rate"] = 0.7
    assert not qualifies_for_extension(reference, regressed, improved)
    assert not qualifies_for_extension(reference, nominal, challenge)


@pytest.mark.skipif(not SOURCE.exists(), reason="Local campaign checkpoint required")
def test_time_budget_preserves_cause_and_resume_bundle(tmp_path):
    import json
    from types import SimpleNamespace

    from mujoco_rl.command_walk_robust_train import train

    output = tmp_path / "deadline"
    args = SimpleNamespace(
        checkpoint=SOURCE,
        output=output,
        variant="aligned",
        seed=44,
        workers=1,
        level=0,
        rollout_steps=16,
        steps=32,
        hours=1e-9,
        pilot=True,
        quick_count=1,
        full_count=1,
    )
    train(args)
    report = json.loads((output / "report.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert report["stop_reason"] == status["stop_reason"] == "time_budget"
    assert report["additional_steps"] == 0
    assert Path(report["checkpoint"]).is_file()
    assert _stats_path(Path(report["checkpoint"])).is_file()


@pytest.mark.skipif(not SOURCE.exists(), reason="Local campaign checkpoint required")
def test_promotion_evaluates_new_profile_before_more_training(tmp_path, monkeypatch):
    import time

    import mujoco_rl.command_walk_robust_train as trainer
    from mujoco_rl.command_walk import FAMILIES

    model, norm, _, _ = create_model(SOURCE, workers=1, rollout_steps=16, level=0)
    from stable_baselines3.common.logger import configure

    model.set_logger(configure(format_strings=[]))
    seen = []

    def fake_evaluate(model, norm, level, seed, count, **kwargs):
        seen.append(level)
        return dict(
            level=level,
            training_steps=model.num_timesteps,
            minimum_pass_rate=1.0,
            mean_pass_rate=1.0,
            mean_violation=0.0,
            criterion_met=True,
            families={k: dict(pass_rate=1.0, count=count, mean_seconds=12.0) for k in FAMILIES},
        )

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    try:
        state = model.command_walk_state
        state["last_evaluation"] = model.num_timesteps - 100_000
        state["pass_streak"] = 1
        monitor = RobustMonitor(tmp_path, model.num_timesteps, time.time() + 60)
        monitor.init_callback(model)
        monitor._on_rollout_start()
        assert state["level"] == 1
        assert seen == [0, 0, 0, 1]
        assert not state.get("needs_entry_probe", False)
        assert state["review_reference"]["level"] == 1
    finally:
        norm.close()
