import json
import pickle
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from mujoco_rl.command_walk import FAMILIES
from mujoco_rl.command_walk_robust import RobustCommandEnv
from mujoco_rl.command_walk_robust_eval import load_pair
from mujoco_rl.command_walk_train import save_bundle
from mujoco_rl.command_walk_v4 import (
    BRAKES,
    MotionEnv,
    Spec,
    panel,
    passes,
    retains,
    select_initialization,
    sequence_schedule,
    specialist_passes,
    teacher_for,
)
from mujoco_rl.command_walk_v4_distill import Teachers
from mujoco_rl.command_walk_v4_train import Monitor, build
from mujoco_rl.run_training import actor_parameters
from mujoco_rl.skill_train import _stats_path

SOURCE = Path("logs/mujoco_rl/command_walk_v1/seed45/checkpoints/005953536_probe/policy.zip")


def report(rates):
    return dict(
        families={k: dict(pass_rate=v, count=50) for k, v in zip(FAMILIES, rates)},
        minimum_pass_rate=min(rates),
        mean_pass_rate=np.mean(rates),
    )


def test_long_sequences_and_routing():
    for seed in range(100):
        sequence = sequence_schedule(np.random.default_rng(seed), 0.2, 0.3, long=True)
        assert sum(s[0] for s in sequence) == pytest.approx(60)
        assert all(3 - 1e-9 <= s[0] <= 8 + 1e-9 for s in sequence)
        assert {(v, w) for _, v, w in sequence} >= {
            (0.2, 0.0),
            (0.2, 0.3),
            (0.2, -0.3),
            (0.0, 0.3),
            (0.0, -0.3),
            (0.0, 0.0),
        }
    assert [teacher_for(c) for c in [(0, 0), (0.2, 0), (0.2, -0.3), (0, 0.3)]] == [
        "braking",
        "baseline",
        "arcs",
        "pivots",
    ]


def test_nominal_cases_preserve_physics_and_observations():
    for family in FAMILIES:
        a, b = MotionEnv(case=family), RobustCommandEnv(family=family, variant="control")
        try:
            x, _ = a.reset(seed=1001)
            y, _ = b.reset(seed=1001)
            np.testing.assert_array_equal(x, y)
            assert a.schedule == b.schedule
            for _ in range(5):
                x, *_ = a.step(np.zeros(23))
                y, *_ = b.step(np.zeros(23))
                np.testing.assert_array_equal(x, y)
        finally:
            a.close()
            b.close()


def test_specialist_commands_balanced_and_braking_restarts():
    env = MotionEnv(Spec(task="pivots"), training=True, gentle=True)
    try:
        env.reset(seed=44)
        signs = []
        for _ in range(30):
            env.reset()
            if env.family.endswith("pivot"):
                signs.append(np.sign(env.schedule[0][2]))
                assert abs(env.schedule[0][2]) <= 0.3
        assert abs(signs.count(1) - signs.count(-1)) <= 1
    finally:
        env.close()
    for case in BRAKES:
        e = MotionEnv(case=case)
        try:
            e.reset(seed=3)
            assert e.schedule[1][1:] == (0.0, 0.0)
            assert e.schedule[2][1] > 0
            assert e.duration == 16
        finally:
            e.close()


def test_initialization_gate_needs_both_seeds_and_prefers_preserved_tie():
    base = report([0.98, 0.92, 0.98, 0.84, 0.98, 1, 0.92])
    bad = report([0.98, 0.92, 0.98, 0.70, 0.98, 1, 0.92])
    good = dict(initial=base, final=base)
    assert select_initialization(dict(current=[good, dict(initial=base, final=bad)])) is None
    assert select_initialization(dict(current=[good, good], preserved=[good, good])) == "preserved"
    assert not retains(base, bad)
    assert not passes(base, FAMILIES)


@pytest.mark.skipif(not SOURCE.exists(), reason="Local source checkpoint unavailable")
@pytest.mark.parametrize("initialization", ["current", "preserved", "calibrated"])
def test_transfer_actor_critic_normalization_and_warmup(initialization):
    torch.set_num_threads(1)
    source, old_norm = load_pair(SOURCE)
    model, norm, _ = build(SOURCE, Spec(initialization, workers=1, rollout_steps=16), 44)
    try:
        e = MotionEnv(case="left_pivot")
        raw, _ = e.reset(seed=301001)
        e.close()
        with torch.no_grad():
            a = source.policy.get_distribution(
                torch.as_tensor(old_norm.normalize_obs(raw[None, :85]))
            ).distribution.mean
            b = model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(raw[None]))).distribution.mean
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        if initialization == "preserved":
            torch.testing.assert_close(source.policy.value_net.weight, model.policy.value_net.weight)
            np.testing.assert_array_equal(old_norm.ret_rms.var, norm.ret_rms.var)
            with torch.no_grad():
                old_value = source.policy.predict_values(torch.as_tensor(old_norm.normalize_obs(raw[None, :85])))
                new_value = model.policy.predict_values(torch.as_tensor(norm.normalize_obs(raw[None])))
                torch.testing.assert_close(old_value, new_value)
        else:
            assert torch.count_nonzero(model.policy.value_net.weight) == 0
            assert norm.ret_rms.var == 1
        assert np.all(norm.returns == 0)
        assert model.command_walk_state["warmup_until"] - model.num_timesteps == 16 * (
            20 if initialization == "calibrated" else 2
        )
    finally:
        norm.close()
        old_norm.close()


@pytest.mark.skipif(not SOURCE.exists(), reason="Local source checkpoint unavailable")
def test_calibration_freezes_actor_and_resume_preserves_state(tmp_path):
    torch.set_num_threads(1)
    spec = Spec("preserved", workers=1, rollout_steps=16)
    model, norm, _ = build(SOURCE, spec, 44)
    try:
        model.set_logger(configure(str(tmp_path), []))
        before = [p.detach().clone() for p in actor_parameters(model)]
        monitor = Monitor(tmp_path, model.num_timesteps, float("inf"), spec, cases=1, diagnostic=True)
        model.learn(total_timesteps=32, reset_num_timesteps=False, callback=monitor)
        assert all(torch.equal(a, b) for a, b in zip(before, actor_parameters(model)))
        checkpoint = save_bundle(model, norm, tmp_path, "test")
        resumed, norm2, _ = build(checkpoint, spec, 44, resume=True)
        try:
            assert resumed.num_timesteps == model.num_timesteps
            assert resumed.command_walk_state == model.command_walk_state
            assert (
                resumed.policy.optimizer.state_dict()["state"].keys()
                == model.policy.optimizer.state_dict()["state"].keys()
            )
            np.testing.assert_array_equal(resumed.get_env().ret_rms.var, norm.ret_rms.var)
        finally:
            norm2.close()
        with pytest.raises(ValueError, match="specification"):
            build(checkpoint, Spec("current", workers=1, rollout_steps=16), 44, resume=True)
    finally:
        norm.close()


@pytest.mark.skipif(not SOURCE.exists(), reason="Local source checkpoint unavailable")
def test_teacher_uses_own_normalizer_and_keeps_unclipped_mean():
    torch.set_num_threads(1)
    teachers = Teachers(dict(baseline=str(SOURCE)))
    env = MotionEnv(case="forward")
    try:
        raw, _ = env.reset(seed=1001)
        model, norm = teachers.pairs["baseline"]
        with torch.no_grad():
            expected = model.policy.get_distribution(
                torch.as_tensor(norm.normalize_obs(raw[None, :85]))
            ).distribution.mean.numpy()[0]
        np.testing.assert_array_equal(expected, teachers.mean("baseline", raw))
    finally:
        env.close()
        teachers.close()


@pytest.mark.skipif(not SOURCE.exists(), reason="Local source checkpoint unavailable")
def test_campaign_deadlines_persist_and_duplicate_launch_rejected(tmp_path, monkeypatch):
    from mujoco_rl import command_walk_v4_campaign as module

    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    args = SimpleNamespace(output=tmp_path / "campaign", baseline=SOURCE, pivot=SOURCE, resume=False, publish=False)
    campaign = module.Campaign(args)
    try:
        campaign.enter("diagnosis")
        deadline = campaign.state["deadline"]
        stage_deadline = campaign.state["stage_deadline"]
        with pytest.raises(BlockingIOError):
            module.Campaign(args)
    finally:
        campaign.lock.close()
    now[0] += 3600
    args.resume = True
    resumed = module.Campaign(args)
    try:
        resumed.enter("diagnosis")
        assert resumed.state["deadline"] == deadline
        assert resumed.state["stage_deadline"] == stage_deadline
        now[0] = stage_deadline + 1
        with pytest.raises(TimeoutError):
            resumed.progress()
    finally:
        resumed.lock.close()


def test_failed_diagnostic_gate_never_starts_specialists(tmp_path):
    from mujoco_rl.command_walk_v4_campaign import Campaign, GateFailure

    campaign = object.__new__(Campaign)
    campaign.root = tmp_path
    campaign.state = {}
    stages = []
    campaign.enter = lambda name: stages.append(name)
    campaign.jobs = lambda jobs: None
    base = report([1] * 7)
    bad = report([0.5] * 7)
    campaign.reports = lambda name: [dict(report=base), dict(report=bad)]
    with pytest.raises(GateFailure, match="All initialization"):
        campaign.run()
    assert stages == ["diagnosis"]


def test_specialist_gate_checks_every_assigned_motion_and_rehearsal():
    r = report([1] * 7)
    for case in BRAKES:
        r["families"][case] = dict(pass_rate=0.96, count=50)
    assert specialist_passes(r, "braking")
    r["families"][BRAKES[-1]]["pass_rate"] = 0.94
    assert not specialist_passes(r, "braking")
    r["families"][BRAKES[-1]]["pass_rate"] = 0.96
    r["families"]["forward"]["pass_rate"] = 0.90
    assert not specialist_passes(r, "braking")


@pytest.mark.skipif(not SOURCE.exists(), reason="Local source checkpoint unavailable")
def test_expired_job_saves_resume_without_actor_updates(tmp_path):
    from mujoco_rl.command_walk_v4_train import train

    output = tmp_path / "expired"
    args = SimpleNamespace(
        checkpoint=SOURCE,
        output=output,
        deadline=0.0,
        initialization="preserved",
        task="all",
        heading_penalty=False,
        workers=1,
        rollout_steps=16,
        seed=44,
        resume=False,
        cases=1,
        diagnostic=True,
        steps=32,
    )
    train(args)
    status = json.loads((output / "status.json").read_text())
    assert status["stop_reason"] == "time_budget"
    assert status["actor_steps"] == status["calibration_steps"] == 0
    assert Path(status["checkpoint"]).exists()
