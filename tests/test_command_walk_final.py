"""Moderate walking contract and optimizer invariants; CPU MuJoCo only."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_rl.command_walk import FAMILIES
from mujoco_rl.command_walk_final import (
    BRAKES,
    CASES,
    QUALITY,
    FinalEnv,
    extra_costs,
    sampling,
    schedule_for,
    validate_command,
)
from mujoco_rl.command_walk_final_eval import Controller, bundle, quality_gate, retain
from mujoco_rl.command_walk_final_train import FinalPPO
from mujoco_rl.command_walk_guarded_ppo import identical_tree
from mujoco_rl.command_walk_robust_eval import load_pair
from mujoco_rl.command_walk_robust_train import create_model

SOURCE = Path("logs/mujoco_rl/command_walk_v4/baseline/policy.zip")


@pytest.mark.parametrize("v,w", [(0, 0), (0, 0.5), (0.1, -0.15), (0.3, 0.5)])
def test_supported_commands(v, w):
    validate_command(v, w)


@pytest.mark.parametrize("v,w", [(0.31, 0), (0.05, 0), (0.2, 0.6), (0.2, 0.1), (float("nan"), 0), (-0.1, 0)])
def test_rejected_commands(v, w):
    with pytest.raises(ValueError):
        validate_command(v, w)


def test_sampling_is_simulation_time_weighted():
    for phase in ("motion", "sequence"):
        names, p, fractions, durations = sampling(phase, 40)
        actual = p * durations
        np.testing.assert_allclose(actual / actual.sum(), fractions)
        assert np.isclose(sum(fractions[:7]), 0.4)


def test_all_schedules_keep_moderate_range():
    for case in CASES:
        for seconds in (20, 40, 60):
            schedule = schedule_for(case, np.random.default_rng(42), seconds)
            for _, v, w in schedule:
                validate_command(v, w)
            if case == "sequence":
                assert sum(s[0] for s in schedule) == pytest.approx(seconds)
                assert all(3 <= s[0] <= 8 for s in schedule)


def test_correction_once_and_scoring_still_nominal():
    env = FinalEnv(case="left_exit")
    try:
        obs, _ = env.reset(seed=1001)
        assert env.correction_calls == 1
        for _ in range(240):
            original = env.command.copy()
            heading = env.reference_heading
            obs, _, done, truncated, info = env.step(np.zeros(23))
            np.testing.assert_array_equal(original, [info["command_v"], info["command_w"]])
            assert env.reference_heading == pytest.approx(heading + original[1] * 0.02)
            assert env.correction_calls == env.step_count + 1
            if done or truncated:
                break
    finally:
        env.close()


def test_aligned_reward_formula():
    info = dict(
        requested_v=0,
        requested_w=0,
        age=2,
        heading_error=np.deg2rad(10),
        yaw_rate=0.1,
        command_w=0,
        lateral=0,
        displacement=0,
        speed=0.05,
    )
    costs = extra_costs(info)
    assert costs["final_heading"] == pytest.approx(-0.5)
    assert costs["final_stop"] == pytest.approx(-1.5)
    assert costs["final_yaw"] == 0
    info["age"] = 0
    assert sum(extra_costs(info).values()) == 0
    info.update(requested_v=0.2, requested_w=0, lateral=0.25)
    assert extra_costs(info)["final_lateral"] == -1


def test_reference_and_guard_restore_optimizer(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Local source checkpoint unavailable")
    import copy

    torch.set_num_threads(1)
    model, norm, _, _ = create_model(
        SOURCE,
        variant="control",
        workers=1,
        rollout_steps=32,
        model_class=FinalPPO,
        transfer=True,
        preserve_critic=True,
        warmup_rollouts=0,
        env_factory=lambda: FinalEnv(training=True),
    )
    try:
        obs = norm.reset()
        with torch.no_grad():
            mean = model.policy.get_distribution(torch.as_tensor(obs)).distribution.mean.clone()
        model.anchor_observations = torch.as_tensor(obs)
        model.anchor_means = mean
        model.guard_enabled = True
        model.guard_mean_kl = model.guard_max_kl = 0.0
        model.guard_path = tmp_path / "guard.jsonl"
        model._diagnostics_path = tmp_path / "updates.jsonl"
        model.audit_hook = lambda _: None
        model.n_epochs = 1
        with torch.no_grad():
            model.policy.log_std.add_(float(np.log(0.1)))
        parameters = copy.deepcopy(model.policy.state_dict())
        optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
        model.learn(32, reset_num_timesteps=False)
        assert identical_tree(parameters, model.policy.state_dict())
        assert identical_tree(optimizer, model.policy.optimizer.state_dict())
        import json

        record = json.loads(model.guard_path.read_text())
        assert record["rejected"] > 0
        assert record["restoration_checks"] == record["rejected"]
        assert record["accepted"] == 0
    finally:
        norm.close()


def test_bundle_reload_and_inference_parity(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Local source checkpoint unavailable")
    bundle(SOURCE, tmp_path, False, [])
    controller = Controller(tmp_path)
    model, norm = load_pair(SOURCE)
    env = FinalEnv(case="left_arc")
    try:
        corrected, _ = env.reset(seed=512)
        raw = env._observe(False)
        expected, _ = model.predict(norm.normalize_obs(corrected[None, :85]), deterministic=True)
        actual = controller.act(raw, *env.requested)
        np.testing.assert_array_equal(actual, expected[0])
        with pytest.raises(ValueError):
            controller.act(raw, 0.2, 0.6)
    finally:
        controller.close()
        norm.close()
        env.close()


def test_quality_gate_uses_paired_successes():
    q = {k: 1.0 for k in QUALITY}
    a = dict(families={"forward": dict(episodes=[dict(seed=1, success=True, quality=q)])})
    b = dict(families={"forward": dict(episodes=[dict(seed=1, success=True, quality={k: 1.2 for k in QUALITY})])})
    assert not quality_gate(a, b)["passed"]
    assert quality_gate(a, a)["passed"]


def test_retention_checks_every_family():
    a = dict(families={k: dict(pass_rate=1.0) for k in FAMILIES})
    b = dict(families={k: dict(pass_rate=0.96) for k in FAMILIES})
    assert retain(a, b)
    b["families"]["left_arc"]["pass_rate"] = 0.94
    assert not retain(a, b)


def test_campaign_sequence_block_paths_are_distinct(tmp_path):
    from mujoco_rl.command_walk_final_campaign import Campaign

    c = Campaign.__new__(Campaign)
    c.root = tmp_path
    c.anchor = tmp_path / "reference.npz"
    c.state = dict(stage="sequence", round=0)
    c.manifest = dict(stage_deadlines=dict(sequence=123))
    a = c.make_job("seed44", "policy.zip", "original", 44, 17, False, "sequence", 20)
    b = c.make_job("seed44", "policy.zip", "original", 44, 17, False, "sequence", 40)
    assert a[1] != b[1]


def test_pilot_requires_both_training_seeds(tmp_path):
    from mujoco_rl.command_walk_final_campaign import REGRESSION_CASES, Campaign, subset
    from mujoco_rl.skill_campaign import atomic_json

    def report(rate):
        data = dict(families={k: dict(pass_rate=rate, mean_violation=1 - rate) for k in REGRESSION_CASES})
        return subset(data, REGRESSION_CASES)

    c = Campaign.__new__(Campaign)
    c.root = tmp_path
    c.state = dict(stage="pilot", round=2, arms={})
    for reward in ("original", "aligned"):
        for seed in (44, 45):
            rate = 0.94 if reward == "original" or seed == 44 else 0.7
            c.state["arms"][f"{reward}_{seed}"] = dict(
                checkpoint=f"{reward}_{seed}.zip",
                seed=seed,
                reward=reward,
                development=report(rate),
                confirmation=report(rate),
                technical_passed=True,
            )
    atomic_json(tmp_path / "evaluations/baseline_development.json", report(0.8))
    atomic_json(tmp_path / "evaluations/baseline_regression.json", report(0.8))
    c.pilot()
    assert c.state["stage"] == "motion"
    assert set(c.state["arms"]) == {"original_44", "original_45"}
