"""Campaign contracts: physical units, teachers, seed separation, and transitions."""

import json
from pathlib import Path

import numpy as np
import pytest

from mujoco_rl.full_body_env import PHYSICS_DT
from mujoco_rl.motion_teacher import MotionTeacher, TeacherController, template
from mujoco_rl.skill_audit import audit_contract
from mujoco_rl.skill_campaign import BUDGETS, SeedRegistry, atomic_json
from mujoco_rl.skill_env import JUMP_MIN_FLIGHT_SECONDS, SkillEnv
from mujoco_rl.skill_eval import wilson_interval
from mujoco_rl.skill_refine import TrajectoryEnv
from mujoco_rl.skill_scenarios import disturb, reset_case
from mujoco_rl.skill_supervisor import SkillSupervisor


def test_seed_ranges_and_failed_holdout_are_persistent(tmp_path):
    path = tmp_path / "seeds.json"
    registry = SeedRegistry(path)
    training = registry.reserve("training", "training", 512)
    development = registry.reserve("development", "development", 100)
    holdout = registry.reserve("holdout", "holdout", 100)
    assert training + 512 <= development < holdout
    registry.record("holdout", tmp_path / "result.json", False)
    reloaded = SeedRegistry(path)
    assert reloaded.panels["holdout"]["role"] == "development_retry"
    assert reloaded.reserve("holdout", "holdout", 100) == holdout
    reloaded.record("holdout", tmp_path / "repair.json", True)
    assert reloaded.panels["holdout"]["role"] == "development_retry"
    with pytest.raises(ValueError):
        reloaded.reserve("holdout", "training", 100)


def test_machine_budget_is_72_hours():
    assert sum(BUDGETS.values()) == 72


def test_atomic_status_never_accepts_nan(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"state": "running"})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text())["state"] == "running"


@pytest.mark.parametrize("successes,count", [(0, 50), (44, 50), (50, 50)])
def test_wilson_interval_includes_estimate(successes, count):
    low, high = wilson_interval(successes, count)
    assert low - 1e-12 <= successes / count <= high + 1e-12
    assert 0 <= low < high <= 1


def test_physics_contract():
    report = audit_contract()
    assert report["passed"], report["checks"]
    assert report["provisional"]["motor_sensor_delay_ms"] == [0, 20, 40]


@pytest.mark.parametrize("skill", ["nav", "squat", "recover", "run", "jump"])
def test_target_inverse_and_transition_preserve_state(skill):
    env = SkillEnv("recover", randomize=False, recovery_level=9)
    try:
        env.reset(seed=3)
        qpos, qvel, target = env.data.qpos.copy(), env.data.qvel.copy(), env.target.copy()
        env.switch_skill(skill)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        np.testing.assert_array_equal(env.target, target)
        for value in (-0.8, -0.1, 0, 0.1, 0.8):
            target = env._target_from_action(np.full(23, value))
            np.testing.assert_allclose(env._target_from_action(env.action_from_target(target)), target, atol=2e-7)
    finally:
        env.close()


def test_jump_gate_counts_physics_substeps():
    assert JUMP_MIN_FLIGHT_SECONDS / PHYSICS_DT == 32
    env = SkillEnv("jump", randomize=False)
    try:
        env.reset(seed=0)
        for _ in range(200):
            env.step(np.zeros(23))
        env.max_flight_steps, env.apex_rise, env.stable_steps = 8, 0.06, 100
        _, _, _, _, info = env.step(np.zeros(23))
        assert not info["success"]
        env.max_flight_steps, env.stable_steps = 32, 100
        _, _, _, _, info = env.step(np.zeros(23))
        assert info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("skill,pose", [("recover", "front"), ("recover", "back"), ("recover", "left"),
                                        ("recover", "right"), ("jump", "up"), ("jump", "forward")])
def test_teacher_roundtrip_and_finite_control(tmp_path, skill, pose):
    teacher = template(skill, pose)
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(teacher.document()))
    loaded = MotionTeacher.load(path)
    assert loaded.targets.shape == (6, 23)
    env = SkillEnv(skill, randomize=False)
    try:
        observation, _ = env.reset(seed=0)
        controller = TeacherController(loaded, env)
        for _ in range(10):
            action = controller.action(observation)
            assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            observation, *_ = env.step(action)
    finally:
        env.close()


def test_teacher_rejects_bad_timing_and_conditions():
    teacher = template("recover", "front")
    teacher.durations[0] = 700
    with pytest.raises(ValueError):
        teacher.validate()
    teacher = template("recover", "front")
    teacher.conditions[0] = {"unknown": 1}
    with pytest.raises(ValueError):
        teacher.validate()


def test_teacher_phase_conditions_require_minimum_duration():
    env = SkillEnv("recover", randomize=False, recovery_level=0)
    try:
        observation, _ = env.reset(seed=0)
        teacher = template("recover", "front")
        teacher.conditions[0] = {"min_height": 0.1}
        controller = TeacherController(teacher, env)
        controller.action(observation)
        assert controller.phase == 0
        controller.age = teacher.durations[0]
        controller.action(observation)
        assert controller.phase == 1
    finally:
        env.close()


def test_physical_disturbance_does_not_teleport():
    env = SkillEnv("nav", randomize=False)
    try:
        env.reset(seed=0)
        before = env.data.qpos.copy()
        disturb(env, "left")
        np.testing.assert_array_equal(env.data.qpos, before)
        assert np.linalg.norm(env.data.xfrc_applied[env.torso_id]) > 0
        env.step(np.zeros(23))
        assert not np.array_equal(env.data.qpos, before)
    finally:
        env.close()


def test_dynamic_recovery_has_full_horizon_and_real_motion():
    env = SkillEnv("recover", randomize=False, recovery_level=9)
    try:
        observation, info = reset_case(env, 0, "dynamic_fall", "front")
        assert info["disturbance_produced_fall"]
        assert env.step_count == 0 and env.episode_recovery_level is None
        assert np.isfinite(observation).all()
        assert not env.data.xfrc_applied.any()
        assert np.linalg.norm(env.data.qvel) > 0
    finally:
        env.close()


def test_supervisor_fall_injection_modes():
    supervisor = SkillSupervisor.__new__(SkillSupervisor)
    supervisor.env = SkillEnv("nav", randomize=False)
    supervisor.events = []
    try:
        supervisor.observation, _ = supervisor.env.reset(seed=0)
        original = supervisor.env.data.qpos.copy()
        supervisor.inject_fall("front", "dynamic")
        assert supervisor.disturbance_steps == 20
        np.testing.assert_array_equal(original, supervisor.env.data.qpos)
        supervisor.inject_fall("right", "prepared")
        assert not np.array_equal(original, supervisor.env.data.qpos)
    finally:
        supervisor.env.close()


def test_training_defaults_do_not_change_legacy_run_speed():
    env = TrajectoryEnv("run", None)
    try:
        env.reset(seed=0)
        assert env.run_training_speed == 0.6
        env.set_run_training_speed(0.3)
        assert env.run_training_speed == 0.3
        with pytest.raises(ValueError):
            env.set_run_training_speed(2)
    finally:
        env.close()


def test_observation_refresh_does_not_advance_sensor_delay():
    env = SkillEnv("nav", randomize=False)
    try:
        env.reset(seed=0)
        env.sensor_lag = 2
        before = np.asarray(env.sensor_buffer).copy()
        env.data.qvel[0] = 0.5
        env._observe(advance_sensors=False)
        np.testing.assert_array_equal(np.asarray(env.sensor_buffer), before)
        env.switch_skill("recover")
        np.testing.assert_array_equal(np.asarray(env.sensor_buffer), before)
    finally:
        env.close()


def test_reference_labels_and_episode_boundaries_are_preserved(tmp_path):
    from mujoco_rl.skill_refine import demonstration_data
    teacher = tmp_path / "teacher.npz"
    replay = tmp_path / "replay.npz"
    np.savez(teacher, observations=np.zeros((8, 83)), actions=np.zeros((8, 23)), episode_starts=[0, 4])
    np.savez(replay, observations=np.zeros((6, 83)), actions=np.ones((6, 23)),
             policy_means=np.full((6, 23), 1.5), episode_starts=[0, 3])
    data = demonstration_data(teacher, replay)
    np.testing.assert_array_equal(data["episode_starts"], [0, 4, 8, 11])
    assert np.max(data["actions"]) == 1
    assert np.max(data["policy_means"]) == 1.5
    assert data["primary_frames"] == 8


def test_campaign_initializer_freezes_matched_bundles(tmp_path):
    from mujoco_rl.skill_campaign import initialize_campaign, sha256
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "policy.zip"
    checkpoint.write_bytes(b"example-checkpoint")
    checkpoint.with_name("policy_vecnormalize.pkl").write_bytes(b"matching-statistics")
    output = tmp_path / "campaign"
    initialize_campaign(output, {"recover": checkpoint})
    registry = json.loads((output / "registry.json").read_text())
    assert registry["recover"]["status"] == "validated_tilt_balance_only"
    frozen = Path(registry["recover"]["checkpoint"])
    checkpoint.write_bytes(b"changed-source")
    assert sha256(frozen) == registry["recover"]["sha256"]["policy.zip"]
    with pytest.raises(FileExistsError):
        initialize_campaign(output, {"recover": checkpoint})


def test_backward_reset_bank_preserves_control_history(tmp_path):
    from mujoco_rl.motion_teacher import rollout
    _, trajectory = rollout(template("recover", "front"), 0, record=True)
    path = tmp_path / "states.npz"
    np.savez_compressed(path, episode_starts=np.array([0]), **trajectory)
    env = TrajectoryEnv("recover", path)
    try:
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            if not env.episode_is_rehearsal:
                break
        frame = env.step_count
        assert frame > 0
        np.testing.assert_allclose(env.data.qpos, trajectory["qpos"][frame])
        np.testing.assert_allclose(env.target, trajectory["target"][frame])
        np.testing.assert_allclose(np.asarray(env.action_buffer), trajectory["action_buffer"][frame])
        assert np.isfinite(obs).all() and obs.shape == (83,)
        env.set_backward_fraction(0)
        assert env.backward_fraction == 0
    finally:
        env.close()


def test_trial_warmup_and_repeated_kl_stop(tmp_path):
    import time
    from types import SimpleNamespace
    import torch
    from mujoco_rl.skill_refine import TrialMonitor
    policy = SimpleNamespace(mlp_extractor=SimpleNamespace(policy_net=torch.nn.Linear(83, 16)),
                             action_net=torch.nn.Linear(16, 23), log_std=torch.nn.Parameter(torch.zeros(23)))
    model = SimpleNamespace(policy=policy, _n_updates=0,
                            logger=SimpleNamespace(name_to_value={"train/ppo_kl": 0.01}))
    callback = TrialMonitor("recover", tmp_path, time.monotonic() + 60, 1000, 100,
                            {"pass_rate": 0}, interval=10000)
    callback.model = model
    callback.locals = {"infos": []}
    callback._on_training_start()
    assert not policy.action_net.weight.requires_grad
    assert not policy.log_std.requires_grad
    callback.num_timesteps = 100
    assert callback._on_step()
    assert policy.action_net.weight.requires_grad and policy.log_std.requires_grad
    model.logger.name_to_value["train/ppo_kl"] = 0.2
    for update in range(1, 4):
        model._n_updates = update
        allowed = callback._on_step()
    assert not allowed
    assert callback.stop_reason == "repeated_high_or_nonfinite_kl"


def test_floor_qualification_cannot_skip_failed_tilt_holdout(tmp_path):
    from types import SimpleNamespace
    from mujoco_rl.skill_campaign import Campaign
    report = tmp_path / "failed_tilt.json"
    report.write_text(json.dumps({"skill": "recover", "recovery_level": 9, "scenario": "standard"}))
    campaign = Campaign.__new__(Campaign)
    campaign.seeds = SimpleNamespace(panels={"known_tilt": {
        "role": "development_retry", "count": 50, "results": [{"report": str(report)}]}})
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"criterion_met": False}
    campaign.evaluate = evaluate
    assert not campaign.qualify(tmp_path / "candidate.zip", "recover", "candidate")
    assert len(calls) == 1
    assert calls[0][1]["level"] == 9
    assert calls[0][1]["reuse_panel"] == "known_tilt"
