"""Imitation reinforcement preserves the critic and respects PPO's KL budget."""

import copy

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_rl.recovery_ppo import RecoveryPPO
from mujoco_rl.skill_env import SkillEnv


@pytest.fixture
def trained_model():
    torch.set_num_threads(1)
    norm = VecNormalize(DummyVecEnv([lambda: SkillEnv("recover", recovery_level=0)]))
    norm.training = False
    model = RecoveryPPO("MlpPolicy", norm, n_steps=8, batch_size=8, n_epochs=1,
                        learning_rate=1e-4, target_kl=0.015, seed=42,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}})
    model.learn(16)
    observations = np.zeros((32, 83), dtype=np.float32)
    observations[:16, 0] = 1.0
    actions = np.zeros((32, 23), dtype=np.float32)
    actions[:16, 0] = 0.5
    model.set_demonstrations(observations, actions, norm, seed=42)
    try:
        yield model, norm, observations, actions
    finally:
        norm.close()


def test_imitation_improves_actor_without_changing_critic_or_exploration(trained_model):
    model, norm, observations, actions = trained_model
    critic = {name: value.clone() for name, value in model.policy.state_dict().items()
              if "value" in name}
    std = model.policy.log_std.detach().clone()
    normalized = torch.as_tensor(norm.normalize_obs(observations))
    targets = torch.as_tensor(actions)
    before = torch.nn.functional.mse_loss(model.policy.get_distribution(normalized).distribution.mean, targets)
    model._imitation_update()
    after = torch.nn.functional.mse_loss(model.policy.get_distribution(normalized).distribution.mean, targets)
    assert after < before
    assert model.logger.name_to_value["train/imitation_updates"] > 0
    assert torch.equal(std, model.policy.log_std)
    assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in critic.items())


def test_imitation_rejects_and_restores_an_excessive_update(trained_model, monkeypatch):
    model, _, _, _ = trained_model
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    before_optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    step = model.policy.optimizer.step

    def excessive_step(*args, **kwargs):
        step(*args, **kwargs)
        with torch.no_grad():
            model.policy.action_net.bias.add_(10.0)

    monkeypatch.setattr(model.policy.optimizer, "step", excessive_step)
    model._imitation_update()
    assert model.logger.name_to_value["train/imitation_rejected_updates"] == 1
    assert model.logger.name_to_value["train/imitation_updates"] == 0
    assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in before.items())
    torch.testing.assert_close(model.policy.optimizer.state_dict(), before_optimizer, rtol=0, atol=0)


def test_reference_imitation_preserves_unclipped_means(trained_model, tmp_path):
    model, norm, observations, actions = trained_model
    means = actions.copy()
    means[:, 0] = 1.4
    path = tmp_path / "reference.npz"
    np.savez(path, observations=observations, actions=actions, policy_means=means)
    model.load_demonstrations(path, norm)
    assert model.imitation_target_kind == "unclipped_policy_mean"
    assert torch.allclose(model._demo_actions[:, 0], torch.full((32,), 1.4))


def test_demonstrations_require_frozen_statistics(trained_model):
    model, norm, observations, actions = trained_model
    norm.training = True
    with pytest.raises(ValueError, match="Freeze normalization"):
        model.set_demonstrations(observations, actions, norm)


def test_imitation_of_exact_reference_means_preserves_actor(trained_model):
    model, norm, observations, _ = trained_model
    with torch.no_grad():
        means = model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(observations))).distribution.mean.numpy()
    model.set_demonstrations(observations, np.clip(means, -1, 1), norm, policy_means=means)
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model._imitation_update()
    assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("invalid", [np.full((32, 23), np.nan), np.zeros((1, 23))])
def test_invalid_reference_means_are_rejected(trained_model, invalid):
    model, norm, observations, actions = trained_model
    with pytest.raises(ValueError, match="reference policy means"):
        model.set_demonstrations(observations, actions, norm, policy_means=invalid)


def test_kl_guard_rejects_an_update_even_when_imitation_improves(trained_model, monkeypatch):
    model, _, _, _ = trained_model
    before = model.policy.action_net.bias.detach().clone()
    step = model.policy.optimizer.step
    losses = []
    mse_loss = torch.nn.functional.mse_loss

    def record_loss(*args, **kwargs):
        loss = mse_loss(*args, **kwargs)
        losses.append(float(loss.detach()))
        return loss

    def excessive_improvement(*args, **kwargs):
        step(*args, **kwargs)
        with torch.no_grad():
            model.policy.action_net.bias[0].add_(0.5)

    monkeypatch.setattr(model.policy.optimizer, "step", excessive_improvement)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", record_loss)
    model._imitation_update()
    assert losses[1] < losses[0]
    assert model.logger.name_to_value["train/imitation_rejected_updates"] == 1
    assert model.logger.name_to_value["train/imitation_updates"] == 0
    assert torch.equal(before, model.policy.action_net.bias)


def test_critic_warmup_skips_imitation(trained_model):
    model, _, _, _ = trained_model
    model.policy.action_net.weight.requires_grad_(False)
    model._imitation_update()
    assert model.logger.name_to_value.get("train/imitation_updates", 0) == 0


def test_saved_actor_remains_compatible_with_standard_ppo(trained_model, tmp_path):
    model, _, observations, _ = trained_model
    model.save(tmp_path / "actor")
    restored = RecoveryPPO.load(tmp_path / "actor", device="cpu")
    standard = PPO.load(tmp_path / "actor", device="cpu")
    assert restored._demo_observations is None
    assert np.array_equal(model.predict(observations, deterministic=True)[0],
                          standard.predict(observations, deterministic=True)[0])


def test_transition_sampling_does_not_join_separate_episodes(trained_model):
    model, norm, observations, _ = trained_model
    actions = np.zeros((32, 23), dtype=np.float32)
    actions[:8, 0] = 0.1
    actions[8:16, 0] = 0.5
    observations[:, 53:76] = actions
    model.set_demonstrations(observations, actions, norm, episode_starts=np.array([0, 16]))
    assert np.array_equal(model._demo_groups[0], [7])


def test_zero_updates_disables_imitation(trained_model):
    model, norm, observations, actions = trained_model
    model.set_demonstrations(observations, actions, norm, updates=0)
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model._imitation_update()
    assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in before.items())


def test_diagnostics_preserve_ppo_kl_and_record_actor_changes(trained_model, tmp_path):
    import json

    model, _, _, _ = trained_model
    model._diagnostics_path = tmp_path / "updates.jsonl"
    model.learn(16)
    rows = [json.loads(line) for line in model._diagnostics_path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["approx_kl"] >= 0
        assert row["ppo_kl"] >= 0
        assert row["post_imitation_kl"] >= 0
        assert row["ppo_action_change_rms"] > 0
        assert row["net_action_change_rms"] >= 0
        assert -1.00001 <= row["imitation_alignment"] <= 1.00001
    model.save(tmp_path / "diagnostic_actor")
    assert RecoveryPPO.load(tmp_path / "diagnostic_actor")._diagnostics_path is None
