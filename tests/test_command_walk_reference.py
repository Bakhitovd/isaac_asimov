from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from mujoco_rl.command_walk import FAMILIES
from mujoco_rl.command_walk_nominal_campaign import eligible, qualified
from mujoco_rl.command_walk_reference import ReferencePPO, gaussian_reference_kl
from mujoco_rl.command_walk_robust_train import create_model


def test_reference_kl_matches_torch_and_has_correct_gradient():
    mean = torch.tensor([[0.2, -0.3]], requires_grad=True)
    log_std = torch.tensor([-1.0, -0.5], requires_grad=True)
    teacher_mean, teacher_std = torch.zeros_like(mean), torch.tensor([-0.5, -1.0])
    loss = gaussian_reference_kl(mean, log_std, teacher_mean, teacher_std)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(teacher_mean, teacher_std.exp()), torch.distributions.Normal(mean, log_std.exp())
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert mean.grad[0, 0] > 0 and mean.grad[0, 1] < 0


def report(rates, count=50):
    return dict(
        families={f: dict(pass_rate=r, count=count) for f, r in zip(FAMILIES, rates)}, minimum_pass_rate=min(rates)
    )


def test_nominal_gate_requires_improvement_and_retention():
    base = report([0.98, 0.92, 0.98, 0.84, 0.98, 1, 0.92])
    assert not eligible(base, base)
    assert eligible(base, report([0.98, 0.92, 0.98, 0.90, 0.98, 1, 0.92]))
    assert not eligible(base, report([0.98, 0.92, 0.98, 1, 0.76, 1, 0.92]))
    assert not qualified(report([1] * 7, count=10))
    assert qualified(report([0.96] * 7))


SOURCE = Path("logs/mujoco_rl/command_walk_v3/baseline/policy.zip")
DATA = SOURCE.parent / "reference.npz"


@pytest.mark.skipif(not DATA.exists(), reason="Local reference dataset unavailable")
def test_reference_training_serialization_and_zero_weight_control(tmp_path):
    torch.set_num_threads(1)
    model, norm, _, _ = create_model(SOURCE, variant="control", workers=1, rollout_steps=16, model_class=ReferencePPO)
    try:
        model.set_logger(configure(str(tmp_path), []))
        model.attach_reference(DATA, norm, 0.1, seed=44)
        obs, means, std = model._reference_data
        with torch.no_grad():
            predicted = model.policy.get_distribution(obs).distribution.mean
            torch.testing.assert_close(predicted, means, atol=2e-6, rtol=2e-6)
        parameters = [p.detach().clone() for p in model.policy.parameters()]
        original_step = model.policy.optimizer.step
        original_zero_grad = model.policy.optimizer.zero_grad
        model.learn(total_timesteps=32, reset_num_timesteps=False)
        assert model.policy.optimizer.step == original_step
        assert model.policy.optimizer.zero_grad == original_zero_grad
        assert model.logger.name_to_value["train/reference_kl"] >= -1e-7
        assert any(not torch.equal(a, b) for a, b in zip(parameters, model.policy.parameters()))
        path = tmp_path / "model.zip"
        model.save(path)
        restored = ReferencePPO.load(path)
        assert restored.reference_weight == 0.1
        assert not hasattr(restored, "_reference_data")
        with pytest.raises(RuntimeError, match="original reference"):
            restored.train()
        model.attach_reference(DATA, norm, 0, seed=44)
        model.learn(total_timesteps=16, reset_num_timesteps=False)
        assert model.logger.name_to_value["train/reference_kl"] == 0
    finally:
        norm.close()
