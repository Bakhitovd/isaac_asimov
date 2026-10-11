"""Selective reference regularization on successful, separate teacher trajectories."""

from pathlib import Path

import numpy as np
import torch

from .run_training import RunningPPO


def gaussian_reference_kl(mean, log_std, reference_mean, reference_log_std):
    """Mean per-joint KL(reference || student), without clipping teacher means."""
    return (
        log_std
        - reference_log_std
        + (reference_log_std.mul(2).exp() + (reference_mean - mean).square()) / (2 * log_std.mul(2).exp())
        - 0.5
    ).mean()


class ReferencePPO(RunningPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_reference_data", "_reference_rng"]

    def attach_reference(self, path, normalizer, weight, seed):
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Reference weight must be between zero and one")
        with np.load(Path(path), allow_pickle=False) as data:
            obs, means, log_std = data["observations"], data["means"], data["log_std"]
            if (
                obs.shape[1:] != self.observation_space.shape
                or not len(obs)
                or means.shape != (len(obs), *self.action_space.shape)
                or log_std.shape != self.action_space.shape
                or not all(np.isfinite(a).all() for a in (obs, means, log_std))
            ):
                raise ValueError("Invalid reference dataset")
            self._reference_data = tuple(
                torch.as_tensor(a, dtype=torch.float32, device=self.device)
                for a in (normalizer.normalize_obs(obs), means, log_std)
            )
        self.reference_weight = weight
        self._reference_rng = np.random.default_rng(seed)

    def train(self):
        weight = getattr(self, "reference_weight", 0.0)
        if weight and not hasattr(self, "_reference_data"):
            raise RuntimeError("Resume requires the original reference dataset")
        original_zero_grad = self.policy.optimizer.zero_grad
        original_step = self.policy.optimizer.step
        losses = []
        self.logger.record("train/reference_kl", 0.0)
        self.logger.record("train/reference_weight", weight)

        def zero_grad(*args, **kwargs):
            result = original_zero_grad(*args, **kwargs)
            if weight and self.policy.action_net.weight.requires_grad:
                obs, means, log_std = self._reference_data
                indices = self._reference_rng.integers(len(obs), size=256)
                predicted = self.policy.get_distribution(obs[indices]).distribution.mean
                loss = gaussian_reference_kl(predicted, self.policy.log_std, means[indices], log_std)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reference loss")
                # PPO calls zero_grad immediately before loss.backward(). Accumulate
                # reference gradients first; PPO then adds its gradient and clips the sum.
                # Reference frames never enter the on-policy rollout buffer.
                (weight * loss).backward()
                losses.append(float(loss.detach()))
                self.logger.record("train/reference_kl", float(np.mean(losses)))
            return result

        def step(*args, **kwargs):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):
                raise FloatingPointError("Nonfinite gradient; refusing optimizer update")
            return original_step(*args, **kwargs)

        self.policy.optimizer.zero_grad = zero_grad
        self.policy.optimizer.step = step
        try:
            super().train()
        finally:
            self.policy.optimizer.zero_grad = original_zero_grad
            self.policy.optimizer.step = original_step
        self.logger.record("train/reference_kl", float(np.mean(losses)) if losses else 0.0)
        self.logger.record("train/reference_weight", weight)
