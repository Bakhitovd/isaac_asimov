"""Diagnostic PPO with reversible steps bounded by full-rollout Gaussian KL."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .command_walk_v4_train import DiagnosticPPO


def identical_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(identical_tree(a, b) for a, b in zip(left, right))
    return left == right


class GuardedPPO(DiagnosticPPO):
    """Bound cumulative KL relative to the policy that collected this rollout.

    This is an experimental optimizer guard, not a guarantee of skill retention.
    Rejected proposals restore all policy parameters and Adam moments/counters.
    """

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["audit_hook", "guard_path"]

    def proposal_limits(self):
        return True, {}

    def train(self):
        raw = self.rollout_buffer.observations
        if not self.rollout_buffer.generator_ready:
            raw = self.rollout_buffer.swap_and_flatten(raw)
        observations = torch.as_tensor(raw, device=self.device)

        def distribution():
            with torch.no_grad():
                d = self.policy.get_distribution(observations).distribution
                return d.mean.detach().clone(), d.stddev.detach().clone()

        old_mean, old_std = distribution()
        original_step = self.policy.optimizer.step
        guarded = bool(self.guard_enabled and self.policy.action_net.weight.requires_grad)
        record = dict(
            training_steps=self.num_timesteps,
            enabled=guarded,
            proposals=0,
            accepted=0,
            rejected=0,
            backtracks=0,
            restoration_checks=0,
            max_accepted_mean_kl=0.0,
            max_accepted_frame_kl=0.0,
            minimum_step_factor=1.0,
        )

        def kl():
            mean, std = distribution()
            # Analytic KL(old Gaussian || new Gaussian), summed over action dimensions.
            values = (
                (torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - 0.5)
                .sum(dim=1)
                .clamp_min(0)
            )
            return float(values.mean()), float(values.max())

        def step(*args, **kwargs):
            if not guarded:
                return original_step(*args, **kwargs)
            parameters = copy.deepcopy(self.policy.state_dict())
            optimizer = copy.deepcopy(self.policy.optimizer.state_dict())
            rates = [g["lr"] for g in self.policy.optimizer.param_groups]
            for attempt in range(9):
                factor = 0.5**attempt
                for group, rate in zip(self.policy.optimizer.param_groups, rates):
                    group["lr"] = rate * factor
                record["proposals"] += 1
                original_step(*args, **kwargs)
                mean_kl, max_kl = kl()
                finite = all(torch.isfinite(p).all() for p in self.policy.parameters())
                extra_passed, _ = self.proposal_limits()
                if (
                    finite
                    and extra_passed
                    and np.isfinite(mean_kl + max_kl)
                    and mean_kl <= self.guard_mean_kl
                    and max_kl <= self.guard_max_kl
                ):
                    record["accepted"] += 1
                    record["minimum_step_factor"] = min(record["minimum_step_factor"], factor)
                    record["max_accepted_mean_kl"] = max(record["max_accepted_mean_kl"], mean_kl)
                    record["max_accepted_frame_kl"] = max(record["max_accepted_frame_kl"], max_kl)
                    for group, rate in zip(self.policy.optimizer.param_groups, rates):
                        group["lr"] = rate
                    return None
                record["rejected"] += 1
                self.policy.load_state_dict(parameters)
                self.policy.optimizer.load_state_dict(optimizer)
                if not identical_tree(parameters, self.policy.state_dict()) or not identical_tree(
                    optimizer, self.policy.optimizer.state_dict()
                ):
                    raise RuntimeError("Rejected step did not restore exact policy and optimizer state")
                record["restoration_checks"] += 1
                if attempt < 8:
                    record["backtracks"] += 1
            return None

        self.policy.optimizer.step = step
        try:
            super().train()
        finally:
            self.policy.optimizer.step = original_step
        record["final_mean_kl"], record["final_max_kl"] = kl()
        extra_passed, extra_metrics = self.proposal_limits()
        record.update(extra_metrics)
        if guarded and not extra_passed:
            raise RuntimeError("Accepted update violated reference protection")
        new_mean, _ = distribution()
        record["mean_action_change_rms"] = float((new_mean - old_mean).square().mean().sqrt())
        if guarded and (
            record["final_mean_kl"] > self.guard_mean_kl + 1e-7 or record["final_max_kl"] > self.guard_max_kl + 1e-7
        ):
            raise RuntimeError("Accepted rollout violated the analytic KL guard")
        with Path(self.guard_path).open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.audit_hook(self)
