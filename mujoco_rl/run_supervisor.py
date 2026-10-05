"""Continuous-state walking/standing to timed running transitions."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .run_contract import RunningEnv
from .skill_env import SkillEnv
from .skill_train import _stats_path


class RunningSupervisor:
    """Use a specialized running slot alongside the preserved navigation slot."""

    def __init__(self, navigation: Path, running: Path, seed: int):
        self.models = {name: PPO.load(path, device="cpu") for name, path in (("nav", navigation), ("run", running))}
        contract = getattr(self.models["run"], "run_contract", None)
        if contract is None:
            raise ValueError("RunningSupervisor requires a versioned running checkpoint")
        self.normalizers = {name: VecNormalize.load(str(_stats_path(path)), DummyVecEnv([lambda: SkillEnv("nav")]))
                            for name, path in (("nav", navigation), ("run", running))}
        for normalizer in self.normalizers.values():
            normalizer.training = False
            normalizer.norm_reward = False
        self.env = RunningEnv(contract, 7)
        self.observation, _ = self.env.reset(seed=seed)
        self.active, self.faulted, self.run_completed = "nav", False, False
        self.command("stand")

    def command(self, command):
        if self.faulted:
            raise RuntimeError("Transition episode failed")
        if command == "run":
            self.active = "run"
            self.env.switch_skill("run")
            self.env.speed_errors, self.env.action_clips, self.env.reward_totals = [], [], {}
            self.env.command[:] = (self.env._run_command(), 0)
        elif command == "stand":
            self.active = "nav"
            self.env.switch_skill("nav", self.env.data.qpos[:2].copy())
        else:
            raise ValueError("Supported commands: stand, run")
        self.observation = self.env._observe(advance_sensors=False)

    def step(self):
        model, normalizer = self.models[self.active], self.normalizers[self.active]
        action, _ = model.predict(normalizer.normalize_obs(self.observation[None]), deterministic=True)
        # Navigation uses the original task reward and completion semantics.
        step = self.env.step if self.active == "run" else lambda a: SkillEnv.step(self.env, a)
        self.observation, _, terminated, truncated, info = step(action[0])
        if info["fallen"] or info["termination_reason"] == "nonfinite":
            self.faulted = True
        elif self.active == "run" and info["success"]:
            self.run_completed = True
            self.command("stand")
        elif self.active == "run" and (terminated or truncated):
            self.faulted = True
        return info

    def close(self):
        self.env.close()
        for normalizer in self.normalizers.values():
            normalizer.close()


def evaluate_transitions(navigation: Path, running: Path, seed_start: int, count: int = 20):
    episodes = []
    for seed in range(seed_start, seed_start + count):
        supervisor = RunningSupervisor(navigation, running, seed)
        try:
            for _ in range(100):
                supervisor.step()
                if supervisor.faulted:
                    break
            if not supervisor.faulted:
                supervisor.command("run")
                for _ in range(600):
                    supervisor.step()
                    if supervisor.faulted or supervisor.run_completed:
                        break
            stable_steps = 0
            if supervisor.run_completed and not supervisor.faulted:
                for _ in range(100):
                    info = supervisor.step()
                    contacts, _, _ = supervisor.env._contacts()
                    stable_steps += int(not supervisor.faulted and contacts.all()
                                        and np.linalg.norm(supervisor.env.data.qvel[:2]) < .1
                                        and supervisor.env.data.qpos[2] > .55
                                        and supervisor.env.data.site_xmat[supervisor.env.imu_site_id].reshape(3, 3)[2, 2] > .96)
                    if supervisor.faulted:
                        break
            episodes.append({"seed": seed, "passed": supervisor.run_completed and not supervisor.faulted
                             and stable_steps >= 90, "standing_steps": stable_steps,
                             "run_completed": supervisor.run_completed})
        finally:
            supervisor.close()
    return {"episodes": episodes, "pass_rate": sum(e["passed"] for e in episodes) / count,
            "criterion_met": sum(e["passed"] for e in episodes) >= .9 * count}
