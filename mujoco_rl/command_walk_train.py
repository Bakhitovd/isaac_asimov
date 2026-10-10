"""Train/resume the versioned command-walking policy with behavioral monitoring."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .command_walk import CONTRACT, DEFAULTS, STAGES, CommandWalkEnv
from .command_walk_eval import compact, evaluate, progress_made, rank, require_contract
from .run_training import FrozenObservationStats, RunningPPO, actor_parameters, reseed_after_loading
from .skill_campaign import atomic_json, sha256
from .skill_train import _stats_path, _warm_start


def create_model(checkpoint, seed=44, config=None):
    config = dict(config or DEFAULTS)
    if config['contract'] != CONTRACT:
        raise ValueError('Incompatible command walking contract')
    source = RunningPPO.load(checkpoint, device='cpu')
    resume = hasattr(source, 'command_walk_contract')
    factories = [lambda: Monitor(CommandWalkEnv()) for _ in range(config['workers'])]
    base = SubprocVecEnv(factories, start_method='forkserver') if config['workers'] > 1 else DummyVecEnv(factories)
    try:
        if resume:
            require_contract(source)
            if source.command_walk_contract != config:
                raise ValueError('Resume settings differ from checkpoint contract')
            normalizer = VecNormalize.load(str(_stats_path(checkpoint)), base)
            model = source
            model.set_env(normalizer)
            normalizer.env_method('set_stage', model.command_walk_state['stage'])
        else:
            if source.observation_space.shape != (83,) or source.action_space.shape != (23,):
                raise ValueError('Initialize from the published 83-input navigation policy')
            normalizer = VecNormalize(base, gamma=config['gamma'])
            model = RunningPPO('MlpPolicy', normalizer, device='cpu', seed=seed, verbose=1,
                n_steps=config['rollout_steps'], batch_size=config['batch_size'], n_epochs=config['epochs'],
                learning_rate=config['learning_rate'], gamma=config['gamma'], clip_range=config['clip_range'],
                target_kl=config['target_kl'], max_grad_norm=1.,
                policy_kwargs={'net_arch': {'pi': [256, 256, 128], 'vf': [256, 256, 128]},
                               'activation_fn': torch.nn.ELU})
            _warm_start(model, normalizer, checkpoint)
            normalizer.ret_rms = RunningMeanStd(shape=())
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.zero_()
                model.policy.log_std.fill_(np.log(config['initial_std']))
            model.command_walk_contract = config
            model.command_walk_state = {'stage': 0, 'pass_streak': 0, 'last_evaluation': 0,
                                        'best_by_stage': {}, 'last_block_steps': 0}
        normalizer.obs_rms = FrozenObservationStats.from_stats(normalizer.obs_rms)
        normalizer.training, normalizer.norm_reward, normalizer.gamma = True, True, config['gamma']
        rng_hash = reseed_after_loading(model, seed)
        return model, normalizer, resume, rng_hash
    except Exception:
        base.close()
        raise


def save_bundle(model, normalizer, output, label):
    folder = output / 'checkpoints' / f'{model.num_timesteps:09d}_{label}'
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = folder / 'policy.zip'
    model.save(checkpoint)
    normalizer.save(_stats_path(checkpoint))
    atomic_json(folder / 'contract.json', model.command_walk_contract)
    atomic_json(folder / 'state.json', model.command_walk_state)
    atomic_json(folder / 'manifest.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
    atomic_json(output / 'latest.json', {'checkpoint': str(checkpoint.resolve()), 'steps': model.num_timesteps})
    return checkpoint


class CommandMonitor(BaseCallback):
    def __init__(self, output, config, deadline, dev_seed):
        super().__init__()
        self.output, self.config, self.deadline, self.dev_seed = output, config, deadline, dev_seed
        self.stop_reason, self.stop_requested = 'step_budget', False
        self.last_heartbeat, self.last_update, self.high_kl = 0., -1, 0
        self.latest = None

    def heartbeat(self, phase='training', **extra):
        atomic_json(self.output / 'status.json', {'state': 'running', 'phase': phase, 'pid': os.getpid(),
                    'steps': self.model.num_timesteps, 'stage': STAGES[self.model.command_walk_state['stage']],
                    'heartbeat': time.time(), 'remaining_hours': max(0., self.deadline-time.time())/3600,
                    'evaluation': self.latest, **extra})

    def evaluation_progress(self, family, index, step):
        if time.time() >= self.deadline:
            self.stop_reason = 'time_budget'
        if self.stop_reason in {'interrupted', 'time_budget'}:
            raise TimeoutError('Command walking interrupted or campaign deadline reached')
        if time.time() - self.last_heartbeat > 10:
            self.last_heartbeat = time.time()
            self.heartbeat('evaluating', family=family, episode=index, episode_step=step)
            print(f'[evaluation] family={family} case={index + 1}/{self.config["development_episodes"]} '
                  f'episode_step={step} training_steps={self.model.num_timesteps}', flush=True)

    def probe(self, label):
        state = self.model.command_walk_state
        result = evaluate(self.model, self.training_env, state['stage'], self.dev_seed,
                          self.config['development_episodes'], progress=self.evaluation_progress)
        self.latest = compact(result)
        state['latest_evaluation'] = self.latest
        state['last_evaluation'] = self.model.num_timesteps
        atomic_json(self.output / 'evaluations' / f'{self.model.num_timesteps}_{label}.json', result)
        checkpoint = self.output / 'checkpoints' / f'{self.model.num_timesteps:09d}_{label}' / 'policy.zip'
        key = str(result['stage'])
        old = state['best_by_stage'].get(key)
        if old is None or rank(result) > tuple(old['rank']):
            state['best_by_stage'][key] = {'checkpoint': str(checkpoint.resolve()), 'rank': list(rank(result)),
                                           'report': self.latest}
        save_bundle(self.model, self.training_env, self.output, label)
        atomic_json(self.output / 'best.json', state['best_by_stage'])
        print(f"[probe] steps={self.model.num_timesteps} stage={STAGES[result['stage']]} "
              f"pass={ {k: round(v['pass_rate'], 3) for k,v in result['families'].items()} } "
              f"unmet_violation={result['mean_violation']:.3f}", flush=True)
        return result

    def _on_rollout_start(self):
        state = self.model.command_walk_state
        enabled = self.model.num_timesteps >= 2*self.config['workers']*self.config['rollout_steps']
        for p in actor_parameters(self.model):
            p.requires_grad_(enabled)
        if self.model._n_updates != self.last_update:
            self.last_update = self.model._n_updates
            kl = self.model.logger.name_to_value.get('train/ppo_kl', 0.)
            self.high_kl = self.high_kl + 1 if kl > .05 else 0
            if not np.isfinite(kl) or self.high_kl >= 3:
                self.stop_reason, self.stop_requested = 'unstable_updates', True
        if self.stop_requested:
            return
        if self.model.num_timesteps - state['last_evaluation'] >= self.config['evaluation_interval']:
            result = self.probe('probe')
            state['pass_streak'] = state['pass_streak'] + 1 if result['minimum_pass_rate'] >= .8 else 0
            if state['stage'] == 3 and result['criterion_met']:
                self.stop_reason, self.stop_requested = 'ready_for_qualification', True
            elif state['pass_streak'] >= 2 and state['stage'] < 3:
                state['stage'] += 1
                state['pass_streak'] = 0
                self.training_env.env_method('set_stage', state['stage'])
                print(f"[curriculum] {STAGES[state['stage']]}", flush=True)
            if self.model.num_timesteps - state['last_block_steps'] >= self.config['block_steps']:
                before = state.get('block_baseline')
                advanced = before is not None and state['stage'] > before['stage']
                improved = before is None or advanced or progress_made(before, result)
                atomic_json(self.output / 'reviews' / f'{self.model.num_timesteps}.json',
                            {'improved': improved, 'before': before, 'after': compact(result)})
                if not improved:
                    self.stop_reason, self.stop_requested = 'behavioral_plateau', True
                state['block_baseline'] = compact(result)
                state['last_block_steps'] = self.model.num_timesteps

    def _on_step(self):
        if time.time() >= self.deadline:
            self.stop_reason, self.stop_requested = 'time_budget', True
        if time.time() - self.last_heartbeat >= 10:
            self.last_heartbeat = time.time()
            self.heartbeat()
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                record = {k: info[k] for k in ('family', 'stage', 'success', 'failures', 'violation', 'episode')}
                record['steps'] = self.model.num_timesteps
                with (self.output / 'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
        return not self.stop_requested


def train(args):
    torch.set_num_threads(1)
    config = dict(DEFAULTS)
    if args.config:
        supplied = json.loads(args.config.read_text())
        if set(supplied) - set(config):
            raise ValueError('Unknown configuration fields')
        config.update(supplied)
    if args.workers is not None:
        config['workers'] = args.workers
    if args.evaluation_episodes is not None:
        config['development_episodes'] = args.evaluation_episodes
    if (config['workers'] < 1 or config['workers'] > 8 or config['development_episodes'] < 1
            or args.hours <= 0 or not np.isfinite(args.hours) or args.steps < 0):
        raise ValueError('Invalid training configuration/budget')
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'config.json', config)
    atomic_json(args.output / 'status.json', {'state': 'starting', 'pid': os.getpid(),
                'heartbeat': time.time(), 'seed': args.seed})
    model, norm, resume, rng_hash = create_model(args.checkpoint, args.seed, config)
    model._diagnostics_path = args.output / 'updates.jsonl'
    deadline = min(time.time() + args.hours*3600, args.deadline or float('inf'))
    callback = CommandMonitor(args.output, config, deadline, args.dev_seed)
    callback.init_callback(model)
    atomic_json(args.output / 'provenance.json', {'source': str(args.checkpoint.resolve()),
        'source_hash': sha256(args.checkpoint), 'normalization_hash': sha256(_stats_path(args.checkpoint)),
        'seed': args.seed, 'resumed': resume, 'rng_hash': rng_hash,
        'source_hashes': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}})
    start = model.num_timesteps
    previous_handlers = {}
    def stop(*_):
        callback.stop_reason, callback.stop_requested = 'interrupted', True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, stop)
    try:
        initial = callback.probe('initial')
        state = model.command_walk_state
        state.setdefault('block_baseline', compact(initial))
        rollout = config['workers'] * config['rollout_steps']
        # This invocation owns a new budget, independent of the checkpoint's age.
        available = min(args.steps, config['maximum_steps'])
        steps = max(0, available // rollout) * rollout
        if steps:
            model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
        for p in actor_parameters(model):
            p.requires_grad_(True)
        final = callback.probe('final')
        checkpoint = save_bundle(model, norm, args.output, 'resume')
        atomic_json(args.output / 'report.json', {'initial': compact(initial), 'final': compact(final),
            'stop_reason': callback.stop_reason, 'checkpoint': str(checkpoint.resolve()),
            'steps': model.num_timesteps, 'additional_steps': model.num_timesteps-start,
            'best_by_stage': copy.deepcopy(state['best_by_stage'])})
        atomic_json(args.output / 'status.json', {'state': 'completed', 'steps': model.num_timesteps,
            'stage': STAGES[state['stage']], 'stop_reason': callback.stop_reason, 'heartbeat': time.time()})
    except TimeoutError:
        for p in actor_parameters(model):
            p.requires_grad_(True)
        checkpoint = save_bundle(model, norm, args.output, 'interrupted')
        atomic_json(args.output / 'status.json', {'state': 'stopped', 'steps': model.num_timesteps,
                    'stop_reason': callback.stop_reason, 'checkpoint': str(checkpoint), 'heartbeat': time.time()})
    except Exception as error:
        atomic_json(args.output / 'status.json', {'state': 'failed', 'steps': model.num_timesteps,
                    'error': repr(error), 'heartbeat': time.time()})
        raise
    finally:
        norm.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--dev-seed', type=int, default=1001)
    parser.add_argument('--steps', type=int, default=10_000_000)
    parser.add_argument('--hours', type=float, default=72.)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--evaluation-episodes', type=int)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
