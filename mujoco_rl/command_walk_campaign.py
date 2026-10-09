"""Run two command-walking trials, qualify by family, and publish accepted artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .command_walk import DEFAULTS, FAMILIES
from .skill_campaign import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def qualify(root, jobs, deadline, cancelled=lambda: False):
    import torch
    from .command_walk_eval import evaluate, load_pair, rank, render_episode
    torch.set_num_threads(1)
    candidates = []
    for job in jobs:
        path = Path(job['output']) / 'best.json'
        if path.is_file():
            best = json.loads(path.read_text()).get('3')
            if best and best['report']['criterion_met']:
                candidates.append(best)
    candidates.sort(key=lambda c: rank(c['report']), reverse=True)
    failed_panels = []
    def progress(family, episode, step):
        if cancelled() or time.time() >= deadline:
            raise TimeoutError('Campaign deadline reached during qualification')
        atomic_json(root / 'qualification_status.json', {'phase': 'evaluating', 'family': family,
                    'episode': episode, 'step': step, 'heartbeat': time.time()})
    for attempt, candidate in enumerate(candidates):
        checkpoint = Path(candidate['checkpoint'])
        model, norm = load_pair(checkpoint)
        try:
            regressions = [evaluate(model, norm, 3, seed, 50, progress=progress) for seed in failed_panels]
            if any(not r['criterion_met'] for r in regressions):
                atomic_json(root / f'qualification_{attempt}_regression_failure.json', {'reports': regressions})
                continue
            seed_start = 10051 + attempt * 1_000_000
            report = evaluate(model, norm, 3, seed_start, 50, progress=progress)
            atomic_json(root / f'qualification_{attempt}.json', {'source': str(checkpoint),
                        'development': candidate['report'], 'holdout': report, 'regressions': regressions})
            if not report['criterion_met']:
                failed_panels.append(seed_start)
                continue
            accepted = root / 'accepted'
            accepted.mkdir(exist_ok=False)
            shutil.copy2(checkpoint, accepted / 'accepted.zip')
            shutil.copy2(checkpoint.with_name('policy_vecnormalize.pkl'), accepted / 'accepted_vecnormalize.pkl')
            qualification = {'contract': model.command_walk_contract, 'source': str(checkpoint),
                             'development': candidate['report'], 'holdout': report,
                             'sha256': {p.name: sha256(p) for p in accepted.iterdir()}}
            atomic_json(accepted / 'qualification.json', qualification)
            for family in FAMILIES:
                if cancelled() or time.time() >= deadline:
                    raise TimeoutError('Campaign deadline reached during demo rendering')
                case = next(e for e in report['families'][family]['episodes'] if e['success'])
                render_episode(model, norm, accepted / 'demos' / f'{family}.mp4', family, 3, case['seed'])
            return accepted
        finally:
            norm.close()
    return None


def publish(accepted):
    """Publish only the qualified bundle; explicit paths keep unrelated changes out of the commit."""
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
    if branch != 'main' or staged:
        raise RuntimeError('Publication needs main with an empty Git staging area; accepted artifacts remain saved')
    destination = ROOT / 'mujoco_rl/checkpoints/command_walk_v1'
    shutil.copytree(accepted, destination)
    (destination / 'README.md').write_text(
        '# Command walking policy v1\n\n'
        'Continuous forward speed (0–0.3 m/s) and yaw rate (−0.6–0.6 rad/s), including '
        'walking turns, stationary turns, and stopping. Controls 23 joints at 50 Hz.\n\n'
        'Keep `accepted.zip` with `accepted_vecnormalize.pkl`. The versioned contract uses 85 observations. '
        'Qualification requires at least 90% success in every command family on separate development '
        'and holdout panels. See `qualification.json` for all results and `demos/` for selected replays. '
        'Results are simulation-only.\n\n'
        'Evaluate with `python -m mujoco_rl.command_walk_eval '
        'mujoco_rl/checkpoints/command_walk_v1/accepted.zip --output logs/command_walk_evaluation.json`.\n')
    subprocess.run(['git', 'add', '--', str(destination.relative_to(ROOT))], cwd=ROOT, check=True)
    subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com', 'commit',
                    '-m', 'Publish qualified command walking policy and demonstrations'], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'fork', 'main'], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('mujoco_rl/checkpoints/nav/accepted.zip'))
    parser.add_argument('--hours', type=float, default=72.)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 72:
        parser.error('Hours must be positive and at most 72')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.hours * 3600
    training_deadline = deadline - min(6., args.hours/4) * 3600
    stopped = False
    processes, handles, jobs = [], [], []
    def stop(*_):
        nonlocal stopped
        stopped = True
        for process in processes:
            if process.poll() is None:
                process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    source_dir = root / 'source_snapshot'
    source_dir.mkdir()
    for path in (ROOT/'mujoco_rl').glob('*.py'):
        shutil.copy2(path, source_dir/path.name)
    atomic_json(root/'config.json', DEFAULTS)
    status = {'state':'training','started':started,'deadline':deadline,'jobs':jobs,'accepted':None}
    try:
        for seed in (44,45):
            output = root / f'seed{seed}'
            log = root / f'seed{seed}.log'
            handle = log.open('w');handles.append(handle)
            command = [sys.executable,'-u','-m','mujoco_rl.command_walk_train',str(args.checkpoint.resolve()),
                       '--output',str(output),'--seed',str(seed),'--hours',str(args.hours),
                       '--deadline',str(training_deadline)]
            environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               MUJOCO_GL='osmesa')
            process = subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=environment)
            processes.append(process)
            jobs.append({'seed':seed,'pid':process.pid,'output':str(output),'log':str(log)})
            print(f'[launch] seed={seed} pid={process.pid} log={log}',flush=True)
        while any(p.poll() is None for p in processes):
            for process,job in zip(processes,jobs):
                job['exit_code']=process.poll()
                path=Path(job['output'])/'status.json'
                if path.exists():job['training']=json.loads(path.read_text())
            status.update(heartbeat=time.time(),elapsed_hours=(time.time()-started)/3600)
            atomic_json(root/'status.json',status)
            if time.time()>=training_deadline or stopped:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    if process.poll() is None:
                        try:process.wait(timeout=30)
                        except subprocess.TimeoutExpired:process.kill()
                break
            time.sleep(10)
        for process,job in zip(processes,jobs):
            job['exit_code']=process.wait()
            path=Path(job['output'])/'status.json'
            if path.exists():job['training']=json.loads(path.read_text())
        status['state']='qualifying' if not stopped else 'stopped'
        atomic_json(root/'status.json',status)
        if not stopped:
            accepted=qualify(root,jobs,deadline,lambda: stopped)
            if accepted:
                status.update(state='qualified',accepted=str(accepted))
                if args.publish:
                    publish(accepted)
                    status['published']=True
            else:status.update(state='review_required',reason='No candidate passed every command family')
        status.update(heartbeat=time.time(),elapsed_hours=(time.time()-started)/3600)
        atomic_json(root/'status.json',status)
        print(f"[campaign] {status['state']}",flush=True)
    except Exception as error:
        stop()
        status.update(state='failed',error=repr(error),heartbeat=time.time())
        atomic_json(root/'status.json',status)
        raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':
    main()
