# Running curriculum on the Loudbox

`configs/running_v1.json` defines the `asimov_running_v1` contract. It uses the
existing 83 observations and 23 joint targets, a fixed 0.25 action scale, frozen
observation statistics, and independently updated reward statistics. Legacy
checkpoints retain their original environments.

## Current results and known limitations

The corrected `running_v1_seedfix` campaign ended incomplete after 6.46 hours;
no running policy qualified. Seed 44 reached 6,008,832 steps at 0.3 m/s and seed
45 reached 9,996,288 steps at 0.4 m/s. Both survived nominal 12-second replays
with mean speed errors below 0.008 m/s, but finished approximately 2.49 m and
2.85 m sideways from the requested path.

The October 5, 2026 replay audit identified unresolved task-definition issues:

- The reward tracks body-relative forward velocity without heading or yaw-rate
  tracking, while success checks sideways position in world coordinates.
- The early-stage success check only constrains final sideways displacement.
  Seed 45's promoted 417,792-step checkpoint walked almost a full circle and
  finished behind its start. Its stage pass does not establish straight walking.
- Nominal reset seeds produce identical deterministic rollouts. Repeated cases
  and their reported confidence intervals do not establish robustness.
- Continuation can reward further speed-error reductions after speed tracking
  already passes, without measuring the unresolved heading/path failure.

This contract remains experimental. Correct command/observation alignment,
path-based rewards and success checks, evaluation diversity, and progress
criteria before treating another campaign as evidence of reliable running.
The replay audit and measurements are stored locally under ignored
`logs/mujoco_rl/analysis_20261005/`.

## Start a campaign

Activate the CPU environment and the locally installed software renderer:

```bash
source .venv-mujoco/bin/activate
export LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m mujoco_rl.skill_campaign logs/mujoco_rl/running_v1 \
  --config mujoco_rl/configs/running_v1.json \
  --nav-checkpoint logs/mujoco_rl/skills_campaign_v1/current/accepted/nav/accepted.zip \
  --squat-checkpoint logs/mujoco_rl/skills_campaign_v1/current/accepted/squat/accepted.zip \
  --recovery-checkpoint logs/mujoco_rl/skills_campaign_v1/current/baselines/recover/bootstrap.zip
```

Choose a new directory for each campaign. Existing evidence is never overwritten.
The runner preserves all supplied baselines, compares legacy/fixed mappings, then
trains four experiments: initial action standard deviations 0.08 and 0.15, seeds
44 and 45. At most two trainers run together. Each receives one million steps;
the selected setting continues in blocks to ten million steps per seed, subject
to measured plateaus and the 72-hour campaign ceiling. Six hours are reserved for
qualification. The runner does not launch recovery or jumping.

## Curriculum and acceptance

Stages are `track_030`, `track_040`, `track_050`, `track_060`, `gait`, `brake`,
`random_half`, and `random_full`. Two consecutive development panels with at
least 80% success advance a stage. Stage changes apply on episode reset. The
nominal stages deliberately use fixed physics; their repeated cases do not
constitute robustness evidence.

Intermediate speed error, survival, flight events, and stage success decide
whether more training is useful. Acceptance still requires the full running
task on 50 development and 50 unseen randomized episodes, plus at least 18/20
continuous-state standing/running/standing transitions. Failed holdouts become
mandatory regression panels. Only `accepted/run/` contains qualified policies.

## Monitor and resume

```bash
python -m mujoco_rl.skill_campaign logs/mujoco_rl/running_v1 \
  --config mujoco_rl/configs/running_v1.json --watch --logs
```

The most recent launch is available through `logs/mujoco_rl/running_current`.
While a campaign is running, use `tmux attach -t asimov_running_v1` (select the
`live` window). The first launch
was excluded after discovering that checkpoint loading reset both trial seeds;
the corrected launch reseeds after loading and retains the original budget clock.

Campaign `status.json` lists child logs, steps, stages, evaluations, and remaining
time. Each trial saves `updates.jsonl`, `episodes.jsonl`, development reports,
videos, and immutable checkpoint bundles. `latest.json` points to a complete
bundle; keep the policy and `policy_vecnormalize.pkl` together.

Resume a trial with its saved configuration and bundle:

```bash
python -u -m mujoco_rl.skill_refine /path/to/checkpoints/000999424_final/policy.zip \
  --skill run --config /path/to/trial/config.json --output /path/to/new_trial \
  --seed 44 --dev-seed 3100100000 --steps 1000000 --hours 4
```

Optimizer, curriculum, statistics, and global step count persist. Simulator
episodes restart with distinct reset seeds. Warmup is not repeated. A mismatched
configuration is rejected. Manual continuation must respect the remaining
campaign budget; the automatic runner accounts for it centrally.

Evaluate or render with `mujoco_rl.skill_eval --skill run`; it automatically
selects the checkpoint's contract and the complete randomized task. The new
`run_supervisor.RunningSupervisor` supports `stand` and timed `run` commands.
Legacy `run_to` uses a different controller contract and rejects these policies.

## Checks

```bash
python -m pytest -q tests/test_running_curriculum.py tests/test_skill_campaign.py
```

Tests validate software behavior. A short training/resume/render check must also
succeed before sustained runs. Numerical instability stops a trial with its
last valid bundle preserved; a process exiting successfully never qualifies a
policy. All results are simulation-only.
