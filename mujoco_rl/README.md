# CPU MuJoCo forward walking

## Download trained walking and turning models

The [moderate walking candidate package](checkpoints/moderate_walk_v1/README.md)
contains all four trained pilots, matching normalization, controller settings,
and complete evaluation records. [Download the bundle](checkpoints/moderate_walk_v1_bundle.zip?raw=true).
The seed-44 candidates improved some transitions, but the campaign did not pass
its replicated improvement gate. All candidates are explicitly unqualified.

## Simulation demos

The [demo collection](demos/2026-10-05/README.md) includes walking, waypoint
navigation, squatting, and prepared-lean balance recovery, with evaluation
records and a downloadable ZIP. These are selected simulation replays of
separate trained policies.

The [qualified waypoint navigation policy](checkpoints/nav/README.md) and its
matching normalization statistics are included in Git under `checkpoints/nav/`.

The [continuous walking and turning guide](COMMAND_WALK.md) covers training one
policy to walk forward, steer left/right, turn on the spot, and stop on command.

## Full-body standing and walking

The new `full_body_*` modules use the pinned `third_party/asimov-1` MuJoCo
model. They control 23 joints (legs, waist, and arms) and train a policy from
random weights to stand, then walk at 0.2 m/s. The older 12-joint experiment
described below remains available for comparison. A trained policy, preview,
and 100-episode evaluation record are in
[`checkpoints/full_body_v1/`](checkpoints/full_body_v1/README.md).

Initialize the robot asset and use the existing CPU environment:

```bash
git submodule update --init third_party/asimov-1
.venv-mujoco/bin/python -m mujoco_rl.full_body_train --benchmark-only
.venv-mujoco/bin/python -m mujoco_rl.full_body_train \
    --stage pipeline --hours 12 --run-dir logs/mujoco_rl/full_body_v1
```

The pipeline saves a standing checkpoint, trains walking only after the
standing gate passes, and stops after the requested wall-clock budget or when
the walking gate passes. Each stage saves checkpoints and matching
`*_vecnormalize.pkl` observation statistics; both files are needed to replay
or resume. To refine from a checkpoint:

```bash
.venv-mujoco/bin/python -m mujoco_rl.full_body_train \
    --stage walk --resume logs/mujoco_rl/full_body_v1/stand/best.zip \
    --hours 8 --workers 24 --learning-rate 0.0001 \
    --run-dir logs/mujoco_rl/full_body_refinement/walk
```

The stand gate requires at least 45 of 50 ten-second episodes to stay upright
with little drift or foot slip. The walk gate requires at least 40 of 50 to
travel 1.5–2.5 m in ten seconds, drift less than 0.3 m laterally, stay below
0.05 m/s mean stance-foot slip, and avoid rapid contact chatter. Each run
stores the exact environment, training, and evaluation source files under
`source_snapshot/` to make the reward settings reviewable. `--mirror` reflects
half of training episodes across the robot's left-right axis while retaining
normal-frame evaluation. Use `--seed-start 10051 --seeds 50` for a separate
evaluation set after selecting a checkpoint.

To evaluate or render a checkpoint:

```bash
.venv-mujoco/bin/python -m mujoco_rl.full_body_eval \
    logs/mujoco_rl/full_body_v1/walk/best.zip --task walk --seeds 50
LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu \
    .venv-mujoco/bin/python -m mujoco_rl.full_body_eval \
    logs/mujoco_rl/full_body_v1/walk/best.zip --task walk --seeds 50 \
    --video logs/mujoco_rl/full_body_v1/walk/replay.mp4
```

The policy receives noisy, delayed simulated gyro, gravity direction, joint
positions and velocities, the speed command, and its previous action. Exact
simulator velocity and contacts are used for rewards and evaluation only.
Motor gains and torque limits start from `isaac_asimov` and the pinned robot
URDF; they have not been calibrated against physical hardware. The current
robot model has fixed neck joints and no actuated fingers.

## Earlier 12-joint baseline

This package trains a 12-leg-joint Asimov policy from random weights on the
Loudbox CPU. It uses a fixed `(vx, vy, yaw_rate) = (0.35, 0, 0)` command and
the pinned Menlo walking reference as a small reward term. Isaac Sim is not
needed. The first stage learns sustained forward motion; the second stage
refines speed to the command.

From the repository root:

```bash
python3 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install -r mujoco_rl/requirements.txt
.venv-mujoco/bin/python -m mujoco_rl.evaluate \
    mujoco_rl/checkpoints/forward_v1/policy.zip --seed 10001
```

The validated [checkpoint and preview](checkpoints/forward_v1/README.md)
are included in Git. To reproduce training from random weights:

```bash
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 8
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 16
.venv-mujoco/bin/python -m mujoco_rl.train --benchmark-only --workers 24
.venv-mujoco/bin/python -m mujoco_rl.train --workers 24 \
    --steps 750000 --reward-stage progress \
    --run-dir logs/mujoco_rl/forward_stage1
.venv-mujoco/bin/python -m mujoco_rl.train --workers 24 \
    --steps 10000000 --learning-rate 0.0001 --reward-stage target \
    --resume logs/mujoco_rl/forward_stage1/checkpoint_750024.zip \
    --run-dir logs/mujoco_rl/forward_stage2
```

To record a new replay on this Loudbox:

```bash
LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu \
    .venv-mujoco/bin/python -m mujoco_rl.evaluate \
    mujoco_rl/checkpoints/forward_v1/policy.zip --seed 10001 \
    --video logs/mujoco_rl/forward_replay.mp4
```

The training run checks five deterministic 10-second replays every 250,000
steps. The target stage stops when all five stay upright, travel 2–4.5 m
forward, and drift at most 0.75 m sideways. It saves periodic checkpoints,
normalization statistics, metrics, and a best checkpoint in the chosen
`--run-dir`. To resume an interrupted run, pass the checkpoint ZIP to
`--resume` with the same `--run-dir`, `--workers`, and `--reward-stage`. The
matching `_vecnormalize.pkl` is required.

The run completed on this Loudbox at 1,000,056 total environment steps. Its
five target-stage replays traveled 3.07–3.17 m forward and at most 0.06 m
sideways in 10 seconds, without a fall. The trained checkpoint is at
`logs/mujoco_rl/forward_v5/best.zip`; the first-stage checkpoint is at
`logs/mujoco_rl/forward_v4/checkpoint_750024.zip`.

This Loudbox has no system OpenGL libraries. To render headlessly without
administrator access, `libosmesa6`, `libglapi-mesa`, and `libllvm15` were
downloaded as Ubuntu packages and extracted under
`/home/ailevate/.local/mujoco-osmesa`; the video command uses that path.
On a machine with system OSMesa, omit `LD_LIBRARY_PATH`.

The asset origin and license are in `assets/asimov/SOURCE.md`. Only the actor
is needed to command the simulated robot after training; the critic is used
during PPO training.
The actor observes simulator linear velocity, heading, and lateral position.
Those signals will need an estimator or a changed observation design before
real-robot deployment. This milestone uses a flat, fixed-friction simulation.

See [Command walking v2](COMMAND_WALK_V2.md) for the delay curriculum, pivot rewards, retention checks, and controlled pilot campaign.

See [Command walking v3](COMMAND_WALK_V3.md) for nominal-only walking/turning/stopping experiments with selective reference-policy protection and separate robustness qualification.

See [Command walking v4](COMMAND_WALK_V4.md) for the 24-hour initialization diagnosis, specialist training, distillation, transition refinement, and nominal qualification campaign.

### Frozen-policy steering and exploration diagnostic

`command_walk_correction` compares bounded speed/yaw input corrections around the saved
whole-body policy, then measures action-noise scales. It updates no network weights and
keeps the original requested trajectories and physical scoring. Tuning cases are separate
from the paired 50-case development/confirmation panels and 60-second sequences.

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-mujoco/bin/python -u -m mujoco_rl.command_walk_correction \
  logs/mujoco_rl/command_walk_v4/baseline/policy.zip \
  logs/mujoco_rl/command_walk_correction_v1 --workers 8 --minutes 90 \
  --expected-baseline logs/mujoco_rl/command_walk_v4/diagnostic_preserved_44/attempt_0/report.json
```

Use an unused output directory. Configuration, source snapshots, checkpoint/statistics hashes,
case reports, paired comparisons, and live `status.json` are saved there. Results are diagnostic;
completion does not publish or qualify a policy. Reviewed findings belong in the
[experiment log](COMMAND_WALK_EXPERIMENT_LOG.md).

Replay a paired case with annotated videos and a physical-trajectory plot:

```bash
MUJOCO_GL=osmesa \
LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
  .venv-mujoco/bin/python -m mujoco_rl.command_walk_correction_replay \
  logs/mujoco_rl/command_walk_correction_v1 --case left_pivot
```

The default selects the first development case improved by the correction, when available.
Use `--seed` for a specific evaluated case and `--panel confirmation` or `--panel sequences`
for another panel (`--case sequence` for the latter). Replays must reproduce recorded outcomes
and metrics. Output goes under the experiment's `replays/` directory; existing replays are
not overwritten.

### Fast command-walking experiments

Run the capability map, exploration calibration, braking screen, reward-component
analysis, and bounded PPO update audit:

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_fast_experiments \
  logs/mujoco_rl/command_walk_v4/baseline/policy.zip \
  logs/mujoco_rl/command_walk_fast_v1 --minutes 90
```

Use a new output directory. See `manifest.json` for the locked configuration,
`status.json` for progress, `result.json` for the completed PPO comparisons,
`reward_quality.json` for reward-versus-success summaries, and each `ppo_*` folder
for its `train.log`, `updates.jsonl`, evaluations, and policy/normalization pairs.
Correlated exploration is evaluated only; PPO uses its own independent Gaussian
sampling and matching likelihood. Nothing is automatically promoted.

### Preflight before a long walking run

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_preflight \
  logs/mujoco_rl/command_walk_v4/baseline/policy.zip \
  logs/mujoco_rl/command_walk_preflight_v1 --minutes 120
```

Use a fresh output directory. This runs controller sequence/envelope checks and
short PPO experiments with post-step full-rollout KL checking and exact optimizer
rollback. Read `manifest.json` for the preregistered gates, `status.json` for live
progress, `controller_gate.json` for controller results, and `result.json` for the
final recommendation. Each training arm has `train.log`, `guard.jsonl`,
`updates.jsonl`, and checkpoint/normalization pairs. No long training starts
implicitly, even if the gates pass.

### Final moderate-turn walking campaign (48-hour ceiling)

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_final_campaign \
  logs/mujoco_rl/command_walk_v4/baseline/policy.zip \
  logs/mujoco_rl/command_walk_final_v1
```

Resume the same output directory by adding `--resume`. Resume keeps the original
wall deadline and verifies the recorded Python source hashes. Simulator episodes
restart at a rollout boundary; saved weights, Adam state, normalization, and
curriculum are restored. A run lock rejects concurrent resumes.

Supported requests: forward speed zero or 0.1–0.3 m/s; yaw rate zero or
±0.15–0.5 rad/s. This is nominal simulation qualification, not hardware approval.
The campaign may finish early if baseline qualification passes, the learning
pilot has no replicated winner, or later training regresses/stalls.

Read `status.json` for live progress, `pilot_decision.json` for reward selection,
`qualification.json` for final evidence, and `result.json`/`SUMMARY.md` for the
outcome. Checkpoints remain in `jobs/`; the selected complete controller is in
`controller/`; labelled standalone demonstrations are in `videos/`.

```python
from mujoco_rl.command_walk_final_eval import Controller

controller = Controller("logs/mujoco_rl/command_walk_final_v1/controller")
controller.reset()  # episode start
# One call per 20 ms tick, using 87 raw observation values before correction:
action = controller.act(raw_observation, requested_v=0.2, requested_w=0.3)
controller.close()
```

The controller applies correction internally. Do not pre-correct these inputs.
The bundle's `qualified` field must be checked before presenting it as a release.
Unsupported requested commands raise `ValueError`. Model, normalization, and
correction settings must travel together. The MuJoCo runtime and pinned robot
assets remain dependencies of this repository's controller loader.
