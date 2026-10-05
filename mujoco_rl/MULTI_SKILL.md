# Flat-ground multi-skill training

## Bounded campaign with independent skill budgets

`skill_campaign.py` runs the audited campaign: physics and baseline checks,
navigation/squat qualification, recovery, running, jumping, then continuous
command chains. A blocked recovery stage does not block running or jumping.
The ceiling is 72 elapsed machine-hours: 6 audit, 20 recovery, 4 navigation/squat,
12 running, 12 jumping, 10 integration, and 8 reserved for a diagnosed repair.
Unused allocations do not launch speculative experiments. The runner currently
executes one job at a time, with eight simulation workers for training/search.

Initialize a new campaign from matched checkpoint bundles, then launch it:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_campaign logs/mujoco_rl/CAMPAIGN --init \
  --nav-checkpoint logs/mujoco_rl/multi_skill_v5/nav/accepted.zip \
  --squat-checkpoint logs/mujoco_rl/multi_skill_v6/squat/accepted.zip \
  --recovery-checkpoint logs/mujoco_rl/recovery_bootstrap_v17/bootstrap.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mujoco/bin/python -u \
  -m mujoco_rl.skill_campaign logs/mujoco_rl/CAMPAIGN
.venv-mujoco/bin/python -m mujoco_rl.skill_campaign \
  logs/mujoco_rl/CAMPAIGN --watch --logs
```

The initializer copies baselines and hashes them. `campaign_config.json` and
`source_snapshot/` preserve the implementation; `seeds.json` separates training,
development, and confirmation cases. Failed confirmations become mandatory
development retry panels. `status.json` records active processes, log paths,
heartbeat, elapsed machine-hours, completed-child CPU-hours, and stage outcomes.
`RESULTS.md` records the final disposition. Existing campaigns are never silently
restarted. Use a new directory for an explicitly justified subsequent campaign.

The runner accepts only independently qualified policies under `accepted/`.
Recovery requires 100 development and 100 holdout cases for each of prepared
lying, settled floor, and physically disturbed falls, plus tilted-balance
retention. Other skills require 50 development and 50 holdout cases. Thresholds
remain 80% overall and 75% per recovery direction/jump variant. Running also
requires navigation retention. Evaluations report counts and Wilson intervals.
Videos replay the first success and first failure in a recorded panel.

`motion_teacher.py` provides version 2 teachers with variable-length whole-body
joint targets in radians, durations in control steps, feedback, and phase
conditions. Legacy three-phase teachers remain supported by `recovery_bootstrap`.
Search templates are experimental; only motions passing a separate 50-case
verification can supply demonstrations. Collection uses separate physics seeds
and command perturbations. `skill_refine.py` fits the actor, rehearses the saved
balance policy, and runs bounded PPO trials with critic warmup, retention checks,
independent PPO/imitation diagnostics, and backward trajectory resets.
Failure to imitate a verified teacher stops PPO for diagnosis.

Environment version 12 fixes the jump flight-duration gate: 0.16 seconds equals
32 physics substeps at 5 ms, not eight substeps. Skill transitions preserve
physical state and remap delayed targets; observation refreshes do not advance
sensor history. `run_to` uses walking to turn and approach the goal, with running
on aligned longer segments. Dynamic-fall evaluations apply an external force;
prepared-pose injections are reported separately. These changes require new
qualification and do not retroactively validate older results.

Run focused CPU checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mujoco/bin/python -m pytest -q \
  tests/test_skill_campaign.py tests/test_skill_recovery.py tests/test_recovery_*.py
```

## Earlier single-stage pipeline

`skill_env.py` adds five 23-joint MuJoCo tasks to the published full-body
walking baseline: arbitrary-direction point-to-point travel, squat, recovery
from front/back/side falls, airborne running, and up/forward jumps. All runs
use the Loudbox CPU and the pinned `third_party/asimov-1` model. The original
walking checkpoint remains usable through `full_body_eval.py`.

From the repository root, after installing `mujoco_rl/requirements.txt` and
initializing the robot submodule, train one gated milestone at a time:

```bash
.venv-mujoco/bin/python -u -m mujoco_rl.skill_train \
    --stage pipeline --workers 24 --hours-per-stage 10 \
    --run-dir logs/mujoco_rl/multi_skill_v1
```

The pipeline begins with the published walking policy. It trains navigation,
squat, recovery, running, and jumping in that order. Each stage evaluates 50
fixed development seeds every 100,000 steps. A development pass triggers a
50-seed holdout evaluation; later attempts use fresh holdout seeds. The next
stage starts only after both sets reach 80% success. Recovery must also pass
at least 75% for each lying orientation, and jumping at least 75% for each
variant. Running must retain 80% navigation success. If a stage exhausts its
time budget, the pipeline stops and saves its best and final checkpoints.

For a single-stage refinement, pass `--stage nav|squat|recover|run|jump`,
`--warm-start PATH` or `--resume PATH`, and a new `--run-dir`. Every checkpoint
requires the matching `*_vecnormalize.pkl` file. Training runs store source
snapshots, settings, checkpoints, and per-seed evaluation records under their
stage directory. Inspect a checkpoint with:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_eval \
    logs/mujoco_rl/multi_skill_v1/nav/accepted.zip --skill nav --seeds 50
```

For recovery diagnostics, add `--recovery-level 6` to evaluate that starter
tilt range, and `--video PATH --video-seed SEED` to replay the same range.
Omitting `--recovery-level` always evaluates full lying recovery.

To resume navigation and continue the gated pipeline, pass `--stage pipeline`
and `--resume PATH` with a new run directory. Navigation resumes at the curriculum
level implied by the saved step count. Use `tmux` for long training sessions.
To resume a later stage, also pass `--resume-stage squat` (or the relevant
skill) and `--prior-run-dir PATH` pointing to accepted earlier stages.
`--resume` is optional for a later stage: omit it to restart that stage from
its accepted prerequisite. Recovery then trains from sixteen overlapping,
steeper floor-contact tilt ranges; full-angle development and holdout tests
remain unchanged. Use `--recovery-lr 1e-5` when resuming a recovery checkpoint
after excessive PPO KL updates.
Recovery starts with the accepted squat network features but a neutral action
head, because squat motion destabilizes even a nearly upright recovery reset.
Small recovery actions request at most 0.25 radians per unit action near zero;
larger actions smoothly reach the joint limits. Exploration starts at a
standard deviation of 0.05. The critic starts fresh for the recovery rewards,
and observation statistics adapt to the new states. Thirty percent of training
episodes rehearse earlier tilt levels; 60% of that rehearsal targets the
immediately preceding level. Advancement requires two passing
48-seed development and current-level validation probes, retention of the preceding level, and a fresh 50-seed
confirmation. These probes run every
25,000 steps. Current-level validation uses seeds 30000–30047, matching the
retention panel used after advancement, so each level must pass that panel first.
Pose sampling targets the weaker directional rates across development and
validation. Confirmation seeds use a distinct, recorded range for each run.
Use `--recovery-stress-seed-start N` to promote a discovered failing panel into
an additional 50-episode development check. It blocks advancement, guides pose
sampling, and is retained after its level first passes. Resume inherits this
panel; changing its seed clears its recorded baseline. Previously checked seeds
are development data; fresh confirmations remain separate.
When a fresh confirmation fails, its panel becomes a development retry: its
weak directions guide sampling and it must pass before another fresh attempt.
The pending panel is stored with the model. This prevents advancement from
discarding a known failure by trying different confirmation seeds.
After advancement, the last repaired panel becomes a recorded preceding-level
retention baseline and is checked alongside the other retention panels.
All results are recorded in `curriculum.jsonl`; full-angle evaluations
still run every 100,000 steps. Three consecutive preceding-level retention
failures, a repeated 20-point success drop,
or one million steps without advancement stops the run and saves checkpoints.
Starter reset poses lean the torso over compensating hip angles instead of
tipping the whole rigid standing posture onto the toe or heel. This assistance
fades between 15 and 45 degrees; full lying resets receive none.
Recovery training normalizes reward targets so large critic gradients do not
dominate the shared gradient clipping limit. Version 4 resumes migrate by
resetting the critic output and its optimizer state while retaining the actor,
observation statistics, and curriculum sidecar. Versions before 4 must start a
new run because action semantics changed. Recovery resets initialize motor
targets and delayed commands to the actual starting joint pose, so the first
policy action does not inherit a stale standing command.
Starter levels up to 25 degrees terminate and penalize a collapse below 0.30 m
or an upright cosine below 0.5. This keeps balance training focused on preventing
the fall instead of spending the remainder of each episode lying still.
Later recovery levels and full lying evaluations retain the full episode.
The recovery completion bonus is 500, above the maximum discounted standing
reward of 300 at gamma 0.995, so delaying completion cannot earn a better return
from standing rewards alone.
After each balanced probe, training orientations are sampled in proportion to
`0.10 + failure_rate`, preserving practice in every direction while spending
more trials on failures. Seeded evaluations remain balanced. The optional
`--recovery-leg-std 0.10` increases initial leg exploration when resuming a
stalled balance stage; arm exploration and the actor mean remain unchanged.
For a direction below 75% success, 80% of its current-level resets sample a
1.5-degree window between the largest successful tilt and the smallest failed
tilt. The remaining resets cover the full range; earlier-level rehearsal and
balanced evaluations remain unchanged. Focus windows appear in `curriculum.jsonl`.
When failures span multiple tilt ranges, the window covers those ranges instead
of assuming one monotonic success boundary. This keeps successful middle tilts
from displacing practice at both failing tails.

When exploration cannot discover coordinated starter corrections, search a
short simulated motion and fit the actor to successful randomized rollouts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mujoco/bin/python -m mujoco_rl.recovery_search \
    --output logs/recovery_controls.npy --tilt 19.5 --tilt-width 1.5 \
    --randomized-seeds 12 --rollout-seed-start 70000 --arms --feedback --workers 12
.venv-mujoco/bin/python -m mujoco_rl.recovery_bootstrap \
    logs/RUN/recover/checkpoint_STEPS.zip logs/recovery_controls.npy \
    --output logs/recovery_bootstrap/bootstrap.zip --episodes 1024 \
    --updates 8000 --action-noise 0.03
```

The search tests randomized physics when `--randomized-seeds` is positive;
otherwise it tests nominal physics. Search seeds are separate from evaluation
seeds. `--arms` includes shoulders and elbows, and `--feedback` adds corrections
from the actor's noisy, delayed IMU readings. Keep the search `.json` metadata
beside its `.npy` array to preserve the pose and feedback gains.
For several motions, use a JSON list containing `pose`, `min_tilt_deg`,
`max_tilt_deg`, `controls` (three phases of 23 actions), and optional `feedback`.
Four gains control hip tilt, hip gyro, ankle tilt, and ankle gyro. Six gains add
hip and ankle body-velocity corrections; search them with `--feedback --velocity-feedback`.
Optional `phase_ends: [5, 15, 40]` specifies three cumulative
control durations in 20 ms steps; changing timing requires fresh physics checks.
Balance-motion searches support angles through 30 degrees. A search score does
not establish coverage: verify the saved motion across the complete advertised
angle range and fresh physics seeds before collecting training demonstrations.
Searches above 25 degrees use the larger-angle stage's termination rules, so
temporary crouching receives the same treatment as during training.
The bootstrap keeps successful randomized trajectories and rehearses
earlier levels. `--action-noise` perturbs executed commands while retaining the
teacher's clean labels, exposing the actor to deviations from the teacher's path.
Actor fitting oversamples command transitions and masks previous-action inputs
to prevent learning to copy the preceding command. For a repair of an existing
PPO actor, `--preserve-motor-history` retains its learned command-history features;
validate closed-loop rollouts because this can introduce teacher-forcing shortcuts.
It reports development, current-level validation, 50 fresh cases, retention,
configured stress tests, and previously validated retry retention panels.
Inspect every report and its directional
rates before resuming PPO; low imitation loss alone does not establish recovery.
If the learner drifts from otherwise successful expert trajectories, use
`--learner-fraction 0.8 --keep-failed-demonstrations` to query expert labels on
states visited by a mixture of learner and expert commands. Retain the earlier
data with `--replay-demonstrations PATH`. These labels can cover failed learner
attempts; inspect the success count and retained episode count separately.
Use `--collection-poses left` to target one direction. When replacing an expert,
exclude its old labels from replay; retain the other directions with successful
reference-policy examples. Recorded initial noisy gravity identifies the pose.
Continue PPO with `--resume` pointing to `bootstrap.zip`. Bootstrapped policies
freeze observation and reward statistics during PPO to preserve action timing.
Use `--recovery-critic-warmup-steps 50000` after resetting the critic: the actor
stays frozen while the critic learns, then resumes updates at `--recovery-lr`.
During evaluation and subsequent PPO, the neural network supplies all actions;
the searched motion is only a source of training examples. These demonstrations
cover starter balance in selected directions; they do not establish full lying
recovery. Full recovery gates still apply.

If PPO forgets demonstrated skills, pass
`--recovery-demonstrations logs/recovery_bootstrap/bootstrap_demonstrations.npz`.
`RecoveryPPO` performs the standard SB3 PPO update followed by up to sixteen
actor-only imitation updates. The additional loss is `100 * mean_squared_error`
against demonstrated actions. Each update must lower that loss and respect
the rollout KL budget; rejected updates restore both weights and optimizer
state. The critic and exploration parameters are unchanged by imitation.
`train/approx_kl` retains SB3's PPO measurement. `train/ppo_kl` measures the
complete rollout after PPO; `train/post_imitation_kl` measures it after imitation.
Normalization is frozen,
the demonstration path is recorded and reused on resume, and saved actors
remain loadable with ordinary `PPO.load`. Inspect `train/imitation_loss`,
accepted/rejected imitation updates, skill success, and preceding-level retention.
To retain a validated policy's actual behavior, collect its successful rollouts:

```bash
.venv-mujoco/bin/python -m mujoco_rl.recovery_retention \
    logs/recovery_bootstrap/bootstrap.zip --workers 8 \
    --output logs/recovery_bootstrap/policy_retention.npz
```

Reference rollouts store both clipped simulator commands and unclipped policy
means. PPO imitates the original means, preserving saturated outputs without
pulling them toward the action limit. Controller demonstrations retain their
physical action targets. The selected target kind is recorded in `config.json`.

Pass this file to `--recovery-demonstrations`. Collection covers the validated
level, its successor, and earlier levels, using separate training seeds.
Every difficulty slot includes all four fall directions. Reports include the
initial tilt, and transition sampling respects episode boundaries.

For controlled recovery experiments, use `--recovery-imitation-weight` and
`--recovery-imitation-updates`; zero updates disables imitation. Keep
`workers * --rollout-steps` fixed when comparing runs (8 workers and 768 steps
match the usual 24 workers and 256 steps). `--seed` also resets the random
stream when resuming. `updates.jsonl` records PPO and imitation action changes;
negative `imitation_alignment` means their action changes oppose each other.
`training_episodes.jsonl` records actual difficulty, fall direction, completion,
termination reason, and episode reward components, including rehearsal episodes.
Use `recovery_diagnostics CHECKPOINT --level 9 --seeds 20000 20003 --output DIR`
to export complete sensor, actuator, and contact trajectories as NPZ files.

`recovery_compare CHECKPOINT --demonstrations REFERENCE.npz --settings SETTINGS.json
--run-dir DIR --steps 196608` launches bounded comparisons and records their commands,
configuration, live status, and results. The budget counts additional steps,
including when the checkpoint already has millions of steps. A settings file is a
JSON list such as `[{"name": "reduced", "learning_rate": 1e-6,
"imitation_weight": 10, "imitation_updates": 2, "leg_std": 0.05, "seed": 44}]`.
Keep the initial checkpoint, demonstrations, training seed, and rollout batch
size identical to isolate one setting. Select candidates using recovery and
retention evaluations; use fresh holdout seeds after selection.

Use `--recovery-stall-steps 250000` for an early stop on a resumed curriculum
that fails to advance. The default remains one million steps. Controller searches
can use `--all-joints` to explore all 23 joints and `--save-best` to preserve a
partial candidate for refinement. A saved partial candidate with
`search_criterion_met: false` requires evaluation before use as a demonstration source.

`recovery_bootstrap --sensor-memory` enables observation version 10. It stores
the initial noisy IMU gravity components in channels 81–82, unused by recovery,
so the actor can remember its starting direction after rotating upright.
This uses measured sensor history; actor inputs remain 83 channels. Collect
new demonstrations when enabling it. Checkpoints record the setting; training,
evaluation, videos, and the supervisor select the matching observation behavior.
Earlier checkpoints retain their original observations. Navigation continues
to use those channels for its goal. Timed skills restart their phase on activation.
`--phase-features` enables version 11, adding two bits of the internal motion
clock in recovery's unused command channels 6 and 78. These distinguish the
first 100 ms, 100–300 ms, 300–800 ms, and later motion; all 23 actions still
come from the neural network. Fitting and online imitation sample frames on
both sides of command transitions to avoid learning to switch prematurely.
Saved episode offsets allow phase features to be added to existing sensor-memory
demonstrations without repeating simulation.

After all five skills pass, `skill_supervisor.py` loads the accepted policies.
It supports `go_to`, `run_to`, `squat`, `jump_up`, and `jump_forward`; a fall
switches to recovery and resumes a pending travel target. Evaluate chained
commands and an injected fall with:

```bash
.venv-mujoco/bin/python -m mujoco_rl.skill_supervisor \
    logs/mujoco_rl/multi_skill_v1 --seeds 50
```

The waypoint controller uses exact MuJoCo position and orientation. Arm
support contacts are local collision overlays for recovery. These are
simulation tools, not calibrated hardware behavior. A physical robot would
need localization, actuator validation, and staged safety trials before using
any of these policies.
