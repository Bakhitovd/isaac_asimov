# V4: initialization diagnosis and focused motion learning

V4 implements a gated, 24-hour Loudbox CPU campaign for one whole-body command policy.
All stages use nominal MuJoCo physics, 23 joint position targets, and the existing 87-channel
control observation layout. The two hold-offset channels remain zero. No delay robustness
or real-robot qualification is claimed. Specialists are teachers, not a deployment router.

## Preserved sources and initialization experiment

The original v1 seed-45 checkpoint at step 5,953,536 supplies walking/braking actors. The
retained V3 protected seed-44 checkpoint at step 6,340,608 supplies the pivot actor. Policy
and normalization files are copied, hashed, and preserved under the campaign directory.

Fresh transfer and resume are explicit. Transfer uses fresh optimizer state, preserves the
actor and observation normalization, and chooses one of these initialization packages:

| Package | Critic | Return statistics | Actor freeze |
|---|---|---|---|
| `current` | Transfer trunk, reset output | Fresh | 2 rollouts |
| `preserved` | Transfer complete critic | Copy source | 2 rollouts |
| `calibrated` | Transfer trunk, reset output | Fresh | 20 rollouts |

Copied return statistics do not include per-environment return accumulators; new episodes
start those at zero. Reward statistics continue adapting while observation statistics stay
frozen. Resume restores weights, optimizer, normalization, counters, and curriculum, and
rejects a different V4 specification. It does not promise bitwise replay of subprocess
physics states after interruption.

Each initialization receives training seeds 44/45 and **98,304 actor-enabled steps**
(100k rounded down to complete 6,144-step rollouts). Calibration adds 12,288 or 122,880
steps separately. Before/after panels use 50 cases per motion. Periodic checkpoints occur
approximately every 25k actor steps; quick probes occur approximately every 50k.
Diagnostics include value-target RMSE, return and advantage statistics, reward scale,
actor/critic gradients, KL, and action changes.

These are package comparisons, not independent causal tests of critic weights versus reward
normalization. Select only packages retaining every family within four percentage points
in both seeds. Rank by worst family, then mean success, preferring `preserved` on a tie.
If all fail, stop before specialist training.

## Stage budgets and gates

| Stage | Maximum time | Behavior and gate |
|---|---:|---|
| Initialization diagnosis | 4 h | Replicated nominal retention |
| Specialists | 8 h | Walking turns, pivots, braking; both seeds must qualify |
| Distillation | 4 h | One student reaches at least 90% in every family |
| Transitions | 6 h | Candidate reaches 48/50 per nominal family |
| Qualification | 2 h | Independent nominal and 60-second sequence holdouts, replays |

Stage deadlines are saved on first entry and bounded by the original 24-hour deadline.
Resuming does not reset either clock. At most two eight-environment PPO jobs run together.
An exclusive campaign lock prevents duplicate launches. Code and source hashes must match
on resume. Failed gates, expired budgets, numerical errors, and interruptions save evidence
and stop rather than starting another unspecified experiment.

### Specialists

Walking turns train left/right arcs; pivots train both directions. Braking episodes contain
movement, stop, forward restart, and stop, covering forward, both arcs, and both pivots.
Allocate 80% expected simulation time to the task and 20% to forward/stop rehearsal,
correcting episode probabilities for duration. Alternate target directions. Begin with
0.1–0.2 m/s and 0.15–0.3 rad/s, expanding to 0.3 m/s and 0.6 rad/s after a 90% gentle panel.

Each specialist/seed gets up to 497,664 actor steps. Qualify assigned motions at 48/50 and
forward/stop rehearsal at 46/50. Unrelated skills need not be preserved by a specialist.
Select retained intermediate checkpoints rather than the last weights.

Only a confirmed plateau over at least 100k actor steps with heading as the dominant failure
permits the predefined reward comparison: unchanged reward versus an added
`-0.5 * min((heading_error / radians(10))**2, 4)` during turns and their following stops.
Both variants use both seeds and up to 196,608 actor steps per trial, within the same stage
deadline. Other failures require review. Failed specialists never become teachers.

### Consolidation and transitions

Collect eight successful teacher episodes per family, 128 frames per episode, using seeds
starting at 4,000,001. Each teacher uses its own observation normalization; student targets
are unclipped means. Routing uses requested commands: forward → baseline, arc → walking-turn
teacher, pivot → pivot teacher, zero → braking teacher. Failed initial teacher episodes are
excluded; insufficient successful examples stop collection.

Train the baseline-initialized student's actor with balanced mean-squared-error batches,
Adam at 1e-4, up to 20 epochs per round, keeping its exploration scale fixed. Run at most
three additional student-state aggregation rounds. Label student states with teacher actions;
count interventions for excessive tilt, falls, and task failures. These labels are not claimed
to be successful demonstrations. Select by simulated behavior, not imitation loss. Distillation
checkpoints preserve data, optimizer, and RNG state for an interrupted-run resume; a partial
minibatch epoch may be repeated.

Transition training uses 3–8-second command segments including both turn directions, pivots,
stop/restart, and turn reversal, with 30% isolated-motion rehearsal. Each seed has a ceiling
of 995,328 actor steps within the six-hour stage. Confirm regressions on a second full panel,
rollback and halve learning rate, and stop after two rollbacks. Full panels run at least
approximately every 150k steps and on promising quick probes.

### Qualification and publication

Use 50 development and 50 holdout cases per family, requiring 48 successes in each. Also
require 19/20 successful 60-second sequences on separate development and holdout panels.
Existing physical limits apply to every scored segment. Standard development seeds start
at 1001, holdouts at 10051, and each case family has a separate 100,000-seed offset. A failed
holdout is not automatically retried. Reports include failures and Wilson intervals.

Qualification creates seven motion videos, a long sequence, available representative failure
replays, policy/statistics files, contract, and hash manifest. `--publish` commits/pushes that
bundle to `mujoco_rl/checkpoints/command_walk_v4/` only after all gates and replay checks pass,
from a clean `main`. An existing release is not overwritten. Training completion is not release.

## Launch and inspect

```bash
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/home/ailevate/.local/mujoco-osmesa/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mujoco/bin/python -u -m mujoco_rl.command_walk_v4_campaign logs/mujoco_rl/command_walk_v4 --publish
.venv-mujoco/bin/python -m mujoco_rl.command_walk_v4_campaign logs/mujoco_rl/command_walk_v4 --watch
```

For an interrupted campaign, use its original output root with `--resume`. PPO and distillation
resume into new attempt folders; immutable prior attempts remain available. Qualification
interruption requires review to avoid reusing a partially observed holdout silently.

`campaign.json` holds persisted budgets and job decisions. `status.json` and per-attempt status
distinguish calibration, training, evaluation, completion, and stop causes. The human-readable
`campaign_journal.md` records stage transitions and stop decisions automatically. Detailed
evidence lives under the run directory; dated reviewed conclusions belong in
[the experiment log](COMMAND_WALK_EXPERIMENT_LOG.md).

## Validation

```bash
MUJOCO_GL=disable .venv-mujoco/bin/python -m pytest -q tests/test_command_walk_v4.py tests/test_command_walk_robust.py tests/test_command_walk_reference.py
```

Tests cover nominal compatibility, transfer/normalization, frozen-actor calibration, resume,
balanced targets, braking restarts, long-sequence timing, teacher normalization/routing,
behavioral gates, deadline persistence, and duplicate-launch protection. Local checkpoint
integration tests skip when the saved source artifacts are unavailable.
