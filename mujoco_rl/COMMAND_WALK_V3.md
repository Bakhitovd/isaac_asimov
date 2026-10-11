# Nominal command walking experiments

V3 first targets reliable forward walking, walking turns, pivots, stopping, and transitions
under fixed MuJoCo physics. Delay robustness is a separate, later milestone. All results
are simulation-only. The original rewards, 23 joint targets, and control observations are
unchanged in the first comparison.

See [the experiment log](COMMAND_WALK_EXPERIMENT_LOG.md) for dated observations, results,
decisions, and outstanding work. The live run status is under `logs/mujoco_rl/command_walk_v3/`.

## Rationale and uncertainties

V2 showed nominal skill regression during updates and sensitivity to delays. Its aligned
variant changed both observations and rewards, so the trials did not isolate either change.
The pilots trained with occasional action delay but extension depended on a much harder
combined action/sensor-delay challenge. That selection mismatch could reject intermediate
progress; it does not explain away the measured nominal regressions.

V3 tests whether selectively preserving successful baseline actions reduces forgetting while
allowing weak skills to improve. Success is not assumed: reference coverage may be insufficient,
the penalty may prevent useful adaptation, or the unchanged reward may still favor poor behavior.
The control and two training seeds are necessary to distinguish these possibilities.

| Result | Next decision |
|---|---|
| Protected PPO improves with retention in both seeds | Extend within the stated budget |
| Both retain skills but plateau | Diagnose remaining failures and test one targeted reward change |
| Both regress | Inspect update size, critic adaptation, sampling, and reference coverage |
| Protection preserves behavior but blocks improvement | Revisit its strength or selected examples |

These interpretations guide review; only the explicit extension and qualification gates below
authorize automatic continuation. Healthy PPO statistics alone do not qualify a policy.

## Baseline and diagnosis

The source is v1 seed 45, step 5,953,536, with its matching VecNormalize statistics.
`command_walk_prepare` copies both files, records hashes and source provenance, and collects
eight successful episodes per command family using seeds starting at 2,000,001. Failed
episodes are recorded but excluded. Each accepted episode contributes 128 evenly spaced
frames. These cases are separate from development (1001) and holdout (10051) panels.

`command_walk_audit` produces annotated successful and failed left-pivot/mixed replays,
physical trajectories, joint positions/velocities/actions, and failure summaries. The full
350-case baseline panel must reproduce the previously recorded family success counts
before the campaign starts training.

## Matched experiments

`command_walk_nominal_campaign` runs control and protected PPO for seeds 44 and 45,
497,664 additional steps each. Two trials run concurrently, eight CPU environments each.
Both use fixed nominal physics, original rewards, identical sampling, frozen observation
normalization, and the same transferred actor. A fresh critic receives two warmup rollouts.
Existing mixed-command training remains unchanged to isolate the protection experiment.

Protected PPO adds `0.1 * mean_joint_KL(reference || student)` on independently sampled
successful reference frames to each PPO optimizer update. The reference Gaussian retains
unclipped means and its original standard deviation. The combined gradient is clipped to
the existing maximum norm. This is a soft regularizer, not a guarantee against forgetting.
Reference data never enters PPO's on-policy rollout buffer. Control uses weight zero.
Resume requires the same dataset hash and weight. The diagnostic `reference_kl` measures
departure from successful reference behavior; increasing it can indicate inadequate protection.

Quick development panels run approximately every 50k steps; full 50-case panels run at
least every 150k steps, rounded to probe boundaries. Two full regressions exceeding six
percentage points in any family trigger rollback and a lower learning rate. Three rollbacks
stop the trial. Selection considers all retained full-panel checkpoints, including the baseline.

Extension requires at least four percentage points of improvement in the baseline's weakest
family, no decline in the minimum family success rate, and per-family retention in both seeds.
One selected approach can receive another one million steps per seed. Otherwise the campaign
stops for review, preserving failure evidence. It never automatically adds harsher physics.

Nominal qualification requires at least 95% observed success in every family on development
and a separate 50-case holdout panel. At 50 cases this requires 48 successes. A qualifying
result still needs longer varied-command demonstrations before release. Later reward changes,
transition-distribution changes, and sensor-history/delay experiments remain separate gated
experiments; this coordinator does not invent or launch those changes automatically.

## Commands

```bash
export MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-mujoco/bin/python -m mujoco_rl.command_walk_nominal_campaign logs/mujoco_rl/command_walk_v3
.venv-mujoco/bin/python -m mujoco_rl.command_walk_robust_status logs/mujoco_rl/command_walk_v3 --watch
.venv-mujoco/bin/python -m pytest -q tests/test_command_walk_reference.py tests/test_command_walk_robust.py tests/test_command_walk.py
```

Prepare `baseline/`, `baseline.json`, and the saved comparison panel `baseline_expected.json`
first; see the preparation/evaluation modules' CLI help.
Use a fresh output root for each campaign. Logs, immutable checkpoints, comparison reports,
and replays live under ignored `logs/`. No new policy is published merely because training ends.
