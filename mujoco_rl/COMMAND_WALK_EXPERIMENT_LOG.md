# Command walking experiment log

Record dated evidence and decisions here when reviewing a run or changing the training system.
For each experiment, record its source checkpoint and normalization, code revision, changed
variables, seeds/budget, evaluation scope, per-motion results, and the next decision. Distinguish
development probes from holdout qualification and observations from explanations being tested.
Update the versioned guide when behavior or configuration changes. Runtime status and detailed
logs remain in the run directory; snapshots below are not live status.

## 2026-10-11 — V3 nominal retention experiment

### Motivation

V2 completed four approximately 500k-step pilots and stopped for review. Neither approach
improved with nominal retention in both seeds. One aligned-reward trial improved left pivots
to 100% but reduced right pivots to 76%; the other reduced nominal right pivots to 0%.
These results demonstrate regression, but do not isolate its cause. V2 also required pilots
trained with rare action delays to improve on a much harder combined-delay challenge.

The new comparison isolates a successful-reference penalty while holding nominal physics,
rewards, observations, and command sampling constant between control and protected PPO.
See [V3 configuration and gates](COMMAND_WALK_V3.md).

### Reproduced baseline

Source: `command_walk_v1/seed45/checkpoints/005953536_probe/policy.zip` and matching
`policy_vecnormalize.pkl`, under `logs/mujoco_rl/`. Copies and SHA-256 hashes are stored in
`command_walk_v3/baseline/`. The 350-case rerun reproduced all previous family success counts.

| Motion | Nominal development success (50 cases each) |
|---|---:|
| Forward | 98% |
| Walking left turn | 92% |
| Walking right turn | 98% |
| Left pivot | 84% |
| Right pivot | 98% |
| Stop | 100% |
| Mixed sequence | 92% |

Replay seed 301001 failed post-turn heading RMS: 11.92 degrees against a 10-degree limit.
Pivot drift (0.071 m), stopping speed (0.0125 m/s), and stopping yaw rate (0.0169 rad/s)
passed their limits. This case needs heading correction; it is not evidence that all pivot
failures require stronger drift penalties. Four annotated successful/failed pivot and mixed
replays and their traces are saved in `command_walk_v3/audit/`.

### Implementation and validation

- Code revision: `0f45c43`, on `main`.
- Reference data: 56 successful episodes, 7,168 frames, distinct from development/holdout seeds.
- Comparison: control weight 0 versus protected weight 0.1, training seeds 44 and 45.
- Budget: 497,664 additional steps per pilot; two trials run concurrently.
- Validation: 189 tests passed; short training/resume smoke runs completed. Smoke evaluations
  used one case per family and are not performance or qualification evidence.
- Evidence: [preparation report](reports/command_walk_v3_preparation.json).

### Status snapshot — 2026-10-11 17:35 UTC

Both seed-44 trials reached 165,888 additional steps and were evaluating. Seed-45 trials had
not started. The latest completed quick probes had only 10 cases per family:

| Motion | Control | Protected |
|---|---:|---:|
| Forward | 80% | 100% |
| Walking left turn | 70% | 70% |
| Walking right turn | 80% | 100% |
| Left pivot | 80% | 80% |
| Right pivot | 20% | 30% |
| Stop | 100% | 100% |
| Mixed sequence | 10% | 40% |

These small probes indicate possible regression, particularly right pivots and mixed sequences.
They do not establish that protection works. Full panels are needed to confirm the change;
the configured retention/rollback gates remain active. No new policy is qualified.

### Outstanding work

1. Review full panels and any rollback records; complete the paired-seed comparison.
2. Select a retained improvement or stop for diagnosis according to the explicit gates.
3. Make targeted reward/transition changes only after identifying remaining failure causes.
4. Qualify nominal behavior on separate holdouts and longer command sequences.
5. Test observation/action history and gradual delays after nominal qualification.

Live view: `tmux attach -t asimov_command_walk_v3`. Runtime evidence is in
`logs/mujoco_rl/command_walk_v3/`: `status.json`, per-trial `status.json`, `evaluations/`,
`updates.jsonl`, and, when available, `comparison.json` and holdout reports.

## 2026-10-11 19:07 UTC — Seed 44 complete; seed 45 still running

Campaign elapsed time: 1.83 hours. Both seed-44 pilots completed 497,664 additional steps.
Both seed-45 pilots reached 276,480 additional steps and were evaluating; their latest completed
full panels were at 165,888 additional steps. All results below use 50 development cases per
family, with nominal physics. No holdout qualification has occurred.

| Motion | Baseline | Best retained protected 44 | Final control 44 | Final protected 44 | Latest full control 45 | Latest full protected 45 |
|---|---:|---:|---:|---:|---:|---:|
| Forward | 98% | 100% | 100% | 98% | 100% | 90% |
| Walking left turn | 92% | 90% | 88% | 98% | 80% | 82% |
| Walking right turn | 98% | 98% | 92% | 100% | 98% | 98% |
| Left pivot | 84% | 98% | 76% | 94% | 78% | 78% |
| Right pivot | 98% | 94% | 98% | 82% | 94% | 98% |
| Stop | 100% | 100% | 100% | 100% | 100% | 100% |
| Mixed sequence | 92% | 94% | 86% | 68% | 92% | 88% |

The protected seed-44 checkpoint at lifetime step 6,340,608 is eligible for extension on that
seed: its weakest family improved from 84% to 90%, and no family declined beyond the configured
six-percentage-point tolerance. It is saved at
`logs/mujoco_rl/command_walk_v3/pilot_protected_44/checkpoints/006340608_full/policy.zip`,
with `policy_vecnormalize.pkl` alongside it. It does not meet nominal qualification thresholds.

Later protected seed-44 updates regressed, especially right pivots and mixed commands. The final
checkpoint is therefore not the selected candidate. Control seed 44 rolled back once to its
starting checkpoint, reduced learning rate to 1.5e-5, and produced no retained improvement;
its selected candidate remains the baseline.

Decision: preserve the intermediate protected checkpoint and finish seed 45 before deciding on
extension. The improvement has not replicated yet. Seed-45 full panels currently show regression
in both approaches. Campaign evidence: `command_walk_v3/comparison.json` and per-trial evaluation
and rollback records. Protection is promising in one checkpoint, not established as a reliable fix.
