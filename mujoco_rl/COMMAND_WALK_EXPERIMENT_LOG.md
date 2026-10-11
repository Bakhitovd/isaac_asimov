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

## 2026-10-11 20:07 UTC — All pilots finished; extension gate not met

The campaign stopped after 2.35 hours with `review_required`. All four pilots completed
497,664 additional steps each (1,990,656 total). No trainer or coordinator process remains
active. No extension, holdout qualification, or policy release was triggered.

Final checkpoints, 50 nominal development cases per family:

| Motion | Control 44 | Protected 44 | Control 45 | Protected 45 |
|---|---:|---:|---:|---:|
| Forward | 100% | 98% | 100% | 100% |
| Walking left turn | 88% | 98% | 82% | 82% |
| Walking right turn | 92% | 100% | 88% | 96% |
| Left pivot | 76% | 94% | 44% | 78% |
| Right pivot | 98% | 82% | 46% | 94% |
| Stop | 100% | 100% | 100% | 100% |
| Mixed sequence | 86% | 68% | 66% | 90% |

All final checkpoints failed nominal retention. Control 44, control 45, and protected 45 each
rolled back once; protected 44 did not roll back during its budget. The best retained protected
44 checkpoint remains step 6,340,608, with the results recorded above. For the other three
trials, selection retained the initial baseline. Thus no approach improved with retention
in both seeds, and the automatic extension gate correctly withheld longer training.

Interpretation: selective protection produced one promising intermediate checkpoint but did
not demonstrate repeatable improvement. Nominal failures persist without artificial delays,
so delay robustness cannot be the sole explanation. The experiment does not isolate whether
critic adaptation, update size, reward conflicts, or insufficient reference coverage causes
these regressions. Next review should compare those mechanisms and per-family failure traces
before choosing a further training change. Do not restart the same campaign or describe the
saved intermediate policy as holdout-qualified.

## 2026-10-11 20:53 UTC — V4 implementation and launch preparation

Implemented the approved 24-hour staged campaign; see [V4 configuration](COMMAND_WALK_V4.md).
Fresh transfer and resume are explicit. The first experiment compares current two-rollout
initialization, preserved critic/return statistics, and twenty-rollout critic calibration,
using seeds 44/45 and equal actor-enabled budgets. Calibration steps are reported separately.
The comparison tests initialization packages, not independent causal effects of critic and
normalization changes.

Later stages are implemented behind behavioral gates: balanced turning/pivot/braking
specialists, a bounded heading-reward comparison only for confirmed heading-dominated plateaus,
teacher distillation with student-state aggregation, transition refinement, and independent
nominal/60-second-sequence qualification. If all initialization packages fail retention, the
campaign stops before training specialists. No delays or physics randomization are added.

Validation: 202 tests passed. PPO smoke training recorded 256 calibration steps and 512 actor
steps; resume reached 640 actor steps without repeating calibration. Distillation completed
four small-data rounds and an optimizer/data resume check, and correctly rejected qualification
with one evaluation case per motion. An expired-job test saved a resume bundle without actor
updates. Nominal replay rendered successfully; the old baseline failed the single 60-second
sequence smoke case. These checks validate implementation paths, not a newly qualified policy.

Runtime output is `logs/mujoco_rl/command_walk_v4/`, with an automatic `campaign_journal.md`
for stage transitions and stop decisions, persisted deadlines in `campaign.json`, and per-job
attempt logs, evaluations, and immutable policy/statistics pairs. The live tmux session is
`asimov_command_walk_v4`. The implementation revision is recorded at campaign launch.
No new motion-learning result is claimed at this preparation milestone.
