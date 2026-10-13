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

## 2026-10-12 — V4 results and retrospective

Revision `c83ef44` completed six initialization diagnostics and stopped at 2026-10-11
21:59:50 UTC after 1.09 hours. Each trial received 98,304 actor-enabled environment steps;
calibration added 12,288 steps for current/preserved or 122,880 for calibrated. No package
retained every family within four percentage points in both seeds. Specialists, distillation,
transition refinement, and qualification did not start. No new policy qualified.

Nominal development success, 50 cases per family; paired entries are training seeds 44/45:

| Motion | Baseline | Current | Preserved | Calibrated |
|---|---:|---:|---:|---:|
| Forward | 98% | 78/100% | 100/100% | 100/96% |
| Left arc | 92% | 82/68% | 80/82% | 94/92% |
| Right arc | 98% | 86/98% | 98/98% | 84/92% |
| Left pivot | 84% | 88/72% | 92/92% | 74/74% |
| Right pivot | 98% | 34/80% | 100/78% | 48/98% |
| Stop | 100% | 100/100% | 100/100% | 100/100% |
| Mixed | 92% | 28/76% | 92/88% | 64/96% |

Evidence: `logs/mujoco_rl/command_walk_v4/initialization_comparison.json`, per-attempt
`report.json`, `updates.jsonl`, and `episodes.jsonl`. This review reads existing evidence;
it does not add simulation evaluations or tests.

### Findings

- Baseline deterministic evaluation passed 331/350 cases without falls or self-collisions.
  Its left pivot failed 8/50 cases; seven had post-turn heading RMS violations and five
  had stop-speed/yaw violations (overlapping failures). Left-arc failures were predominantly
  heading errors. These are precision/transition failures in the evaluated nominal cases.
- Preserving critic/return statistics reduced first actor-update value-target RMSE to
  0.016/0.018, versus 0.256/0.263 for current initialization. Longer calibration achieved
  0.058/0.057. These values use each package's normalized reward scale. Preservation helped
  critic calibration but did not establish behavioral retention.
- Across all six trials, 0/1,296 logged stochastic training episodes passed the acceptance
  criteria. Stop-speed and stop-yaw violations occurred in 1,294 episodes each. Mean action
  standard deviation stayed near 0.132: about 0.033 radians of joint-target noise before
  clipping/actuator limits at the 0.25-radian action scale. Deterministic evaluation removes
  this action sampling. This mismatch warrants an exploration-scale experiment; it does
  not prove that noise caused regression or that PPO cannot learn from failed episodes.
- Preserved seed 44 retained the aggregate 331/350 successes while left arcs declined from
  92% to 80%. Aggregate performance can hide losses in an individual motion.
- V4's gate successfully prevented extending a regressing package, but also made specialist
  experiments conditional on broad PPO retaining all skills. The specialist hypothesis
  remains untested. Per-family regression is established; conflicting task gradients have
  not been measured directly.

### Recommended next experiment (not launched)

Preserve the baseline actor and its normalization. First compare unchanged control with
bounded heading/speed command corrections on separate tuning cases, then on development
cases. Keep the reference heading and all scoring tied to the original requested trajectory;
internal corrections must not redefine success. Include turns while walking, both pivots,
turn-to-stop, restart, and reversals. A frozen actor can still regress under changed inputs.

If command corrections demonstrate useful control authority, train only a small correction
module with task-relevant exploration, keeping the whole-body motor policy deterministic
and frozen. Measure the effect of exploration before choosing its scale. Train varied
command sequences and score heading, braking, drift, and existing motion retention. This
is a proposed constrained adaptation, not a proven solution. Escalate to focused motor-policy
training if command corrections cannot fix the observed failures. Reserve separate holdouts
and 60-second sequences for qualification; report naturalness separately from task success.

## 2026-10-12 17:59 UTC — Frozen-policy correction experiment launched

Implemented `command_walk_correction.py` and launched the requested diagnostic in tmux
`asimov_command_walk_correction`, with eight CPU evaluation workers and a 90-minute maximum.
The preserved V4 baseline policy/normalization are copied and hashed. No optimizer is used.
Every batch checks that model weights/statistics remain unchanged; every physics step checks
that reference-heading integration and the scoring command use the original trajectory.

The preregistered configurations are zero correction; heading gains 0.25, 0.5, and 1.0;
heading gain 0.5 with yaw-rate feedback gain 0.15; and heading gain 0.5 with speed feedback
gain 0.25. Corrections are limited to 0.15 rad/s and 0.04 m/s, with correction slew limits
0.6 rad/s² and 0.15 m/s². Velocity feedback uses a 0.1-second filter. Only copies of raw
policy command channels 6/78 change; reference heading, heading-error inputs, gait clock,
physics, sensor histories, and acceptance criteria remain owned by the original environment.
Effective command inputs retain the existing 0–0.3 m/s and ±0.6 rad/s bounds.

- Tune on 20 cases per family starting at seed 6,000,001, plus existing family offsets.
  Lock one nonzero setting before development: prefer retention within 10 percentage points,
  then worst-family success, mean success, and lower violation. Evaluate the selected setting
  even if tuning finds no improvement, to measure whether the finding repeats.
- Compare noise scales 0, 0.25, 0.5, and 1 times the source per-joint standard deviations
  on 20 paired cases per family starting at 7,000,001. Use identical underlying Gaussian
  draws per case across scales; the actor mean is evaluated on each resulting trajectory.
- Compare baseline/selected correction on 50 cases per family starting at 1001 and the
  separate standard confirmation range 10051. The latter is not asserted to be globally
  unseen across the repository's historical experiments. Reproduce archived development
  baseline success counts before accepting the comparison.
- Compare 20 paired 60-second sequences using the existing sequence generator and limits.

Development benefit requires no family loss beyond four percentage points, at least four
percentage points of improvement in worst-family success, and no mean-success decrease.
Report paired improvements/regressions and continuous errors, including negative results.
This experiment does not automatically train a correction network or qualify a release.
Evidence/configuration/snapshots: `logs/mujoco_rl/command_walk_correction_v1/`.

## 2026-10-12 — Correction experiment completed: useful improvement, remaining turn failures

The preregistered experiment completed in 683.8 seconds (11.4 minutes), evaluating 2,840
episodes. The selected heading gain was 0.5, with zero yaw-rate/speed feedback gains.
The policy receives the original yaw command plus bounded, rate-limited heading correction;
all evaluation targets remain original. Selection used only tuning cases (138/140 versus
133/140 baseline successes). The actor and normalization remained unchanged.

Paired nominal results; each entry is baseline → corrected, 50 cases per family:

| Motion | Development (1001) | Confirmation (10051) |
|---|---:|---:|
| Forward | 98% → 100% | 90% → 100% |
| Left walking turn | 92% → 90% | 92% → 92% |
| Right walking turn | 98% → 100% | 96% → 100% |
| Left pivot | 84% → 98% | 94% → 98% |
| Right pivot | 98% → 100% | 98% → 100% |
| Stop after forward motion | 100% → 100% | 100% → 100% |
| Mixed commands | 92% → 92% | 90% → 88% |

Development total improved from 331/350 to 340/350 (12 cases improved, three regressed).
Confirmation improved from 330/350 to 339/350 (12 improved, three regressed). Twenty paired
60-second development sequences improved from 13/20 to 19/20, fixing six failures without
introducing a new one. These are observed counts, not a guarantee of population reliability.
The correction passed the predefined development benefit gate but did not meet full per-motion
qualification. A fresh long-sequence holdout has not been evaluated.

### Exploration measurement

On the same 140 cases at each scale (20 per family):

| Fraction of original action standard deviation | Successful episodes |
|---|---:|
| 0 (deterministic) | 133/140 |
| 0.25 | 111/140 |
| 0.5 | 0/140 |
| 1.0 | 0/140 |

Half-scale noise caused stop-yaw violations in every case. Full-scale noise caused yaw-tracking,
stop-speed, and stop-yaw violations in every case. Quarter-scale noise caused stop-yaw
violations in 25/140 cases. This controlled comparison establishes that sampled joint noise
disrupts precision in the frozen policy. It does not establish that noise caused PPO's weight
regressions or identify the best exploration level for subsequent learning updates.

### Remaining work and decision

The correction removed all development left-pivot heading/stopping failures. The remaining
development pivot failure was 0.2544 m drift against a 0.25 m limit. The confirmation pivot
failure involved stopping. Left walking turns remain the principal weak family: five
development failures have yaw-rate error above 0.10 rad/s, with requested forward speeds
0.240–0.296 m/s and turn rates 0.513–0.586 rad/s. Mixed-command failures include fast left
turns and, in confirmation, two stop-yaw violations. Do not infer that stopping after every
possible motion is solved from the standalone stop-family score.

Preserve this frozen-policy controller as a prototype. Focus the next intervention on these
left-turn and transition states; measure whether bounded command corrections have enough
authority before choosing a learned correction or a motor-policy update. The ZIP alone is
still the baseline: improved results require the correction wrapper. No PPO run or release
was launched after this experiment.

Validation: archived development baseline counts reproduced in every family; runtime goal,
policy/statistics, and paired-case invariants passed. Replay tooling preserves the original
inference batch: scalar inference produced small contact-trajectory differences despite
matching failure classifications. Initial mismatched replay artifacts are retained under
`replays_incomplete/`; reviewed replays are under `replays/` and require matching outcomes
and physical metrics. No unit-test suite was run for this experiment.

Both paired replays completed and reproduced their recorded physical metrics. Left-pivot
seed 301001 improved post-turn heading RMS from 11.919° to 5.427°, passing the 10° limit.
Hard left-arc seed 101010 still failed during the turn, although post-turn heading RMS
improved from 19.366° to 7.274°. In that arc, the corrected yaw input was at the 0.6 rad/s
ceiling throughout the measured turning segment: increasing heading gain alone has no
remaining command authority there. This is evidence for investigating the joint speed/turn
command response or targeted motor-policy adaptation, rather than increasing this gain again.
The replay folders are `replays/development_left_pivot_301001/` and
`replays/development_left_arc_101010/`, each containing baseline/corrected videos, traces,
and a comparison plot.

Reviewed evidence: [compact report](reports/command_walk_correction_v1.json),
[results chart](reports/command_walk_correction_v1.png), and the full run directory above.

## 2026-10-12 — Fast follow-up experiments (preregistered)

Run: `logs/mujoco_rl/command_walk_fast_v1/`; tmux: `asimov_fast_experiments`.
Source: frozen `command_walk_v4/baseline/policy.zip` and its normalization.

1. **Capability map:** requested speeds 0.1/0.2/0.3 m/s, yaw magnitudes
   0.2/0.4/0.6 rad/s, both arc directions, ten paired initial conditions per
   cell. Compare heading gain 0.5 with internal speed reductions 0/.02/.04 m/s.
   Requested trajectories, physics, reward, and success thresholds stay fixed.
   Select on the map, then evaluate 50 development and 50 confirmation seeds
   per family. These standard panels have historical exposure; they are separate
   from this run's tuning, not globally unseen data.
2. **Exploration:** 5/10/25% of source Gaussian action standard deviation;
   independent noise versus stationary AR(1) noise with 0.1 s correlation time.
   Twenty paired seeds per family plus deterministic reference. Correlated
   noise is an evaluation diagnostic, not silently inserted into PPO likelihoods.
3. **Braking:** heading correction with 0/2/5 degree soft deadbands while stopped;
   twenty seeds each for five move-stop-restart cases, mixed commands, and 60 s
   sequences. This screens candidates; it does not qualify a new controller.
4. **Reward quality:** save reward components grouped by physical success/failure,
   alongside existing heading, speed, yaw, slip, and effort/smoothness evidence.
   No reward redesign in this experiment. Foot clearance and step regularity
   need additional instrumentation and are not measured by this report.
5. **PPO audit:** preserve critic/return stats, freeze observation statistics,
   use two critic-only warmup rollouts at a common learning rate, then compare
   actor-phase learning rates 3e-5 and 3e-6 with training seeds 44 and 45.
   Use the largest independent-noise scale retaining every family within 5pp
   on the noise screen; if none qualifies, use .05 and label it exploratory.
   No command correction during this audit. Save after 0/1/4/16 actor rollouts;
   evaluate ten paired quick seeds at each checkpoint, then 50 development and
   50 confirmation seeds after 16. One rollout is 6,144 simulator steps with up
   to four PPO epochs; it is not a single optimizer minibatch.

The runner has a 90-minute total budget, immutable baseline copies, source hashes,
per-batch output, per-job training logs, status files, and paired final summaries.
No automatic long training, checkpoint promotion, or publication. This is a
bounded experiment; a successful rollout-noise screen does not prove learning
will improve. Completion and numerical results are pending at launch.

### Fast follow-up completion

All stages completed in 2,568.7 seconds (42.8 minutes). No policy was promoted;
training has stopped. Evidence: `logs/mujoco_rl/command_walk_fast_v1/result.json`
and the per-stage reports in that directory.

**Frozen controller improvement:** heading gain .5 plus a .02 m/s reduction of
only the policy's forward-command input while turning was selected. Original
requested motion and scoring were unchanged. Grid success increased 160/180 to
165/180. Against heading correction alone, development left arcs increased
45/50 to 47/50 and mixed commands 46/50 to 48/50; confirmation increased 46/50
to 48/50 and 44/50 to 46/50 respectively. Other family counts were unchanged.
Paired development cases included five improvements and one regression;
confirmation had four improvements and no regressions. This is a controller
wrapper improvement, not new network weights. The hardest grid cell, left arc
at .3 m/s and .6 rad/s, still failed all ten cases. Long sequences have not been
qualified with the speed adjustment.

**Exploration:** deterministic 136/140; independent noise at .05/.10/.25 of
original std passed 135/137/125 of 140; correlated noise passed 135/132/124.
None of the independent arms met every-family retention: forward fell from
20/20 to 18/20 even at .05 and .10. The declared .05 fallback was therefore
exploratory, not an approved noise setting. No clear correlated-noise benefit.

**Braking:** 0/2/5 degree deadbands all passed 132/140, including 17/20 long
sequences; no success-count improvement. Reward analysis also found failed
mixed episodes still averaged 6.671 reward per step versus 6.720 for successful
ones (two failures versus eighteen successes). This small sample suggests
aggregate reward weakly distinguishes some threshold failures; it does not
establish the optimal replacement reward.

**PPO:** 16 actor rollouts per arm (98,304 actor-training steps), two training
seeds, common preserved-critic initialization and .05 exploration scale.
At 3e-5, seed 44 confirmation right-pivot success fell 98% to 54%; seed 45
forward/mixed fell to 0%, stop to 26%. At 3e-6, seed 44 retained all families
within the allowed 4pp but had no aggregate improvement; seed 45 failed
retention. The first actor update's post-update rollout KL was 5.80/5.33 at
3e-5 and .0585/.0531 at 3e-6, against target .015. Initial action std was about
.00661. Thus reducing exploration alone did not produce safe learning; even
small absolute mean changes were large relative to this narrow distribution.
SB3's subsequent KL early-stop check does not undo an already-applied step.
This diagnosis concerns these low-noise audit runs, not proof of a single
cause for every historical regression.

Next proposed experiment: keep the frozen improved controller as the candidate;
validate its long transitions and hard-turn envelope. Before more PPO training,
audit a reversible actor step with post-update KL checking and optimizer-state
rollback/backtracking. Do not start another long run on the strength of aggregate
reward or the ten-seed quick panel.

## 2026-10-12 — Preflight experiments before long training (preregistered)

Run: `logs/mujoco_rl/command_walk_preflight_v1/`; tmux:
`asimov_preflight`. Source remains the frozen V4 baseline and matching statistics.
The experiment is bounded to 120 minutes and never starts long training itself.

- Compare heading correction against heading + .02 m/s internal turn-speed
  reduction on two new 50-episode, 60-second sequence panels, 50 new seeds per
  motion family, and 20 per braking/restart case. Sequence seed bases are
  12,000,001 and 14,000,001; family base 16,000,001 (case offsets still apply).
- Map both turn directions at speeds .20/.25/.30 m/s and yaw magnitudes
  .4/.5/.6 rad/s with ten new paired initial conditions per cell and unchanged
  requested motion/scoring. This establishes the observed command envelope;
  passing ten cases is not a hardware guarantee.
- Compare one unguarded actor rollout with guarded optimization from identical
  initialization at seeds 44/45. Then run 16 guarded actor rollouts per seed at
  .05 and .10 of original exploration std. Std is frozen in all these new arms;
  this differs from the previous audit, where std could learn. Rewards,
  preserved critic, two warmup rollouts, and observation normalization match
  across arms. No command correction is used during these motor-policy audits.
- The guard measures analytic KL(old Gaussian || proposed Gaussian) on **every
  state in the entire rollout**, after each optimizer proposal. Limits: mean
  .01 and maximum state .1, cumulative against the rollout's original policy.
  Overshoots restore all weights and Adam state exactly, checked recursively,
  and retry at half the step size (at most nine attempts). An exhausted proposal
  leaves weights and optimizer unchanged. These are diagnostic constraints,
  not a mathematical guarantee of behavior retention on unseen states.
- Save checkpoints at 0/1/4/16 actor rollouts. Evaluate ten fresh quick seeds
  at each; final candidates receive 50 standard development and 50 separate
  standard confirmation cases per family. Standard panels have historical
  exposure; the fresh controller/sequence panels are separate from these.

Technical gate: every guarded rollout respects both KL bounds, every rejected
state restores exactly, and every actor rollout accepts a proposal and makes a
nonzero action change. Learning gate: at the same noise setting, both training
seeds must retain each family within 4pp on both standard panels, improve the
minimum family success by at least 4pp on both, and not reduce mean success.
Only learning-gate winners receive additional fresh, paired evaluations with
the command-correction wrapper, including 50 long sequences. The runner records
an explicit recommendation, all gate failures, checkpoints, and diagnostics.
Full controller qualification (48/50 per fresh family and each sequence panel)
is reported separately from evidence that a training update is useful.
No model publication or long training is automatic. Results pending at launch.

Preflight optimizer mechanical check completed before the main learning audits:
`logs/mujoco_rl/preflight_guard_smoke_v1/guard.jsonl`. After two critic-only
rollouts, the first actor rollout accepted 48 minibatch steps, rejected and
exactly restored 229 oversized proposals, and made nonzero action changes
(RMS .0001726). Maximum accepted mean KL was .009983 (limit .01); maximum
accepted per-state KL .062944 (limit .1). Final rollout KL was .008632.
This validates the safeguard mechanics for this rollout; skill improvement and
retention still require the pending physical evaluation panels.

### Preflight completion — reviewed 2026-10-13

Completed in 2,640.9 seconds (44.0 minutes). Status: **no-go for long training**.
No training is continuing and no checkpoint was promoted. Raw evidence:
`logs/mujoco_rl/command_walk_preflight_v1/result.json` and its per-stage panels.

The corrected frozen controller passed fresh cases as follows: forward 50/50,
left arc 46/50, right arc 50/50, left pivot 49/50, right pivot 50/50, stop 50/50,
mixed 50/50. Braking/restart passed 99/100 across five cases (one left-pivot
failure). However, the two fresh 60-second sequence panels passed 42/50 and
39/50, versus 41/50 and 40/50 for heading correction alone: both total 81/100.
The speed correction has not demonstrated a long-sequence advantage. The 48/50
controller qualification gate failed on left arcs and both sequence panels.
Across the candidate sequence panels, failure reasons included 13 left-turn yaw
tracking violations, three left-turn heading RMS violations, ten forward lateral
excursions, and one right-turn yaw violation. Reasons can overlap within an
episode. This points to both turning and transitions back to straight walking.

The dense grid confirmed asymmetry: at yaw demand .6 rad/s, left turns passed
4/10 at .20 m/s and 0/10 at .25/.30 m/s; corresponding right turns passed 10/10.
Both directions passed all ten sampled cases at .4/.5 rad/s for each speed.
This is a sampled envelope, not proof that every state inside it is reliable.

All four guarded PPO arms passed the mechanical/technical gate. Across 64
actor rollouts they accepted 3,072 minibatch updates and exactly restored
13,235 rejected proposals. The largest accepted mean KL was .009999 and largest
per-state KL .064294 (limits .01/.1); actions changed in every actor rollout.
Matched unguarded first actor rollouts had mean analytic KL 5.832 and 5.098.
Thus the guard controlled optimizer overshoot in this experiment.

Nevertheless, none of the four arms passed the learning gate. Confirmation
mean success: baseline 94.29%; .05-noise seed44/45 93.71%/93.14%; .10-noise
seed44/45 94.00%/92.57%. Only .10-noise seed44 retained every family within 4pp
on both panels, and it did not improve. All used 16 actor rollouts (98,304
actor-training steps) after two critic-only warmups. Smaller safe updates alone
have not solved the learning objective or skill retention. These short runs do
not prove longer training can never help; they fail to justify spending that
budget now. Conditional integration checks were skipped because no common
noise setting produced a learning-gate winner at both training seeds.

Next recommended direction: retain the guarded optimizer; build targeted short
training cases from left-turn and turn-to-forward failures, with explicit
rehearsal/behavior retention on already-successful motions. Compare those on
fixed failure cases plus full retention panels before another long run. Avoid
repeating unrestricted all-motion PPO on the unchanged objective.

## 2026-10-13 — Final moderate-turn campaign implementation

The agreed release range is forward 0.1–0.3 m/s and yaw ±0.15–0.5 rad/s,
plus stop and pivot commands. ±0.6 rad/s is outside this release. The campaign
has a 48-hour wall-clock ceiling, not a requirement to consume that time.
Historical full-range evaluators and reports remain separate.

The implementation uses one corrected whole-body controller during training
and evaluation (heading gain .5, internal turn-speed reduction .02 m/s), while
original requested motion owns trajectory integration, reward, and scoring.
A versioned inference bundle includes policy, matching normalization, correction
configuration, command validation, integrity hashes, and qualification status.

Training retains the tested full-rollout KL guard (.01 mean, .1 maximum state)
and adds hard reference-action constraints (RMS .01, per-state RMS p95 .02).
The reference contains 512 states per family from 16 successful episodes per
family. All guard rejections restore weights and Adam state exactly. Exploration
is fixed at .1 of source std; observation statistics are frozen and critic/return
statistics preserved. Each independent pilot uses eight critic-only rollouts.

Pilot: original versus explicitly phase-aligned reward, seeds 44/45, evaluations
after 16 and 64 actor rollouts. The aligned variant adds bounded heading,
yaw-tracking, straight lateral displacement, pivot displacement, and stopping
costs without changing the physical success evaluator. Both seeds must retain
every family within 4pp on development and separate regression panels, and
improve mean target success by 10pp or lower mean target violation by 20% without
lowering target success. Target panels include exit transitions, braking/restart,
and 60-second sequences. No replicated winner means stop, not more hours.

Motion refinement allocates expected simulation time 30% left-turn exit, 15%
right-turn exit, 15% braking/restart, 40% balanced rehearsal. Sequence refinement
uses 60% sequences and 40% rehearsal, progressing through 20/40/60 seconds.
Episode probabilities compensate for duration. Training pauses globally while
physical evaluation runs. Full retention gates confirm regressions before
restoring complete saved training state; two confirmed regressions or one
million steps without target progress stop the run. Best development candidates
are retained independently of the latest training checkpoint.

Wall deadlines are cumulative from launch: setup 2h, pilot 8h, motion 22h,
sequences 40h, qualification/delivery 48h. This transfers unused earlier budget
forward. Resume preserves the original deadlines and source hashes. A process
lock prevents duplicate workers. Resume restores weights, optimizer, statistics,
and curriculum and resets simulator environments at a rollout boundary; it is
not bitwise continuation of a mid-episode simulator snapshot.

Final qualification uses locked seed bases 90,000,001 (families/brakes),
94,000,001 and 98,000,001 (independent 60-second sequences), and 102,000,001
(command grid). None are used for selection or reference collection. Require
48/50 per family/braking case and per long-sequence panel; 19/20 per direction
and command-grid cell; no falls or self-collisions; and paired-success action
rate, action acceleration, and effort no more than 10% above baseline. Failure
does not trigger retuning on these holdouts. Early baseline qualification, pilot
failure, regression, plateau, or budget exhaustion can all end the campaign.

Validation: 57 related CPU tests passed, covering schedule range and duration,
time-weighted sampling, physical command/scoring invariance, reward formulas,
exact optimizer restoration, inference bundle reload/parity, reference-quality
checks, seed replication gates, and distinct sequence-stage checkpoint paths.
The fresh/resumed training integration check also passed before launch.
The first actor rollout accepted 48 steps and exactly restored 188 rejected
proposals; final mean KL .008543, reference RMS .000334. A resumed actor
rollout preserved training state and switched successfully to 20-second sequences.
Evidence: `logs/mujoco_rl/command_walk_final_check_v1/result.json`. The campaign will write its own SUMMARY.md, result.json, qualification
or pilot decisions, per-motion evaluations, immutable checkpoint pairs, and
labelled videos (including evaluated failure seeds when available).

Launched on 2026-10-13 in tmux `asimov_final_walk`, output
`logs/mujoco_rl/command_walk_final_v1/`. Initial moderate-range baseline
evaluation is running; 23/75 development batches had completed at the launch
check. The campaign then handles qualification or pilot/curriculum progression
according to its gates. Headless video rendering was also checked successfully
at 480×368; the validation frame is in
`logs/mujoco_rl/command_walk_final_check_v1/render.png`.

### Final campaign pilot results — 2026-10-13

The campaign stopped at `pilot_no_replicated_improvement` after approximately
100 minutes of setup/training/evaluation, followed by completed video rendering.
No extended motion or sequence refinement ran. No final holdouts were opened.
All four pilots completed 64 actor rollouts (393,216 actor-training steps each)
and passed the optimizer technical checks. All retained the seven established
motion families within the specified tolerance on both standard panels.

Confirmation results (50 cases per entry):

| Controller | Left-turn exit | 60-second sequence |
| --- | ---: | ---: |
| Corrected starting policy | 42 | 43 |
| Original reward, seed 44 | 47 | 44 |
| Original reward, seed 45 | 37 | 41 |
| Aligned reward, seed 44 | 48 | 42 |
| Aligned reward, seed 45 | 43 | 42 |

Every pilot passed forward, both walking arcs, and stop 50/50 on confirmation.
Original-44 and aligned-44 passed the pilot improvement gate on development and
confirmation (through the violation-reduction alternative). Neither seed-45 run
passed it on both panels, so neither reward configuration met the replicated
promotion rule. This is improved skill preservation and some seed-specific
transition gains, not replicated long-sequence improvement or final qualification.

The exported `controller/` remains the corrected starting policy with
`qualified=false`; trained candidate checkpoint pairs remain in `jobs/`.
All ten labelled videos completed in `videos/`. They are independent replays,
with failed evaluation seeds selected where available, not replacement evidence
for qualification. Relevant evidence: `pilot_decision.json`, `state.json`, and
`evaluations/pilot_confirmation_*.json` under the campaign output directory.

### GitHub model packaging — 2026-10-13

Added `checkpoints/moderate_walk_v1/` with all four actually trained pilot
checkpoints, their matching normalization files, correction metadata, and both
physical evaluation panels. The baseline is labelled separately. The portable
`checkpoints/moderate_walk_v1_bundle.zip` contains the same package and SHA-256
checksums. `original_44` and `aligned_44` passed their individual pilot improvement
checks; neither reward configuration passed across both seeds, and no artifact
is labelled qualified. Top-level and MuJoCo READMEs link to the downloadable
models and usage instructions. Replay now resolves packaged relative report
paths against the selected controller directory.
