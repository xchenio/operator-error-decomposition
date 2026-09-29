# Objective alignment and modal learnability — locked development protocol

> Historical2026-09-28 development record. On2026-09-29 the reserved sets were opened
> only after a new candidate seal in INCREMENTAL_MODAL_V1.md. They are now consumed;
> references to unopened sets below describe the earlier stage, not their current status.

## Development-only diagnosis

The previous selected residual improves the lowest-energy development quintile from
relative L2 0.698359 to0.677225 (47 points); the highest-energy group worsens from0.118216
to0.121598 (42 points). Thus energy strongly affects relative-error scale, but the observed
development evidence does not establish that low-energy cases caused confirmation failure.
Quintile boundaries and the proposed5th-percentile floor are fitted on training targets.
The latter equals0.0155348191 mean target-square units on the original1000 trajectories.

## Prospective experiment (no new confirmation labels)

- Tasks: original Burgers split; fully disjoint Burgers train1000/development200 split;
  public PDEBench Advection beta0.4, train1000/development200. All use t0→t1, stride4,
  rank16 and alpha10. Advection is a **linear transport negative control**, not evidence
  of success on a second nonlinear PDE. Its pinned official dataset is
  https://huggingface.co/datasets/pdebench/Advection/tree/33d90a245d98b8a250884eab613c6f9c26a13b68 .
- The independent Burgers partition excludes every historical train/dev/test ID, all
 500 already-consumed confirmation IDs, smoke prefix0–127 and the new reserved original-task
 confirmation IDs. Separate500-trajectory confirmation lists are reserved per task, but
 this stage extracts/trains/evaluates **only train and development**, never those targets.
 The independent task is a new training split, not a prospective confirmation score.
- PartA: full1000 training basis/ridge, width32 one-hidden-layer zero-output correction,
  penalty1, compare MSE versus guarded normalized squared error, five paired seeds each.
  This extends the previous chosen configuration without another validation grid search.
- Let E_i=mean(y_i²), epsilon=max(training quantile_0.05(E),1e-12).
  Normalized weights w_i=(1/max(E_i,epsilon))/training_mean(1/max(E,epsilon)).
  Train mean_i w_i[||delta_i-target_residual_coef_i/s||²+||delta_i||²]/256.
  MSE uses w_i=1. Both retain AdamW1e-4 weight decay, batch32, cosine1e-3→1e-5,
  3000-update maximum and prior baseline-inclusive development early stopping.
  Multiplying both error and correction penalty by w preserves relative regularization.
  This is guarded **squared** relative error, not the mean nonsquared relative L2 metric;
  success is not guaranteed and all original metrics remain reported.
- PartB strict CV: fixed seeded permutation splits training1000 into representation/anchor400
  and correction600. Fit Q, input statistics and ridge only on anchor400. Never refit these
  using correction labels. Split correction600 into three200-point folds. For each fold,
  train residual on the other400 points with two paired seeds, normalized loss, fixed1500
  steps. Fold evaluation labels never control stopping/checkpoint selection. The per-fold
  denominator floor and scalar output scale use its400 fit targets, not held labels.
- For each mode/fold compute weighted held coefficient squared risk versus frozen ridge,
  averaging the two seed risks before calculating relative gain. Select modes whose gain
  is positive in all three folds AND mean fold gain>=1%. No threshold tuning on development.
  Common Q is valid here because it sees a disjoint anchor400 set; ordinary full1000 POD
  followed by apparent internal CV would leak held-fold target information into the basis.
- Retrain on correction600 with five seeds and evaluate on the unchanged development200:
  MSE/all modes; normalized/all modes (uniform regularization); normalized/stable modes;
  normalized/random modes (same k, fixed per-seed random masks); normalized/top-energy modes
  (same k, POD order from anchor400). Same output dimension and network parameter count;
  masked outputs contribute no correction. If k=0 or16 controls can degenerate; disclose it.
  Compare these arms **within the anchor400/correction600 protocol**. PartA has a different
  data budget and basis; differences between PartsA/B cannot isolate modal selection.
- After the neural panels, add eight-alpha pure-ridge development grids at both full1000
  and anchor400 data budgets. These controls do not feed CV or residual fitting. In particular,
  beating alpha10 on the linear Advection task does not establish a need for nonlinearity.
- Per task:10 PartA +6 CV +25 PartB =41 neural records. Three tasks planned=123 records.
  CV overhead must be reported; random/energy controls share k derived by CV and are
  not independent tuning experiments. Multiple masks on one seed are not independent units.
- Persist optimizer/model/RNG/patience/best state every50 steps; evaluation every100.
  Data extraction caches50-trajectory chunks. Source/cache/mask provenance is mandatory.
  No historical training source is edited, and all completed items skip after identity checks.

Commands:

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py plan
# Commit protocol/source first, then each task can resume independently:
D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py burgers_original
D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py burgers_independent
D:/miniconda3/envs/pytorch/python.exe scripts/download_advection.py
D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py advection
```

This tests a proposed diagnostic rule, not an established novel algorithm. Original
development data have been reused; new task development sets also cease being untouched
once viewed. A later confirmation stage must commit fixed models and predefined claims
before opening reserved targets. No cross-PDE confirmation claim is licensed by this stage.

## Original Burgers development results (completed41 runs)

Full1000 comparison: MSE relative L2=0.366400±0.003278; normalized=0.358258±0.000460.
RMSE also improves from0.145048 to0.143010. These are development-selected five-seed
metrics, not fresh confirmation or an isolated proof of the cause of previous failure.

Within the strict anchor400/correction600 protocol:

| Arm | Development relative L2 |
|---|---:|
| MSE, all modes | 0.370971 ± 0.000997 |
| Normalized, all modes | 0.362092 ± 0.001604 |
| Normalized, stable modes | 0.360077 ± 0.001411 |
| Normalized, random k modes | 0.375807 ± 0.001481 |
| Normalized, top-energy k modes | 0.360077 ± 0.001411 |

CV selects zero-based modes0/1/2/3. Stable and top-energy masks are identical, so their
five paired runs are duplicates, not independent evidence for a superior modal rule.
Expanded ridge controls select alpha100/full1000 (0.373600) and alpha1000/anchor400
(0.375627); the latter lies on the declared alpha-grid boundary. Pure-ridge controls
and residual early-stopping procedures have different tuning costs.

### Targeted novelty update

[Koike et al., Sparse POD Mode Selection and Manifold Dimensionality Reduction with Neural
Networks,2026](https://arxiv.org/abs/2605.27756) proposes SparseModesNet, with neural nonlinear
decoding and sparsity-based informative-mode selection rather than energy truncation alone.
Our cross-fitted correction mask has a different target and training procedure, but the
broad idea of neural/POD mode selection is not new. Combined with the energy-mask equality
above, no first-mode-selection or validated-new-method claim is justified. This is a
targeted literature check, not an exhaustive novelty review.

## Independent Burgers development replication (completed41 more runs)

The1000 training and200 development IDs are disjoint from all prior splits and consumed
confirmation IDs. Both Q and coefficient models are fitted anew, with the same predeclared
hyperparameters. This is a new training split, not simply a new evaluation set for old weights.

| Protocol / arm | Development relative L2 |
|---|---:|
| Full1000, MSE | 0.407464 ± 0.002837 |
| Full1000, normalized | 0.398468 ± 0.001796 |
| Anchor400/correction600, MSE all | 0.411704 ± 0.000599 |
| Anchor400/correction600, normalized all | 0.404991 ± 0.001940 |
| Anchor400/correction600, normalized stable | 0.403133 ± 0.002563 |
| Anchor400/correction600, normalized random | 0.412182 ± 0.003265 |
| Anchor400/correction600, normalized energy | 0.403133 ± 0.002563 |

The normalized full-data objective improves relative L2 in5/5 paired seeds on each split
(10/10 total); RMSE improves4/5 on the original split and5/5 on the new split.
The strict masked rule improves L2 over normalized/all in5/5 original and3/5 new seeds,
and over random masks in5/5 on both. It selects modes0/1 on the new split, again equal
to the top-energy mask. Across both splits, ten energy/stable pairs have exactly identical
histories and final metrics. This does not establish incremental value over energy truncation.

Post-training development stratification of full-data models: normalized loss lowers the
lowest-energy bin from0.677225→0.647515 on the original split and0.778757→0.752588 on the new
split. The middle bin worsens slightly on both. The benefit is not uniform across samples;
these strata were inspected after fitting and are descriptive, not a new primary endpoint.
Expanded pure-ridge on the new split selects alpha1000 at both budgets (grid boundary):
full1000 relative L2=0.409068; anchor400=0.409678.

At the preceding checkpoint82 neural records were complete and Advection was pending.
The final status and third-task evidence follow.

## Completed Advection negative control and final audit

All123 planned neural records completed, plus48 deterministic ridge grid refits.
Advection download SHA256 and pinned revision verified. Its41 records used89600 updates;
the overall total is236600 updates. No newly reserved confirmation targets were loaded.

| Advection development arm | Relative L2 |
|---|---:|
| Full1000 MSE residual | 0.169283 ± 0.000283 |
| Full1000 normalized residual | 0.175051 ± 0.000597 |
| Full1000 tuned ridge, alpha0.0001 | 0.015378 |
| Anchor400/correction600 MSE all | 0.170576 ± 0.000087 |
| Anchor400/correction600 normalized all/stable/random/energy | 0.175883 ± 0.000473 |
| Anchor400-only tuned ridge, alpha0.0001 | 0.019756 |

Normalized loss worsens full-data L2 and RMSE in5/5 pairs and mean L2 in all five
energy strata. CV selects every mode, so the four normalized arms are identical;
selection comparisons are degenerate. Across the entire family25 panel records are
redundant repeats (10 on Burgers,15 on Advection), not independent evidence.

The full-data alpha10 anchor is0.341140; a nonlinear correction improves this weak
anchor but is much worse than the alpha-retuned pure linear control. Both selected
Advection alphas lie on the lower grid boundary. This demonstrates why gain relative
to a fixed weak ridge anchor does not establish a need for nonlinear correction.
Future work should estimate incremental gain over a tuned linear correction, include
rejection/no-correction and continuous shrinkage controls, and hold data/tuning budgets
constant. These extensions have not been trained in this round.

Audit verifies123 records, immutable frozen bases, fold separation, fixed-step CV,
selected checkpoint hashes, masks and group summaries. Post-training energy-bin
reaggregation reproduces each overall development score. Evidence is in audit.json,
summary.json and objective_energy_strata.json under results/public_objective_modal_v1.
Paper tables and all caveats are synchronized in
[the dated addendum](../paper/FOLLOWUP_OBJECTIVE_MODAL_20260928.md).
