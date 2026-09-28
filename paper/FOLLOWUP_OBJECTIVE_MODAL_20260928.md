# Objective alignment and cross-fitted modal correction

Development-only addendum, 2026-09-28. The earlier failed confirmation remains a failed
confirmation; none of the newly reserved confirmation targets are evaluated here.
Protocol and complete evidence: [OBJECTIVE_MODAL_V1](../docs/OBJECTIVE_MODAL_V1.md).

## Question and design

Ordinary squared field error weights high-energy trajectories more heavily than the
reported mean per-trajectory relative L2. We compare it with a guarded normalized squared
loss, retaining the frozen rank-16 ridge map, zero-output width-32 residual branch,
regularization and baseline-inclusive early stopping. Let E_i=mean(y_i^2); weights are
proportional to 1/max(E_i,epsilon), normalized to training mean one. Epsilon is the
training-only fifth percentile of energy, bounded below by 1e-12. The same weights
multiply prediction error and correction penalty. This is a squared surrogate, not
exact equality with the nonsquared primary metric. No causal explanation of the earlier
confirmation failure follows from this comparison alone.

The full-data loss comparison uses 1000 training trajectories. A separate strict modal
experiment fits the POD basis and ridge anchor on 400 trajectories, and trains corrections
on the remaining 600. Three internal folds, each with two fixed-1500-step seed probes,
estimate held-fold weighted coefficient-risk reduction. Fold targets never choose a
checkpoint, basis, normalization or loss floor. We retain a mode only if its gain is
positive in all folds and its mean gain is at least 1%. Final corrections compare all
modes, this stable mask, random masks of matching size, and leading-energy masks of
matching size; five seeds per arm. Full-data and split-budget scores cannot isolate
the effect of mode selection and must not be directly treated as a paired comparison.

The second Burgers training/development split excludes all historical sample IDs and
consumed confirmation IDs. The second PDE is fixed-time linear Advection: a useful
negative control for unnecessary nonlinear correction, not a second nonlinear PDE.
Expanded pure-ridge alpha grids are necessary because correcting an over-regularized
linear anchor would not demonstrate that nonlinearity is needed.

## Completed development results

All 123 neural records completed (41 per task: 10 full-data comparisons, 6 CV probes,
25 correction-panel runs), plus 48 pure-ridge grid refits. These are run records, not
123 independent experimental units. There are 25 redundant correction-panel records:
five stable/energy repeats per Burgers split and fifteen repeats of normalized/all on
Advection, where every mask selects all modes. Source/protocol were locked before runs;
an independent audit verifies frozen anchors, folds, checkpoints and aggregates.

Mean development per-sample relative L2 across five seeds (lower is better):

| Full1000 protocol | Original Burgers | Independent Burgers | Advection |
|---|---:|---:|---:|
| MSE residual | 0.366400 | 0.407464 | 0.169283 |
| Guarded normalized residual | 0.358258 | 0.398468 | 0.175051 |
| Development-tuned pure ridge | 0.373600 | 0.409068 | 0.015378 |

Seed standard deviations for MSE/normalized are respectively 0.003278/0.000460,
0.002837/0.001796 and 0.000283/0.000597. They describe optimizer variation, not
confidence intervals over fresh trajectories. The deterministic ridge row has no
optimizer-seed uncertainty and uses a separate eight-alpha development grid.

| Anchor400 / correction600 protocol | Original Burgers | Independent Burgers | Advection |
|---|---:|---:|---:|
| MSE, all modes | 0.370971 | 0.411704 | 0.170576 |
| Normalized, all modes | 0.362092 | 0.404991 | 0.175883 |
| Normalized, stable modes | 0.360077 | 0.403133 | 0.175883 |
| Normalized, random k modes | 0.375807 | 0.412182 | 0.175883 |
| Normalized, top-energy k modes | 0.360077 | 0.403133 | 0.175883 |
| Pure ridge fitted to anchor400 only | 0.375627 | 0.409678 | 0.019756 |
| Selected k | 4 | 2 | 16 |

The anchor400 ridge control uses fewer labels than anchor-plus-correction, so it is
not a matched-data proof of nonlinear necessity. The full1000 comparison is the
stronger linear control. Selected ridge alphas at full1000/anchor400 are 100/1000,
1000/1000 and 0.0001/0.0001 respectively. Several optima lie on grid boundaries;
no global optimality claim is made, and tuning costs differ.

On Advection, normalized loss worsens relative L2 and RMSE in all five full-data pairs,
and mean relative L2 worsens in every energy quintile. CV selects all 16 modes: stable,
random, energy and all-mode normalized panels have identical histories and scores.
The full-data alpha10 anchor has relative L2 0.341140; correction helps that anchor,
but the retuned linear map reaches 0.015378 without a nonlinear branch. Thus the rule
detects correctable anchor error, not specifically nonlinear learnability. It does not
support a claim that these modes require nonlinear correction. This negative control
does not rule out residual learning with a stronger anchor or different regularization.

All three new 500-trajectory confirmation sets remain unopened. The independent
Burgers split and second-PDE scores are development evidence, not sealed confirmation.

## Interpretation and novelty boundary

On both Burgers splits, normalized loss improves the full-data development relative L2
in all five matched seeds. The lowest-energy group benefits most; the middle-energy
group worsens slightly. This supports objective alignment as a practical intervention
on these development sets, not a universal improvement or an unseen-data guarantee.

The CV mask selects the leading four modes on the original split and the leading two
on the independent split. Thus it equals the energy-truncation control. Ten paired
stable/energy histories and scores are exact repetitions, not independent evidence.
Improvement over random masks does not establish superiority over energy selection;
stable versus all-mode correction improves only three of five seeds on the new split.

POD-plus-neural residual correction already appears in
[Demo, Tezzele and Rozza (2023)](https://arxiv.org/abs/2302.12682).
[Koike et al. (2026)](https://arxiv.org/abs/2605.27756) also study informative POD-mode
selection with neural networks and sparsity. Our cross-fitted correction target differs,
but neither residual learning nor broad learned mode selection can be claimed as new.
The current rule is a tested research hypothesis, not a validated methodological novelty.

## Next decision

Do not enlarge the residual network merely because its training fit can improve. First
separate errors removable by retuning the linear anchor from errors requiring nonlinear
correction. A future version should cross-fit the incremental benefit over a well-tuned
linear correction and allow explicit rejection of correction. Compare hard CV masks with
energy truncation and continuous shrinkage at equal data and tuning budgets. This is a
proposal, not an implemented or confirmed improvement. Seal models, comparators, primary
metric, effect-size threshold and paired trajectory inference before opening the reserved
sets. Multiple seeds on one development split are not independent dataset replications.
