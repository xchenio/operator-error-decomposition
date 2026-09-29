# Incremental nonlinear correction after nested linear tuning

Evidence addendum, 2026-09-29. Protocol/source were committed at e352c0b and all
selection artifacts at ad94e69, before opening three reserved500-trajectory sets.
These sets are now consumed and cannot serve as fresh holdouts in later experiments.
Earlier failed confirmations remain part of the record; this is a new intervention.

## Methods and estimands

We retain the disjoint representation400/correction600 design and the rank16 POD axes.
Within each outer correction fold, a linear coefficient map is selected by three-fold
inner validation using mean per-trajectory relative L2. The grid includes MSE and
guarded energy-weighted ridge, an unregularized truncated-SVD solution and17 positive
alphas from10^-8 to10^8. Weighted centering gives an unpenalized intercept. Inputs and
loss floors are estimated only from fitting rows. Shared alpha across modes is a
limitation: this finite search is not an optimal linear oracle.

Each outer training400 fits a zero-output width32 nonlinear residual over its tuned
frozen linear map, with two seeds and fixed1500 updates. Three held folds estimate
incremental modal risk, using guarded energy-weighted coefficient squared errors.
The hard rule requires positive gains in every fold and mean gain>=1%. Continuous
shrinkage clips the pooled weighted residual/delta cross-product divided by delta
energy to[0,1]. These are cross-fitted estimates over the chosen linear comparator,
not evidence that any physical mode intrinsically requires nonlinearity.

Final linear tuning uses correction600; five residual seeds use fixed1500 updates.
No-correction, all-output, hard-mask, energy-mask and shrinkage predictions reuse
these same fitted branches. This is posthoc output gating, not arm-specific retraining.
There are33 neural fits (18 CV probes and15 final), not75 independent arm fits;
1296 inner-grid candidates and12 selected linear refits reuse SVD factorizations.
Final linear choices are MSE/alpha100, MSE/alpha10000 and normalized/alpha1e-6.
An original-Burgers outer fold chooses the upper alpha boundary1e8.

Development selects hard versus shrinkage and falls back to no correction unless
gain over tuned linear is at least1%. Selection retains all five seeds. The locked
choices are hard/original Burgers, shrinkage/independent Burgers and none/Advection.
The primary confirmation estimand averages each trajectory's five seed errors, not
the error of an ensemble prediction. Success requires>=1% relative improvement and
a paired-trajectory bootstrap interval wholly below zero. Six comparisons (three
tasks versus linear and energy) use Bonferroni-adjusted two-sided99.1667% percentile
intervals,5000 resamples. Intervals condition on the fitted seeds, not retraining.

## Results

Development mean relative L2 (lower is better):

| Rule | Original Burgers | Independent Burgers | Advection |
|---|---:|---:|---:|
| Tuned linear / none | 0.373252 | 0.409382 | 0.019756125100 |
| All outputs | 0.363277 | 0.403154 | 0.019756122297 |
| Hard stable mask | 0.362262 | 0.402368 | 0.019756125076 |
| Energy mask | 0.362262 | 0.402368 | 0.019756125159 |
| Continuous shrinkage | 0.362297 | 0.402073 | 0.019756122211 |

Fresh confirmation mean relative L2, all locked arms reported descriptively:

| Rule | Original Burgers | Independent Burgers | Advection |
|---|---:|---:|---:|
| Tuned linear / none | 0.387273 | 0.379517 | 0.024182229260 |
| All outputs | 0.376687 | 0.369175 | 0.024182223957 |
| Hard stable mask | 0.375353 | 0.369348 | 0.024182228981 |
| Energy mask | 0.375353 | 0.369348 | 0.024182229270 |
| Continuous shrinkage | 0.375399 | 0.368605 | 0.024182224731 |

| Locked candidate versus linear | Relative gain | Adjusted interval for absolute error difference | Primary success |
|---|---:|---|---|
| Original Burgers: hard mask | 3.0779% | [-0.018826, -0.005641] | Yes |
| Independent Burgers: shrinkage | 2.8752% | [-0.016384, -0.005921] | Yes |
| Advection: no correction | 0% | [0, 0] | No improvement claim; fallback |

Hard selection keeps zero-based modes0–3 and0–1 on the Burgers splits, still identical
to energy truncation. Original candidate versus energy has zero difference. Independent
shrinkage versus energy gains only0.2013%, with adjusted interval[-0.001847,0.000219]:
it fails both the1% and interval requirements. No new selection-rule superiority is
established. There was no post-confirmation model selection or retraining.

On Advection, modes1/7 pass the coefficient-level hard rule, but field-level gain is
negligible. The predeclared fallback retains the linear prediction. This exposes the
need for an absolute/field-risk relevance criterion in addition to modal percentage
gain. It does not validate a novel selection rule or a second nonlinear PDE result.

Secondary mean RMSE for linear/locked candidate is0.149478/0.146517 (original) and
0.153602/0.150764 (independent); Advection remains0.027846. RMSE is descriptive and
does not substitute for primary or selection-rule evidence.

## Interpretation and next research boundary

The positive finding is now narrower and stronger: this controlled nonlinear correction
improves over this extensively tuned linear comparator on two disjoint Burgers training
and confirmation splits. It is not a comparison with the historical full-data FNO,
nor state-of-the-art evidence, nor universal generalization across nonlinear PDEs.
The two hard masks align with POD energy order; no individual mode has a separate
fresh-data significance claim. Basis labels themselves can change across new splits.

Residual/POD learning and mode selection already have related literature, discussed
in the prior [objective-modal addendum](FOLLOWUP_OBJECTIVE_MODAL_20260928.md).
The combination alone is not methodological novelty. A useful next hypothesis is
to reject modal corrections whose estimated reduction is immaterial to total field
risk, while testing continuous shrinkage against energy truncation at equal budgets.
That hypothesis is not evaluated here and must use new prospective data after locking.
Before a strong method paper, add a genuinely second nonlinear PDE and matched strong
operator-learning baselines. Preserve the present confirmation evidence unchanged.

Full protocol, checkpoints provenance and reproducible commands:
[INCREMENTAL_MODAL_V1](../docs/INCREMENTAL_MODAL_V1.md).
