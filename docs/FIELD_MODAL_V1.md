# Whole-field relevance and a second nonlinear PDE — prospective protocol

2026-09-29. Historical confirmation sets are consumed. This experiment reserves new
500-ID lists per task, excluding all existing public train/development/test/confirmation
IDs, historical Burgers smoke IDs0–127, and each other where they share a source.

## Change being tested

Reuse frozen nested-tuned linear maps and residual branches for the two Burgers splits
and Advection. Do not retrain or select using their consumed confirmation scores.
For each training CV fold, divide each mode's weighted coefficient-risk reduction by
the entire weighted baseline field squared error, INCLUDING orthogonal projection
error. Retain an old-stable mode only if it removes at least0.1% of this field risk in
EVERY fold. A large relative improvement on a negligible mode should no longer pass.
This uses an exact orthogonal squared-error decomposition, not a decomposition of the
mean nonsquared relative L2 metric.

Evaluate hard and continuously shrunken versions of this mask on shared OOF predictions.
Require at least1% mean per-sample relative-L2 improvement in EACH fold; otherwise set
that mask to zero. These OOF gates are internal selection, not unbiased final estimates
after the mask has been selected from the same folds. Fresh confirmation is necessary.
Thresholds are fixed before result inspection, with no threshold sweep.

Compare none/all, old stable/old energy/old shrinkage, field mask/field shrinkage and a
leading-energy control matching the surviving field-rule support count. If neither
field rule survives, its energy control has k=0. The two field rules share support
unless one is rejected by its joint gate. All use the same branch predictions; there
are no arm-specific retraining or parameter-budget advantages.

Development chooses field versus field_shrink, with a1% improvement requirement versus
none or fallback to none. Separately lock the old stable/shrink selector with the same
fallback. Commit candidates/masks/all artifacts before new confirmation. Three claims
per task (versus tuned linear, energy, old selected rule), four tasks:12 comparisons.
Require>=1% relative gain AND a paired-trajectory bootstrap interval wholly below0;
Bonferroni two-sided99.5833% intervals,10000 resamples,seed2026092919. Average errors
over the five fixed seeds per trajectory; do not treat seeds as independent data units.

## Second nonlinear PDE: synthetic Allen-Cahn

Use periodic u_t=0.01 u_xx+5u(1-u^2), x in[0,2), t=0→0.5,256 spatial points.
Smooth randomized12-mode Fourier initial conditions define1000 train/200 development
and500 reserved trajectories. Each ID has a deterministic independent RNG seed;
reserved targets are NOT generated in development. Representation400/correction600,
rank16, nested linear tuning and11 neural fits are identical to the previous protocol.
This is a custom synthetic operator dataset, not a reproduction of a public benchmark
or of a single-initial-condition PINN task. The nonlinear equation family is documented
in the [PINNs authors' source](https://github.com/maziarraissi/PINNs/blob/master/docs/index.md);
our diffusion coefficient and initial-condition distribution are explicitly different.

Strang splitting uses exact Fourier diffusion half-steps and the analytic pointwise
reaction flow. Step0.001; preserve50-trajectory chunks. QA uses separate IDs100000–100015:
constant logistic flow, equilibria0/±1, exact pure diffusion, step halving,256→512 grid
refinement and an independent DOP853 method-of-lines solve. Maximum step-refinement
relative error2.471e-7, grid error5.721e-14; independent solver error3.091e-7. All passed
predeclared1e-4/1e-5 tolerances before neural training. These finite QA checks are not
a mathematical error bound for every random trajectory.

The public1D reaction-diffusion alternative was not selected because of an unresolved
[upstream equation/data consistency report](https://github.com/pdebench/PDEBench/issues/94).
That report is a warning, not our own proof that every published file is incorrect.
No downloaded public reaction-diffusion benchmark result is claimed here.

## Provenance and interruption recovery

New family results/public_field_modal_v1 preserves all prior families. Old checkpoints
are referenced by hash, not copied/retrained. Allen-Cahn model/optimizer/RNG state every
50 steps; training and data chunks resume. Lock/source/selection hashes enter Git before
confirmation; targets become consumed as soon as extraction/generation starts.

Commands (D:/miniconda3/envs/pytorch/python.exe):

```text
scripts/allen_cahn_data.py
scripts/run_field_modal.py plan
# commit protocol and source
scripts/run_field_modal.py develop
scripts/run_field_modal.py seal
# commit selection lock
scripts/run_field_modal.py confirm
```

## Development, before new confirmation

Source/protocol commitb2d554c. Reused33 historical neural checkpoints across three
public tasks; only11 new Allen-Cahn fits (16500 updates). Independent audit replays
field risks, modal gains, all joint gates, nested linear choices and final predictions.

| Task | New retained modes (one-based) | Linear | Field rule | Matching energy | Old selected rule |
|---|---|---:|---:|---:|---:|
| Original Burgers | 1,2,4 | 0.373252 | 0.362597 | 0.364460 | 0.362262 |
| Independent Burgers | 1,2 | 0.409382 | 0.402368 | 0.402368 | 0.402073 |
| Advection | none | 0.019756125100 | 0.019756125100 | 0.019756125100 | 0.019756125100 |
| Synthetic Allen-Cahn | 1–15 | 0.458885 | 0.359218 | 0.359218 | 0.359048 |

Hard and field-shrunken masks give identical predictions in these four cases; tie
selection picks field on the nonlinear tasks and none on Advection. New thresholds
remove original Burgers mode3, Advection's irrelevant modes2/8 and Allen-Cahn mode16.
Only original Burgers now differs from the matched-count energy mask. Every new rule
is slightly worse than the old selected rule in nonlinear development means. This
is disclosed before confirmation; no threshold or candidate rule is changed to repair it.

## New confirmation, completed after seal8a4020b

Four500-ID sets were extracted/generated ONLY after committing all selection artifacts.
All2000 targets are now consumed, in addition to the earlier consumed confirmation sets.
No threshold changes, retraining or model reselection followed opening.

| Task | Linear | Locked candidate | Gain over linear | Primary |
|---|---:|---:|---:|---|
| Original Burgers | 0.405879 | 0.393444 | 3.0636% | Pass |
| Independent Burgers | 0.402369 | 0.389518 | 3.1939% | Pass |
| Advection | 0.020532878061 | 0.020532878061 | 0% | No gain; rejection |
| Synthetic Allen-Cahn | 0.467579 | 0.366325 | 21.6550% | Pass |

All comparisons against matched energy and old selected rules FAIL the predeclared
success criteria. Original Burgers versus energy improves0.1904%, interval crosses0;
independent Burgers and Allen-Cahn equal energy exactly. Allen-Cahn's old all-mode rule
has0.366134 versus new0.366325: a0.0521% deterioration, adjusted difference interval
[0.000153,0.000233]. Small helpful coefficients should not automatically be removed
merely because their absolute contribution is small. Statistical detectability is
not evidence of a practically large harmful effect.

Independent audit replays all frozen CV/development/confirmation predictions, field
risk decompositions, gate decisions and12 paired bootstrap statistics. Full arm tables,
intervals and limitations are in the [paper addendum](../paper/FOLLOWUP_FIELD_MODAL_20260929.md).
Next proposals are continuous field-aware shrinkage and explicit uncertainty/cost
objectives, not post-confirmation threshold tuning. A downloaded second nonlinear
benchmark and matched strong nonlinear baselines remain future validation, not completed work.
