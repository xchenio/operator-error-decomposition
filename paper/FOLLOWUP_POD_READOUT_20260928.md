# Same POD basis: linear and nonlinear coefficient readouts

The proposed ridge-residual experiment below has now been completed; see the
[fresh-confirmation addendum](FOLLOWUP_RIDGE_RESIDUAL_20260928.md). Its primary improvement
was not confirmed; the observations and record counts below describe the earlier stage.

**2026-09-28 exploratory addendum.** Thirty additional neural records (15 affine Adam
controls and 15 nonlinear branches) extend the prior 175 neural records to 205, including
the previously reported 20 repeated controls. The 18 deterministic ridge fits are reused,
not counted as new runs. No new test-set evaluation was performed.

## Controlled protocol

For each rank16/32/64, all methods share the exact training-only residual POD basis,
per-position training input normalization and input-spatial-mean offset. The nonlinear
branch is a three-hidden-layer, width128 tanh MLP. MLP and affine Adam controls use
zero output initialization, five seeds, batch32, 3000 updates, the same sampled indices
within each seed and cosine learning-rate decay from 1e-3 to 1e-5. Physical field
reconstruction uses the stored float64 Q; training uses float32 coefficient predictions.
Coefficient loss differs from normalized physical field MSE only by a fixed projection
term. We report the final step, not a selected checkpoint.

## Results

Validation mean relative L2 (mean ± sample SD for neural arms; lower is better).
Ridge alpha10 was selected separately at each rank from the existing six-alpha grid;
this does not constitute matched tuning or establish an optimal ridge solution.
The oracle uses validation labels and is only a representation diagnostic.

| POD rank | Validation oracle | Ridge (selected alpha) | Adam linear | Nonlinear MLP |
|---|---:|---:|---:|---:|
| 16 | 0.108982 | 0.373656 | 0.376468 ± 0.000273 | 0.398761 ± 0.007220 |
| 32 | 0.067159 | 0.373736 | 0.376728 ± 0.000274 | 0.401114 ± 0.005065 |
| 64 | 0.036648 | 0.373751 | 0.376800 ± 0.000271 | 0.402642 ± 0.008874 |

| POD rank | Ridge train L2 | Adam linear train L2 | MLP train L2 | MLP train coefficient error (relative Frobenius) |
|---|---:|---:|---:|---:|
| 16 | 0.389775 | 0.388387 | 0.236997 | 0.123122 |
| 32 | 0.389695 | 0.388284 | 0.233625 | 0.131276 |
| 64 | 0.389663 | 0.388226 | 0.234100 | 0.134906 |

The nonlinear branch improves training relative L2 and RMSE in all 15 comparisons
against the matched affine Adam branch, yet worsens both validation metrics in all 15.
All final nonlinear models also trail ridge and the initial-spatial-mean baseline
(validation relative L2 0.374509). Better training fitting is therefore not sufficient
for better generalization, even when the decoder space is held fixed.

## What this localizes—and what it does not

Writing c*=(y-offset)Q and using orthonormal Q gives the exact squared-error identity:
total field error squared = projection residual squared + coefficient mismatch squared.
After division by total target energy, coefficient mismatch contributes 93.18–93.87%,
97.24–97.44% and 99.01–99.11% of nonlinear validation squared Frobenius error at
ranks16/32/64. The maximum numeric closure discrepancy is 1.32e-16. This identity is
not an additive decomposition of per-sample relative L2 and is not a pure optimization
decomposition.

These observations localize most prediction error to the learned coefficients under
the fixed POD representation. They do not identify a unique causal failure mechanism:
none of the 15 MLP runs fits training coefficients to the predefined 0.01 relative
Frobenius threshold, and capacity, finite optimization, regularization and input
inductive bias remain entangled. The affine Adam control narrows the comparison, but
different parameterizations can still have different optimization and implicit
regularization. Five optimizer seeds on a reused validation split do not establish
independent-split or held-out significance.

The next discriminating control is a small zero-initialized nonlinear correction to a
frozen ridge map, with a predeclared regularization/early-stopping protocol and an
untouched confirmation split. Simply increasing POD rank or branch size is not
supported as the first intervention by these results. This is evidence for a
fixed-representation diagnostic study, not a new POD-DeepONet algorithm or SOTA claim.

See [the protocol and full interpretation](../docs/PUBLIC_POD_READOUT_V1.md).
Raw histories, provenance and the independent audit are in public_pod_readout_v1.
