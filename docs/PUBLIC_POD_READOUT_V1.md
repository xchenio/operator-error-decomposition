# Same-POD readout comparison — protocol locked 2026-09-28

Question: with representation fixed, can a nonlinear input-to-coefficient map improve
over linear ridge? This is a supervised, validation-only diagnostic, not a new-method claim.

- Reuse the exact float64 Q, per-input-position training mean/std and input spatial-mean
  offset from `public_pod_ridge_v1` checkpoints, at ranks 16/32/64. POD is fitted to
  training residuals only; no validation or test target enters a fit.
- Compare each rank's validation-selected ridge alpha from the existing six-alpha grid
  against zero-output MLP (three 128-wide tanh hidden layers) and an additional zero-output
  affine linear branch trained with the same Adam protocol. Report all ranks/seeds.
- Each neural arm: seeds 0–4, 1000 training / 200 validation trajectories, original cache
  row order, batch32 without replacement within each step, 3000 updates, cosine LR
  1e-3 to 1e-5 inclusive. No weight decay, early stopping or best-seed selection.
- Both Adam arms use identical sampled indices for each seed. One scalar equal to the
  training target standard deviation scales all coefficients. No mode-wise scaling.
  Loss is mean sample sum of normalized coefficient squared errors / 256, equal to
  normalized field MSE minus its fixed projection residual. The coefficient targets are
  computed with the stored float64 Q then cast to float32 for Adam. Evaluation reconstructs
  float64 coefficients with the exact stored Q; it verifies squared Frobenius closure.
- Log train/validation predictions, projection oracle and coefficient error every 250
  updates. Save atomic model/optimizer/RNG/step/history/POD checkpoints every 50 updates.
  Same command resumes interrupted runs and skips identity-matched completed runs.
- Add the Adam affine control because comparing closed-form penalized ridge to an
  unpenalized neural optimizer alone confounds nonlinearity, regularization and optimization.
  This control reduces, but does not eliminate, confounding (parameterization and implicit
  regularization differ). Linear arms have no nonlinear hidden layers.
- Ridge's six-alpha validation selection differs from the fixed neural settings. Alpha10
  is the old grid boundary; neither ridge nor MLP is claimed optimally tuned. Validation
  has been reused extensively; improvement is development evidence, not held-out proof.
- Initial spatial-mean baseline and rank-matched oracle anchor interpretation. Lower oracle
  error alone does not certify a learnable map. Residual train coefficient error is not
  automatically pure optimization error; capacity and inductive bias may contribute.

Command (run from repository root):

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_public_pod_readout.py
```

Results will be stored in `results/public_pod_readout_v1`; the protocol declares 30 neural
runs plus reuse/reproduction of the existing 18 deterministic ridge candidates. New test
arrays are not loaded. No comparison to previous test-set FNO scores is made here.

## Completed results (2026-09-28)

All 30 runs completed under source commit `5ad1676`. The three selected ridge
checkpoints reproduced their original train/validation metrics exactly. All arms at a
given rank have identical POD checkpoint hashes, preprocessing and projection oracles.
The historical float32 spatial-mean baseline differs from the new float64 mean baseline
by only 1.12e-12 in validation relative L2 (0.37450899472); audit tolerance is 1e-10.

Values below are validation mean per-sample relative L2; lower is better. Neural values
are mean ± sample SD over five optimizer seeds on one fixed split. Ridge is deterministic;
each rank selected alpha10 from the existing six-alpha grid. Oracle uses target labels
offline and is not a deployable predictor.

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

### Interpretation

1. MLP improves training relative L2 and RMSE against the paired Adam affine branch in
   15/15 comparisons, but worsens both validation metrics in 15/15. No final MLP beats
   rank-matched ridge or the initial-mean baseline. This is consistent with a
   generalization failure despite improved training fit, not evidence that adding
   nonlinearity solves the coefficient-map problem.
2. For the MLP, coefficient mismatch accounts for 93.18–93.87%, 97.24–97.44% and
   99.01–99.11% of total validation squared Frobenius error at ranks16/32/64 respectively.
   These are exact orthogonal error components within floating-point accuracy, not an
   additive decomposition of mean relative L2. Maximum squared closure error across
   all evaluated runs/steps/splits is 1.32e-16.
3. Training is also not fully solved: 0/15 MLPs reaches coefficient relative-Frobenius
   error ≤0.01. Generalization and residual training fit can both be limiting. The
   remaining coefficient error cannot be labeled purely optimization or purely capacity.
4. Increasing rank sharply reduces oracle error without improving learned validation
   predictions. Under this protocol, prioritizing a better learnable coefficient map
   is more justified than adding POD modes. This does not establish that POD representation
   is sufficient for all accuracies, nor that nonlinear maps are intrinsically inferior.
5. Earlier checkpoints are diagnostic only; no post-hoc best-checkpoint numbers replace
   the declared final-step results. Validation was repeatedly reused, so these seed counts
   are descriptive, not independent-split significance or held-out test evidence.

### Most useful next experiment (not yet run)

Keep the same POD basis, start with rank16, and compare a frozen ridge coefficient map
against **ridge + a zero-initialized small nonlinear residual**. Select only a small,
predeclared regularization/width grid with early stopping on a designated development
split, giving ridge a comparable tuning budget. Use an untouched confirmation split
before drawing a generalization claim. This tests whether a restrained nonlinear
correction can preserve the linear baseline while improving on it, without assuming
that additional training or width alone will help. Expansion to ranks32/64 should
follow only if this control succeeds.

Checkpoint files remain local under ignored `checkpoints/` folders. Versioned manifests,
250-step histories, 50-step atomic progress snapshots, summary and independent audit
are under `results/public_pod_readout_v1`. This family uses progress/history files rather
than an append-only event stream. Historical experiment directories are unchanged.
