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
