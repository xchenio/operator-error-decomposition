# Evidence audit v2 — 2026-09-22

## Completed

Recomputed 35 saved checkpoints from expQ/expV/expW, on all train/val/test grid points.
Results are in `results/audit_v2/`. Every result includes checkpoint/data SHA-256 and code commit.
Historical JSON files remain unchanged. Completed audit files are skipped on rerun; use a new
output directory when changing the audit definition.

The new calculation uses the complete solution matrix, SVD projection, a constant feature
(relaxing the shared scalar bias to independent sample coefficients), and the same norm and
samples for each decomposition. Squared relative Frobenius error is split into orthogonal
projection error and in-space coefficient discrepancy. The latter is not pure optimization
error. The projection is conditional on the recorded numerical rank tolerance (1e-12).

| 2D group | Checkpoints | Mean projection relative Frobenius | Mean actual relative Frobenius | Mean projection fraction of squared error |
|---|---:|---:|---:|---:|
| MLP baseline | 5 | 0.858195 | 1.002545 | 73.28% |
| Analytic alignment | 5 | 0.661478 | 0.810329 | 66.75% |
| Random smooth alignment | 5 | 0.634039 | 0.779696 | 66.20% |
| Fourier | 5 | 0.475013 | 0.682779 | 48.40% |
| Orthogonalization | 5 | 0.839136 | 1.001122 | 70.29% |
| IC weight x10 | 5 | 0.856419 | 1.002108 | 73.05% |

These are new metrics, not replacements for historical sample-mean relative L2 values.
The old phrase “85% of error is representation” must not be used as an orthogonal attribution.
The corrected result still motivates representation interventions in the tested 2D regime.
It does not establish that the architecture lacks capacity, or that changing basis is causal.
1D checkpoint counts and widths differ between groups; no pooled 1D method comparison is claimed.

## Historical statistics

`scripts/audit_statistics.py` resolves default configurations, separates budgets and code hashes,
keeps exact duplicate provenance, and quarantines conflicting repeat values. It emits 13 groups,
11 comparisons, and one conflicting seed group. All these historical groups lack code hashes,
so their statistics are explicitly exploratory, including Holm-adjusted p-values.

The conflict is baseline w64 seed 1 at 6000 steps: expN gives 0.777564 while expQ gives
0.765172. They are not averaged. The old phase CLI is retired to prevent accidental reuse.
The old 8/10 and 6/10 significance summaries and variance-floor claim are not reaffirmed.

## Recovery work

DeepONet Adam stages now accept `--resume --checkpoint-every 100` in scripts/run.py.
Checkpoints include model, optimizer, step/stage, histories, causal state, Python/NumPy/Torch/CUDA
random state, and configuration/source/data identity. Atomic replacement protects the previous
checkpoint while writing. The same command resumes compatible work; changed code/config/data
requires a new output directory. Local checkpoints are ignored by Git.

Forced interruption at step 2 and recovery to step 5 on CPU produced bit-identical model tensors
and training/diagnostic histories versus continuous execution. GPU determinism is not yet tested.
L-BFGS recovery is explicitly unsupported and rejected, not silently advertised as resumable.
FNO now saves/restores sampling state and checks data/config identity; its CPU integration test
also passed a forced step-2 interruption followed by bit-identical final model tensors.
Runner logs stdout/stderr, rejects changed manifest identities, and
records keyboard interruption. Abrupt power loss may leave status “running”; checkpoints are
the recovery source, not that status label.

## Next experiments

### Completed frozen-basis pilot

`python scripts/probe_coefficients.py` ran 1000 Adam steps for a new 3-layer, width-128
coefficient MLP on 200 labeled training functions, with the seed-0 saved trunk frozen.
Coordinates are orthonormalized; a per-input constant coefficient is allowed (relaxed bias).
All values below are full-grid test relative Frobenius, not historical mean relative L2.

| Fixed trunk | Original | Supervised coefficient fit | Offline projection oracle |
|---|---:|---:|---:|
| MLP baseline | 1.002217 | 0.876922 | 0.856052 |
| Analytic alignment | 0.790710 | 0.697375 | 0.646002 |
| Random alignment | 0.799133 | 0.692224 | 0.645677 |
| Fourier | 0.711125 | 0.570444 | 0.495834 |

This supports further investigation of fixed-basis coefficient prediction. It uses solution
labels, input normalization, and a relaxed bias; it does not isolate optimizer effects or
establish superiority of a zero-data physics method. One trunk seed and one fit seed only.
Raw records and per-50-step histories are under `results/coefficient_pilot/`.

1. Fixed-trunk optimal coefficients are now available as an offline projection reference.
   Fit a branch to these coefficients on training functions; evaluate held-out functions to
   distinguish coefficient prediction from learned-basis limitations.
2. Compare hard initial conditions versus soft constraints on matched 2D checkpoints/budgets.
3. Public PDEBench Burgers, followed by a second equation, with identical data splits/tasks
   for DeepONet and FNO; supervised FNO is not PINO.

Public source checked: https://huggingface.co/datasets/pdebench/Burgers/tree/main lists
1D_Burgers_Sols_Nu0.001.hdf5 at about 8.23 GB. Disk F has about 466 GB free. File size is not
a blocker. Download has started with pinned revision and publisher SHA-256 verification;
no public-data training result has been claimed or generated in this audit phase.
