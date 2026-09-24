# Run log

## 2026-09-22 — second audit and actual recovery implementation

- Completed `python scripts/probe_coefficients.py`: four fixed-trunk supervised pilot runs,
  1000 steps each. Results and limitations recorded in AUDIT_V2.md.
- Ran `python tests/test_fno_resume.py`: CPU FNO final tensors were bit-identical after
  simulated interruption and restart versus continuous training.
- Started pinned PDEBench download using `python -u -B scripts/download_pdebench.py`.
  Expected 8,232,968,312 bytes; revision and publisher SHA-256 embedded in the script.
  Live state: external_data/pdebench/download_status.json (local, ignored by Git).
  Resume by rerunning that command after interruption. Completion requires checksum validation.

- Ran `python tests/test_projection_audit.py`: three tests passed (known omitted-mode
  counterexample, bias/Pythagoras, rank deficiency).
- Ran `python scripts/audit_checkpoints.py`: 35 checkpoints completed, each independently
  persisted with data/source provenance. All three data splits evaluated on full grids.
- Ran `python scripts/audit_statistics.py`: 13 configuration groups, 11 exploratory
  comparisons, one conflicting repeat quarantined. Unknown source hashes remain unknown.
- Ran `python tests/test_resume.py`: forced CPU Adam interruption/resume matched uninterrupted
  parameters and histories bit for bit. See AUDIT_V2.md for limits and measured results.
- Corrected legacy claims: original LB proxies are not certified bounds; supervised value
  fit does not guarantee small derivative residual. The old phase aggregation CLI is retired.

This append-only log records work performed after Git adoption. Raw manifests and event streams
for future long runs live under `results/runs/<family>/`.

## 2026-09-22 — repository stabilization and validation extension

- Created Git repository and committed the pre-existing project baseline as `b31d444`.
- Verified the requested conda interpreter:
  `D:\\miniconda3\\envs\\pytorch\\python.exe`, Python 3.11.16, PyTorch 2.14.0+cu130,
  CUDA available, NumPy 2.4.6, SciPy 1.17.1.
- Added `h5py 3.16.0` to support official PDEBench HDF5 files; exact environment packages are
  frozen in `requirements-pytorch.lock`.
- Ran `tests/test_sampler_contracts.py`: passed for Burgers, diffusion--reaction, and Darcy.
  The test compares batched paired evaluation to an independent one-sample-at-a-time oracle and
  rejects the historical `(B, B, N)` broadcast shape.
- Ran a synthetic-HDF5 end-to-end FNO smoke test: dataset discovery, two optimizer steps,
  checkpoint serialization, and JSON result serialization all completed. Synthetic artifacts are
  deliberately local/ignored and are not scientific evidence.
- Ran the existing `scripts/check_residual.py` in the CUDA environment. The process completed;
  rerun it before a paper release through a logged experiment family if its numeric residual
  output is needed as a fresh evidence artifact.

## Next queued run (not started)

The public PDEBench 1D Burgers file has not been downloaded automatically because the official
training file is multi-GB. Once placed at the path in `docs/PDEBENCH_PROTOCOL.md`, execute the
committed `experiments/pdebench_burgers_fno_smoke.json` plan with `--resume`. The plan writes a
manifest before launching and the FNO job checkpoints every 25 steps.

## 2026-09-22 — downloaded public data and matched protocol

- Supersedes the queued-run note above: the public file is present. Independently verified
  SHA-256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d`.
- Inspected the actual HDF5 and official generator: 10000 x 201 x 1024 solutions,
  with one unused final time-coordinate entry; the generator uses epsilon/pi diffusion.
- Executed `scripts/prepare_public_benchmark.py`: immutable fixed-t=1 subset and train-only
  normalization persisted in `external_data/pdebench/fixed_t1_v1`.
- `tests/test_public_benchmark.py` passed (2 tests). `tests/test_matched_public_resume.py`
  passed for both DeepONet and FNO: forced CPU interruption restores bitwise-identical
  final model tensors and metrics, and changed configurations are rejected.
- Added matched supervised three-architecture/five-seed protocol, actual cell-centered FNO
  coordinates, source/data/environment identities, atomic checkpoints every 50 steps,
  progress JSON and append-only events. See PUBLIC_BENCHMARK_V1.md for the locked protocol.
- Formal CUDA command queued after implementation commit:
  `D:/miniconda3/envs/pytorch/python.exe -u -B scripts/run_public_benchmark.py`.
  No public-data model performance claims are made before that run completes.

## 2026-09-22 — completed public CUDA pilot

- Executed the formal command above at source commit `4a11b63`; 15 / 15 runs completed.
  Every model trained 1500 updates; each has identity, atomic status, event log and local
  resumable checkpoint. Historical results were not overwritten.
- Added and executed `scripts/summarize_public_benchmark.py`; validates common source,
  environment, data and matched configurations before aggregating all five seeds.
- Test mean relative L2 (mean +/- sample standard deviation): MLP 0.39357 +/- 0.01014,
  Fourier 0.41231 +/- 0.01344, FNO 0.22835 +/- 0.00680. Initial spatial-mean baseline
  is 0.37577. Report negative DeepONet baseline comparisons explicitly.
- Fourier offline projection error 0.04770 (0.05937 at rcond 1e-6) contrasts with its
  actual 0.41231; next hypothesis concerns coefficient learning / conditioning, not an
  established pure-optimization cause. Full limitations and next controls are in
  PUBLIC_BENCHMARK_V1.md. Updated CURRENT_CONCLUSIONS.md to prevent stale extrapolation.
- `python -B -m unittest discover -s tests -p "test*resume.py"`: 3 tests passed.
- `PUBLIC_TEST_DEVICE=cuda python -B tests/test_matched_public_resume.py`: both model
  subcases passed bitwise parameter and exact metric equality after forced interruption.
- Re-executed the formal command: all 15 completed identities checked and skipped.
  No experiment was retrained. Source files used by training were unchanged during the run.
- This is a supervised, fixed-time, single-split pilot, not a physics-informed or official
  full-dataset benchmark claim. Reserve unused trajectory IDs for final confirmatory tests.

## 2026-09-22 — fixed-trunk factorial protocol locked

- Added PUBLIC_COEFFICIENTS_V1.md and a separate resumable runner; 2 trunk families x
  5 seeds x conditioning/mean-lift factorial = 40 runs, predeclared 1500/3000 budgets.
- Both conditioning arms retain exactly the same SVD subspace at rcond 1e-6, 65 branch
  outputs, unit element RMS, common branch initialization and minibatch sequence.
- New runner checkpoints parent/source/data/environment identity, model/optimizer/RNG,
  basis, history, cumulative timing and milestone results every 50 steps.
- Tests added for equal span/scale and forced interruption plus identity rejection.
  Actual test/run outcomes will be appended after execution.

- Fixed-trunk CPU and CUDA tests both passed (2 tests per device, including two resume
  subcases). Full unittest discovery passed 10 tests. No claim of physical power-loss testing.
- Formal 40-run CUDA command started from commit `f3d7685`; checkpoints and results are
  written incrementally. Training source is frozen while running.

## 2026-09-23 — fixed-trunk factorial completed and audited

- On continuation, inspected all progress files and active Python processes: all 40 runs
  had completed 3000 steps; no training process remained. No restart was necessary.
- Executed `python -B scripts/summarize_public_coefficients.py`: all common identities,
  matched parent hashes and configurations validated; recorded eight groups, both budget
  milestones and paired effects. Generated and visually inspected validation_curves.png.
- Executed `python -u -B scripts/run_public_coefficients.py` again: all 40 were skipped
  after identity checks, with no repeated updates.
- Whitening alone reduced mean validation error by 0.00626 (MLP) and 0.01831 (Fourier),
  with 4/5 seeds improved in each. Mean-lift effects were mixed. All eight group test
  means remain above the 0.37577 no-training baseline; no practical improvement claim.
- Every group mean validation error improved between 1500 and 3000 updates. This supports
  budget sensitivity, not eventual attainment of the offline projection oracle.
- Preserved all negative results and historical families. Updated current conclusions and
  ledger; next pre-experiment checks should target small-sample branch fitting and residual
  initialization before adding architecture or physics complexity.

## 2026-09-23 — small-subset / initialization protocol

- Locked PUBLIC_SMALLFIT_V1.md: two trunks, five seeds, nested n=32/128/1000 subsets,
  random versus zero residual initialization; 60 runs, 3000 updates, validation only.
- Reuse whitened bases and input-mean offsets; compare actual subset fit with its own
  affine projection oracle. Predeclare excess relative Frobenius <= 0.01 fit criterion.
- Explicitly not a few-shot benchmark: pretrained trunks and normalization already saw
  original 1000 training samples. No new final holdout is consumed.
- Added dedicated runner and tests for subset selection, exact zero-init baseline,
  forced resume equality and configuration rejection. Verification outcomes follow.

## 2026-09-24 — resume the interrupted smallfit study

- Inspected process command lines and all saved progress: 37 completed runs, one stopped
  at step 1350, no active smallfit training process. Other Python services were untouched.
- Restarted the identical `python -u -B scripts/run_public_smallfit.py` command, preserving
  source commit `05f5879`, identities, subset rows and hyperparameters; completed runs skip.
- Before the original launch, CPU and CUDA smallfit tests passed, including exact initial
  mean-baseline predictions, forced resume equality, subset identity and config rejection.
- Full regression suite and final evidence aggregation are being verified on continuation.

## 2026-09-24 — smallfit results finalized

- Actual events confirm resume at step 1350 for deeponet_fourier_s1_n32_z1. All 60 runs
  reached 3000 steps without modifying training sources or hyperparameters.
- `python -B -m unittest discover -s tests`: 11 tests passed.
- `python -B scripts/summarize_public_smallfit.py`: validated all identities, parent hashes,
  nested row indices and configurations. Saved paired summaries and inspected fit_curves.png.
- At n=32 zero initialization reaches excess relative Frobenius <=0.01 in 10/10 runs.
  At n=128 only 1/10, n=1000 0/10. Fourier random n=32 previously reached but did not retain
  the threshold in every seed; do not confuse final instability with inability to fit.
- Zero initialization lowers final validation mean-relative-L2 in 29/30 matched pairs;
  all six group averages and seed standard deviations are lower. No significance claim.
- Every group mean remains worse than validation initial-mean baseline 0.374509; the
  best group mean is MLP n=1000 zero init 0.389815. No test data evaluated this round.
- Updated conclusions and ledger. Next proposed, not executed: fixed small-subset learning-
  rate/batch stability controls, with MSE and relative-error metrics reported together.

## 2026-09-24 — stability factorial protocol locked

- User requested preserving zero residual initialization and testing decay / batch size.
- Locked 40-run n=128 protocol: 2 trunks x 5 seeds x constant/cosine LR x batch 32/128.
  Predeclared final and last-four-logged-point stability thresholds; no test evaluation.
- Explicitly report unequal sample exposure, plus descriptive full-batch 750 versus
  mini-batch 3000 comparison. Learning-rate position is checkpoint-step derived.
- Added independent runner and PUBLIC_STABILITY_V1.md; verification/run outcomes follow.
