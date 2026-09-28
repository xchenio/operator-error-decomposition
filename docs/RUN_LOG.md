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

- CPU and CUDA stability tests passed all four schedule/batch resume subcases; endpoints,
  exact prior-control metrics and changed-config rejection also verified.
- Full unittest discovery passed 16 tests, including stability/rebound summary boundaries.
- Formal run launched from source commit `672c9d9`. Regression checks overlap part of the
  run, so update timings are operational measurements, not isolated performance benchmarks.

## 2026-09-24 — completed optimization stability controls

- All 40 runs reached 3000 steps. Summary validated identities and exact final metric
  reproduction for all 10 previous constant-LR batch32 controls.
- Cosine decay: 20/20 stable fits over steps 2250/2500/2750/3000. Constant LR: 0/10
  stable with batch32 and 1/10 with batch128. Every constant/full-batch run previously
  crossed the fit threshold and rebounded; final fit alone hides this instability.
- With batch32, decay worsened validation relative L2 and RMSE in 5/5 seeds for each
  trunk despite better training fits. MLP/full-batch decay improves validation relative
  to constant/full-batch; Fourier/full-batch does not. No overall performance claim.
- Saved same-sample-exposure descriptive comparisons and all raw paired differences;
  equal updates do not equal compute. No test data evaluated.
- Generated and inspected stability_curves.png. Re-executed original command: all 40
  identities checked and skipped, no repeated training. Results and next steps documented
  in PUBLIC_STABILITY_V1.md; historical families preserved.

## 2026-09-24 — manuscript synchronized with audited evidence

- User requested updating the paper plus publication-readiness and next-direction advice;
  no new experiments or external submission were authorized or performed this turn.
- Archived four pre-audit drafts under paper/archive/pre_audit_20260924. Rewrote main.md
  around the supervised public-data intervention evidence, and synchronized short Markdown
  and legacy-named LaTeX versions. Original data, checkpoints and figures were preserved.
- Removed active claims of pure optimization attribution, universal variance floor,
  confirmed causal mechanism and low-value-error guaranteeing low PDE residual. Removed
  unverified submission/template assertions; new public evidence is not physics-informed.
- Checked primary sources for Lee/Shin two-step training, Williams et al. basis diagnostics,
  Lu et al. POD comparisons, PDEBench, ReZero, SGDR and large-batch generalization work.
  References and novelty boundaries are documented in the manuscript and
  PUBLICATION_ASSESSMENT.md; this is targeted screening, not an exhaustive novelty review.
- Added evidence tests for numeric tables, run/control counts, citation keys, local links
  and basic TeX balance. No TeX compiler was detected; no PDF or page-count certification.
- `tests/test_paper_evidence.py`: 2 tests passed. Full unittest discovery: 18 tests passed.
  No training source or experiment result was changed by this manuscript revision.

## 2026-09-28 — larger-subset controls and POD ridge protocol

- Locked PUBLIC_SCALE1000_V1.md: 20 n=1000 zero-init/batch32 constant/cosine runs and
  an 18-case training-only residual-POD ridge grid. Validation selection; no test loading.
- Separate runners preserve previous source hashes and identities. Neural runs checkpoint
  every 50 updates; ridge saves each complete candidate and model before continuing.
- Tests added for closed-form ridge normal equations, basis orthogonality, nonmutating
  prediction, neural resume equality, config rejection and old constant-control parity.

- CPU and CUDA new tests passed (two tests per device). Full regression discovery passed
  20 tests. Formal source commit `5cb07f5`; 20 neural runs and 18 ridge cases completed.
- Summary validates all shared identities and ten exact old-control metric reproductions.
  Cosine improves validation relative L2 in 10/10 pairs; RMSE in 9/10. No full train fit.
- MLP cosine validation mean 0.374807 versus initial-mean baseline 0.374509; selected
  rank16/alpha10 ridge is 0.373656 after 18-candidate validation selection, not test evidence.
- Re-executed both runners: all completed items skipped after identity checks. Preserved
  local checkpoints and historical results. No test arrays were loaded.
- Added dated paper follow-up without replacing the core study's original tables; records
  now total 175 neural runs including 20 repeated controls, plus 18 deterministic ridge fits.
  Next not yet executed: nonlinear readout on the identical training-only POD basis.

## 2026-09-28 — same-POD readout protocol

- Locked PUBLIC_POD_READOUT_V1.md: ranks16/32/64, five seeds each, zero-output MLP
  and matched Adam affine control, 3000 cosine updates / batch32. Reuse exact stored
  ridge Q and per-position normalization; reproduce rank-matched ridge metrics first.
- New runner saves atomic full-state checkpoints every50 updates and evaluations every250;
  float64 reconstruction checks squared projection/coefficient error closure. No test loading.
- CPU and CUDA readout tests both passed (2/device): zero prediction, objective identity,
  validation-target isolation, uninterrupted/resumed bitwise weights and history equality,
  completed-run skip and changed-config rejection. Full regression:22 tests passed.
- Command: D:/miniconda3/envs/pytorch/python.exe scripts/run_public_pod_readout.py

- Formal source commit `5ad1676`; all30 runs completed on the RTX5060 Laptop GPU.
  Independent audit reproduces group statistics from per-run files, checks exact shared
  POD checkpoint hashes/oracles, final-step histories, completion manifests and protocol.
  Three selected ridge checkpoints reproduce their original train/validation metrics exactly.
- Audit initially rejected a 1.12e-12 historical-baseline difference: old spatial means
  were float32, new means float64. Adjusted only the audit tolerance to1e-10; no training
  code or results changed. Maximum squared projection closure error is1.32e-16.
- Nonlinear versus matched Adam affine: train L2/RMSE improves15/15, validation L2/RMSE
  worsens15/15. Final MLP beats ridge0/15, initial-mean baseline0/15; full projection fit0/15.
  Validation means at ranks16/32/64:0.398761/0.401114/0.402642. Coefficient mismatch explains
  about93–99% of squared validation error, not a certified pure optimization gap.
- Added paper/FOLLOWUP_POD_READOUT_20260928.md and linked the active manuscript. Historical
  core tables remain unchanged. Combined records:205 neural (including20 old repeats),
  plus18 deterministic ridge fits reused here. No new test arrays were loaded.
- This runner uses atomic progress plus checkpointed history rather than events.jsonl;
  ledger now explicitly documents both persisted logging formats. Local checkpoints remain
  ignored by Git; configuration, metrics, histories and provenance are versioned.
- Final regression:23 tests passed, including numeric report/paper tables, exact shared
  basis identities and squared-error closure. Re-ran the formal runner:all30 completed
  items skipped after identity checks; no training was repeated. `git diff --check` passed.

## 2026-09-28 — frozen ridge residual and fresh-confirmation protocol

- User requested analysis and implementation. Targeted primary-source check found prior
  POD/DeepONet residual learning (Demo et al.2302.12682) and ReZero; no new residual-method
  novelty claim. Proposed future contribution is development-only modal learnability
  selection, not implemented or verified in this experiment.
- Locked docs/RIDGE_RESIDUAL_V1.md and results/public_ridge_residual_v1/plan.json:
  frozen rank16/alpha10 map + zero-output one-layer width16/32 MLP, four correction penalties,
  five seeds, baseline-inclusive early stopping. Eight-alpha expanded ridge control.
- Reserved500 IDs outside every historical split plus conservative smoke prefix0–127.
  Plan stage read only existing protocol/checkpoint metadata, not new trajectory labels.
  Confirmation command requires the selected model hashes/selection lock committed to Git.
- Atomic full training state every50 steps, flushed event stream, selected/latest checkpoints;
  confirmation data cached in50-trajectory atomic chunks for interruption recovery.
- CPU and CUDA tests passed(3 per device): deterministic continuation including patience and
  best model, frozen baseline/zero equality, disjoint IDs, paired bootstrap, commit gate
  preventing label access and completed-confirmation skip. Full regression:26 tests passed.
- Commands: run_public_ridge_residual.py plan; after source commit, train; after selection
  commit, confirm. All use D:/miniconda3/envs/pytorch/python.exe. No confirmation results yet.

- Training at source commit4e2e3e2 completed40 residual runs/60200 updates and8 expanded
  ridge fits. Selected width32/lambda1 by five-seed development mean0.3663995661;
  frozen ridge0.3736562256, expanded ridge alpha100=0.3735998498.
- All40 frozen anchors are array-identical to the original; independent audit replays
  patience/best-step rules and verifies model/result identities.17 runs retained step0.
  Added direct reproduction/training-only test for fixed-basis ridge; CPU4 tests pass.
- Selection lock and all development evidence are being committed BEFORE the confirmation
  stage. No500-ID confirmation pair has been read, and no confirmation score is known yet.

- Selection sealed in Git8881a9c before running `run_public_ridge_residual.py confirm`.
  Full8.23GB source SHA256 matched;500 disjoint pairs extracted in10 resumable chunks.
  Only five selected residual checkpoints and predeclared controls evaluated, with no tuning.
- Fresh relative L2:residual0.4051353931±0.0018933127, frozen ridge0.4053541806,
  expanded ridge0.4045177708, mean baseline0.4043734692. Primary difference=-0.0002187875,
  paired95% CI[-0.0074978679,0.0089227175], relative gain0.0539744%; primary_success=false.
- Secondary RMSE improves5/5:mean0.1477012653 versus frozen ridge0.1509313323 (~2.14%).
  Kept the negative primary conclusion; metric disagreement is not proof of a specific
  low-energy/outlier mechanism. No retraining or retuning followed confirmation.
- Independent audit reproduced bootstrap intervals from versioned per-trajectory errors.
  Updated current conclusions, publication assessment and a dated paper addendum while
  preserving earlier tables and counts. Combined public records:245 neural including20
  historic repeats,26 deterministic ridge fits including six new repetitions of old settings.
- Full regression before confirmation:27 tests passed. New follow-up work must reserve
  another untouched confirmation set; the current500 IDs are now consumed.
- Final regression:28 tests passed, including numeric paper/report tables and paired
  confirmation statistics. Re-executing train/confirm checked identities and skipped both;
  no retraining and no confirmation labels reloaded. `git diff --check` passed.

## 2026-09-28 — objective alignment and modal learnability follow-up

- Development-only energy diagnostic completed; train-only quintile thresholds, no old
  test or500-ID confirmation labels loaded. Lowest-energy development group has47 samples,
  ridge L2=0.698359, selected residual=0.677225; highest group42 samples,0.118216→0.121598.
  This does NOT support claiming low-energy samples caused the previous confirmation failure.
- Training target-energy5th percentile=0.0155348191; proposed normalized-loss denominator
  floor is fitted from training only. Squared relative loss remains a surrogate, not equal
  to the mean nonsquared relative-L2 evaluation metric.
- Preparing strict disjoint representation-fit versus branch-CV sets to prevent POD/ridge
  label leakage into held-out CV folds. Equal-data all-mode and random-mode controls required.
- Second public PDE chosen: PDEBench Advection beta0.4, a linear transport negative control
  (not a second nonlinear task). Publisher HF revision33d90a245d98b8a250884eab613c6f9c26a13b68,
  file8232966952 bytes, SHA256d973ff2bb3c2a5edf42957ff029b78672451532ebd6bf475848ff23cfd3ee3b6.
  Added separate resumable downloader/status path, preserving the original Burgers file.
- Locked OBJECTIVE_MODAL_V1.md and machine-readable plan with disjoint new Burgers training
  split and three reserved500-ID confirmation lists (not evaluated in this development stage).
  Per task:10 full-data objective comparisons,6 strict fixed-step CV probes,25 equal-data
  correction panels;123 neural records planned over three tasks. Extra pure-ridge grids
  distinguish residual improvements from a weak alpha10 anchor, especially on Advection.
- CPU/CUDA new tests passed(2 per device): bounded normalized weights, modal criteria,
  masked continuation equality and held-CV-label perturbations leave final weights unchanged.
  Full regression30 tests passed. Formal runs follow the source/protocol Git commit.
- Source commitf807853; original Burgers41 runs completed,79500 updates. Full-data guarded
  loss improves development L2:0.366400→0.358258; RMSE0.145048→0.143010.
  Strict CV selects modes0–3; stable mask equals top-energy mask, both0.360077 vs all
  normalized0.362092 and random0.375807. Do not claim superiority over energy selection.
- Independent audit checks disjoint samples, exact frozen anchors, fixed-step CV selection,
  masks and group statistics. Added rejection of numerically degenerate modal denominators
  rather than certifying gains caused by the denominator floor; original case passes.
- Targeted primary-source novelty check: Koike et al.2605.27756 uses learned sparse POD
  mode selection for nonlinear decoding. Our procedure differs, but broad mode-selection
  originality is unsupported. Independent split is running; second-PDE download continues.
- Independent Burgers41 runs completed,67500 updates; aggregate82 records complete.
  Full-data normalized L2=0.398468 vs MSE0.407464;10/10 paired L2 improvements across
  the two independent training splits,9/10 RMSE improvements. New strict mask selects0/1,
  equals top-energy again; masked0.403133 vs all0.404991, random0.412182. Only3/5 paired
  gains over all modes on this split; do not turn group means into universal superiority.
- Ten stable/energy pairs across the two tasks have exactly identical histories/metrics.
  New post-training energy strata use development data only; both low-energy bins improve
  with the normalized objective, while the middle bin worsens slightly. This is descriptive.
- Full regression33 tests passed. Advection remains downloading; its experiment is not
  yet reported as completed. All three newly reserved confirmation sets remain unopened.
- Final Advection download completed and matched the pinned SHA256. Ran
  `D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py advection`:
  41 additional records completed,89600 updates. All123 records now complete,236600
  updates total, plus48 deterministic ridge grid refits. Training source remainsf807853;
  Advection launch Git checkpointcd0eaeb; no source edits during training.
- Advection full-data normalized loss worsens L2 from0.169283 to0.175051 and RMSE
  from0.137108 to0.139769 in5/5 pairs. All five energy-bin means worsen. Stable selection
  keeps16/16 modes, so normalized/all, stable, random and energy are exact repeats.
  Across three tasks25 panel records are redundant; record counts are not independent N.
- Expanded Advection ridge selects alpha0.0001 (lower grid boundary): full1000 L2
  0.015378 and anchor400 L2=0.019756, much better than residuals. Improvement relative
  to the original alpha10 anchor0.341140 does not imply nonlinear correction is needed.
- Ran `scripts/summarize_objective_modal.py` and `scripts/diagnose_objective_energy.py`:
  audit checks all123 records, frozen anchors, strict CV, selected checkpoint hashes,
  mask identities and aggregates; stratified aggregation reproduces development scores.
  No reserved confirmation labels loaded. All three500-ID reserved sets stay unopened.
- Synced current conclusions, experiment ledger, publication assessment and dated paper
  addendum. Proposed next step is incremental nonlinear benefit beyond tuned linear
  correction, with rejection/continuous-shrinkage controls, not an implemented success.
- Final `D:/miniconda3/envs/pytorch/python.exe -m unittest discover -s tests`:34 tests
  passed, including all-three-task completion and numeric paper-table checks. Resume
  command `scripts/run_objective_modal.py all` verified identities and skipped all three
  completed tasks without retraining. `git diff --check` passed (CRLF notices only).
- Local datasets and full optimizer/RNG/model checkpoints remain ignored by Git;
  protocol, provenance, selected hashes, progress logs and results are versioned. Local
  sleep/interruption recovery is supported; a Git clone alone is not a checkpoint backup.
