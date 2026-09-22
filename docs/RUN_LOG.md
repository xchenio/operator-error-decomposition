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
