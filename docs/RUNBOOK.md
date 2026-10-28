# Resumable experiment runbook

The machine may sleep at any time. Long jobs must therefore be *configuration-resumable*, not
dependent on a live terminal.

## Rules

1. Use the experiment runner introduced in `scripts/experiment_runner.py`, not an ad-hoc shell
   loop.
2. The runner writes `manifest.json` before the first command and emits one JSON event per state
   transition. It skips a job only after its declared completion artifact exists.
3. Use a distinct output directory per experiment family; never reuse an old result directory
   for a different configuration.
4. Commit the manifest, event log, summary, and code changes after each completed experiment
   family. Do not commit transient checkpoints unless they are needed for a published analysis.
5. After a sleep/interruption, rerun the exact same command with `--resume`; the runner restarts
   only jobs lacking their completion artifact.

## Environment

```powershell
conda activate pytorch
python --version
```

The validated environment is documented in `environment.yml` (Python 3.11.16,
CUDA-enabled PyTorch 2.14.0+cu130). Commands in this repository invoke plain `python` so they
work on any machine once the environment is active; the `.cmd` sweep drivers take the
interpreter from `set PY=python`, which can be overridden by defining `PY` beforehand.
Historical log entries name an absolute interpreter path and are left unchanged.

## Required evidence per run family

- command line and immutable configuration;
- Git commit and source/environment fingerprints;
- dataset provenance and checksum;
- job-level start/finish/exit events;
- raw result files and an aggregated summary;
- an explicit status when a run is interrupted or fails.
