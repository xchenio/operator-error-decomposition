# PDEBench + FNO validation protocol

This is the public-benchmark, cross-architecture extension of the project. PDEBench is an
officially released PDE benchmark with 1D Burgers data and baseline FNO code. We use its HDF5
file as a *supervised* initial-to-final-state FNO control; it is not substituted into the
PI-DeepONet physics residual.

## Obtain the data once

Create `external_data/pdebench/` and use PDEBench's official direct-download tooling or its
official data repository to obtain the desired 1D Burgers HDF5 file. The full official training
files can be large (several GB), so they are deliberately ignored by Git and never downloaded by
project scripts. Record the source URL, file SHA-256, and chosen PDE parameters in the run
manifest before training.

Recommended first target: `1D_Burgers_Sols_Nu0.001.hdf5`.

## Validate file schema before training

```powershell
& 'D:\miniconda3\Scripts\conda.exe' run -n pytorch python scripts/train_fno_pdebench.py `
  --data external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5 --inspect-only
```

If the file contains multiple trajectory arrays, rerun the training command with `--field` set to
the selected HDF5 dataset key. The smoke configuration intentionally limits itself to 128 samples
and 200 steps. Its only purpose is validating data semantics, FNO execution, checkpoint resume,
and record keeping.

## Sleep-safe execution

```powershell
& 'D:\miniconda3\Scripts\conda.exe' run -n pytorch python scripts/experiment_runner.py `
  --plan experiments/pdebench_burgers_fno_smoke.json --resume
```

The runner creates `results/runs/pdebench_burgers_fno_smoke/manifest.json` and `events.jsonl`.
The FNO script saves a `.pt` checkpoint every 25 steps; the committed plan already includes
`--resume`, so rerunning the same command resumes an incomplete training job automatically. Do
not treat the 200-step smoke result as a paper number.

## Evidence standard for a paper result

1. Lock the exact data file checksum and HDF5 field name.
2. Run five seeds and publish all raw JSON values.
3. Report a budget curve, not only a terminal value.
4. Keep the architecture comparison honest: FNO is a supervised public-data control, whereas
   the PI-DeepONet result is physics-informed. Do not claim direct method superiority from the
   first comparison.
