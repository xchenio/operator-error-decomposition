# Matched public-data benchmark v1

## Locked protocol

- Dataset: PDEBench Burgers Nu0.001, revision `97bf5bb5315b9a083b4f9fc03e5c84d35286d52c`.
- SHA-256 independently rechecked against publisher: `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d`.
- HDF5 solution shape: `(10000, 201, 1024)`, float32. Cell-center x grid.
- Time coordinates have 202 entries; solution has 201. The upstream generator allocates
  one extra time coordinate. Index 100 is t=1.0; the unused final coordinate t=2.01 is excluded.
- The generator uses diffusion epsilon/pi, while the file Nu attribute is epsilon=0.001.
  This first experiment uses supervised values only; it makes no claim about physical residuals.
- Initial field (index 0) -> field at t=1.0 (index 100). Both have 256 points, stride 4,
  point subsampling without filtering. No temporal rollout or unseen-resolution evaluation.
- Fixed random split from all 10000 trajectories: 1000 train, 200 validation, 200 test;
  split seed 20260922. Exact IDs and data hash recorded in protocol.json.
- This is a project-specific public-data subset, not a reproduction of official benchmark scores.
- Input/target scalar normalization is fitted only on their respective training arrays.

## Comparisons

DeepONet MLP trunk, DeepONet Fourier-feature trunk, and FNO1d. All receive the same inputs,
coordinates, outputs, sample order for the same seed, batch size 32, Adam lr=1e-3,
1500 updates, and five training seeds 0..4. Use the final checkpoint; validation is logged
every 250 updates and does not select the test checkpoint. Architectures have different
parameter counts and step costs; report parameter count and measured GPU update time.
Equal update counts are NOT equal-compute evidence. No significance claim from this pilot.

DeepONet: 64 features, branch/trunk hidden width 128, three hidden layers.
FNO: channel width 32, 16 spectral modes, four blocks; physical cell-center coordinates.
Complex FNO parameters count as two real degrees of freedom.

Projection diagnostics apply to DeepONet only: same full 256-point grid and same split,
complete target matrix, additional constant column, offline per-sample coefficient relaxation.
Record SVD tolerance 1e-12 and sensitivity at 1e-6. No shared-trunk bound is assigned to FNO.

## Execute and recover

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' -u -B scripts/run_public_benchmark.py
```

Rerun the identical command after interruption: completed results are identity-checked and
skipped; unfinished models resume from checkpoints every 50 updates. Random state, optimizer,
history and cumulative update time are restored. Source/data/config/environment mismatch rejects
reuse. Checkpoint and status writes are atomic. The latest uncheckpointed updates may be lost.
Results and events are versioned; intermediate model checkpoints stay local under checkpoints/.

Source references (consulted for metadata semantics, not copied implementation):
- https://huggingface.co/datasets/pdebench/Burgers/tree/main
- https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py

## Next scientific step

Use these same held-out IDs to test whether an early frozen-trunk coefficient probe predicts
the benefit of changing the basis. Physical training and a second PDE remain separate next
experiments; fitting these supervised controls does not establish PINO performance.
