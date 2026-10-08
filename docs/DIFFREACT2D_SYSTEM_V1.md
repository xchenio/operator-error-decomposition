# A second system, and two-dimensional: 2D reaction-diffusion — predeclared protocol

Date: 2026-10-08. Committed **before** any run on this data.

## 1. Why this file

The threshold rule failed on the 1D compressible CFD system (`CFD_SYSTEM_V1.md`): 3 of 6
signs, every positive prediction wrong. That is a single system, so the failure could be
specific to compressible flow. This is a **second system**, and also two-dimensional, so it
separates two candidate explanations at once:

* if it fails too, "scalar operators only" is supported by two structurally different systems;
* if it succeeds, the boundary has to be restated rather than attributed to systems as such.

## 2. Data, and the two properties measured before writing any code

`2D_diff-react_NA_NA.h5`, MD5 `b8d0b86064193195ddc30c33be5dc949`, 12,634? — exactly
13,243,560,688 B, verified. One HDF5 group per sample, `"0000"`..`"0999"`, each with
`data (101, 128, 128, 2)` and `grid/t (101,)`, `grid/x (128,)`, `grid/y (128,)`.

**There are only 1000 samples**, not 10000, so the split is 600/200/200 rather than the
1000/200/200 used elsewhere. This is recorded because it changes the comparison, not hidden.

Two properties were measured first, because the 1D diffusion-sorption file had already shown
that a stored initial layer can be unusable:

1. **The stored initial layer is broadband noise.** Only 4.9% of its spectral energy lies
   inside r < 0.25 Nyquist and 19% inside r < 0.5. Subsampling such a field discards the
   fine-scale detail that determines the later pattern, so layer 0 is **not** used.
2. **The field smooths very fast.** Energy inside r < 0.25 Nyquist rises to 70.6% by layer 5
   and 96.8% by layer 20. The input is therefore **layer 20** (t = 1.0), where subsampling is
   safe, and the target is strictly later. This follows the same convention as the 1D
   diffusion-sorption family.

## 3. Setting

Input: both components at layer 20, stride 4 to 32x32 = 1024 points per component, so
`branch_in = 2048` and the trunk takes `(x, y)`. Each cache predicts **one** component at
layer 100 (t = 5.0), which keeps every downstream script unchanged. Same budget as every
other family: Adam 1e-3, batch 32, **1500 updates**, seeds 0-4.

**No FNO control.** `FNO2d` lifts a single value channel plus two coordinates, and the
two-component input would need a different lift. `src/fno2d.py` is hashed into the Darcy run
identities, so a variant would have to be a new module; that is deferred and stated, not
implied.

## 4. A-priori statistics and the predeclared predictions

From training targets only. Thresholds unchanged: 0.2477 (MLP DeepONet), 0.2728 (Fourier).

| component | fluctuation energy | modes for 99% | CV of \|\|y\|\| | spread | predicted mlp / fourier |
|---|---:|---:|---:|---:|---|
| 0 | 0.9098 | 218 | 0.04 | 1.3 | **+ / +** |
| 1 | 0.8265 | 165 | 0.07 | 1.6 | **+ / +** |

Both are far above every threshold, so this is a **non-boundary** test: it can falsify the
rule but a confirmation would be weak, exactly like Darcy beta 10.0. Both settings satisfy
the scope condition (CV at most 0.07, spread at most 1.6).

## 5. Falsification, fixed in advance

Four predictions (2 components x 2 trunks). Any architecture at or below zero falsifies the
rule for that trunk on this operator class. Thresholds are applied unchanged and will not be
refitted. The outcome is reported whatever it is; the point of the experiment is to decide
between the two explanations in section 1, not to obtain a favourable number.

## 6. Commands

```bash
python scripts/prepare_diffreact2d.py --time-indices 100
for c in 0 1; do
  python scripts/run_darcy_benchmark.py \
      --data-dir external_data/pdebench/diffreact2d/c${c}_in020_t100_v1 \
      --architectures deeponet_mlp,deeponet_fourier \
      --steps 1500 --out results/diffreact2d_c${c}_v1 --device cuda
  python scripts/diagnose_fluctuation.py \
      --data external_data/pdebench/diffreact2d/c${c}_in020_t100_v1 \
      --families diffreact2d_c${c}_v1=endtoend:test,val \
      --out results/fluctuation_diffreact2d_v1/c${c} --device cuda
done
```

## 7. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
