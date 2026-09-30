# Viscosity ladder: Burgers Nu = 0.01 — predeclared protocol

Date: 2026-09-30. Committed **before** any model is trained on this data.

## 1. Why this is the decisive test

The diffusion-sorption result (`THIRD_PDE_DIFFSORP_V1.md`) confirmed the threshold rule, but
weakly: every model saturated near 1.000 and no setting lay near a threshold, so the rule's
*boundary* was never tested out of sample. This file fixes that.

Burgers at `Nu = 0.01` has ten times the viscosity of the primary family. Dissipation is
stronger, so the fluctuation energy decays faster with horizon and the curve crosses the
thresholds at **shorter** forecast times. The rule therefore makes 21 concrete sign
predictions, several of them at crossings.

## 2. Data

`1D_Burgers_Sols_Nu0.01.hdf5`, MD5 `e6d9a4f62baf9a29121a816b919e2770`, size 8,232,968,312 B,
verified. Layout is identical to the Nu0.001 file (`tensor (10000,201,1024)`, `t-coordinate`
202 with one unused trailing entry, `x-coordinate` 1024, `Nu = 0.01`), so
`scripts/prepare_pdebench_1d_horizons.py` is used; `src.public_benchmark.prepare()` is not,
because it pins the Nu0.001 digest and editing it would change `source_hash()` in every run
identity.

Same split as every other family: seed 20260922, sizes 1000/200/200, so trajectory ids match.
Same grid policy: stride 4 to 256 points, no filtering. Same budget: 1500 updates, 5 seeds,
3 architectures. Seven horizons: `t = 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00`.

## 3. A-priori statistics and the predeclared predictions

Fluctuation energy fraction, from training targets only, with the Nu0.001 values for
comparison at the same horizons:

| t | fluctE (Nu=0.01) | fluctE (Nu=0.001) | modes for 99% | **predicted sign** mlp / fourier / fno |
|---:|---:|---:|---:|---|
| 0.10 | 0.5714 | 0.5991 | 31 | + / + / + |
| 0.25 | 0.4558 | 0.4831 | 25 | + / + / + |
| 0.50 | 0.3424 | 0.3675 | 22 | + / + / + |
| 0.75 | 0.2695 | 0.2935 | 19 | **+ / - / +** |
| 1.00 | 0.2211 | 0.2437 | 17 | **- / - / +** |
| 1.50 | 0.1609 | 0.1823 | 14 | **- / - / +** |
| 2.00 | 0.1254 | 0.1456 | 12 | **- / - / -** |

Thresholds, fixed in advance and taken unchanged from the fit on Burgers Nu0.001 plus
advection: 0.2477 (MLP DeepONet), 0.2728 (Fourier DeepONet), 0.1502 (FNO). They are not
refitted on this data.

**Predicted skill horizons from the rule**, by interpolating where the fluctuation energy
crosses each threshold:

| Model | predicted here (Nu=0.01) | measured earlier (Nu=0.001) | predicted shift |
|---|---:|---:|---|
| Fourier DeepONet | ~0.74 | ~0.86 | shorter |
| MLP DeepONet | ~0.86 | ~0.98 | shorter |
| FNO | ~1.65 | ~1.94 | shorter |

## 4. Falsification criteria, fixed in advance

1. **Sign test.** Each of the 21 settings (7 horizons x 3 architectures) is predicted
   positive or negative as tabulated. A wrong sign falsifies the rule for that architecture.
   Reported as raw counts per architecture, not as an aggregate score.
2. **Ordering test.** The rule predicts all three skill horizons are shorter than at
   Nu=0.001. If any measured skill horizon here is longer, the ordering prediction is
   falsified.
3. **No refitting.** If the rule fails, the thresholds are reported as falsified on this
   data and are not adjusted. A new fit would be a new protocol with its own held-out test.

## 5. Known limitation, stated in advance

The two Burgers families share the same operator, initial-condition generator and solver, so
this is a parameter ladder rather than a new operator class. It tests whether the rule
tracks a physical parameter, not whether it generalises across operators; the latter rests on
the diffusion-sorption result.

## 6. Commands

```bash
python scripts/prepare_pdebench_1d_horizons.py \
    --source external_data/pdebench/darus/1D_Burgers_Sols_Nu0.01.hdf5 \
    --out external_data/pdebench/burgers_nu001 \
    --equation "1D Burgers, Nu=0.01 (10x the viscosity of the primary family)"
# per horizon, e.g. t = 0.75
python scripts/run_public_benchmark.py \
    --data-dir external_data/pdebench/burgers_nu001/fixed_t075_v1 \
    --steps 1500 --out results/burgers_nu001_t075_v1 --device cuda
python scripts/diagnose_fluctuation.py \
    --data external_data/pdebench/burgers_nu001/fixed_t075_v1 \
    --families burgers_nu001_t075_v1=endtoend:test,val \
    --out results/fluctuation_burgers_nu001_v1/t075 --device cuda
```

## 7. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
