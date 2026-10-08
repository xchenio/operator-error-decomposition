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

## 7. Result (measured after the protocol was committed)

Per-sample fluctuation R^2 on the test split, 5 seeds, 1500 updates:

| t | fluctuation energy | MLP DeepONet | Fourier DeepONet | FNO | predicted |
|---:|---:|---:|---:|---:|---|
| 0.10 | 0.5661 | +0.2346 ± 0.0521 | +0.6891 ± 0.0215 | +0.9975 ± 0.0004 | + / + / + |
| 0.25 | 0.4507 | +0.1737 ± 0.0126 | +0.4392 ± 0.0336 | +0.9800 ± 0.0038 | + / + / + |
| 0.50 | 0.3320 | +0.1500 ± 0.0152 | +0.1878 ± 0.0747 | +0.9519 ± 0.0060 | + / + / + |
| 0.75 | 0.2596 | +0.0628 ± 0.0215 | -0.0122 ± 0.0795 | +0.8751 ± 0.0126 | + / - / + |
| 1.00 | 0.2131 | -0.0273 ± 0.0159 | -0.2443 ± 0.1387 | +0.8072 ± 0.0335 | - / - / + |
| 1.50 | 0.1559 | -0.2387 ± 0.0258 | -0.8703 ± 0.4734 | +0.4670 ± 0.0419 | - / - / + |
| 2.00 | 0.1217 | -0.2972 ± 0.1299 | -1.1128 ± 0.3465 | -0.2219 ± 0.4127 | - / - / - |

All recomputations reproduce their stored metrics exactly.

### Test 1: signs — passed

**21 of 21 correct** (7 horizons x 3 architectures); 7/7 for each architecture separately.
Every crossing in the table above was predicted in advance and landed where predicted.

### Test 2: skill horizons — passed, with quantified error

| Model | predicted | measured here | measured at Nu=0.001 | absolute error |
|---|---:|---:|---:|---:|
| Fourier DeepONet | 0.740 | **0.735** | 0.860 | 0.005 |
| MLP DeepONet | 0.860 | **0.924** | 0.980 | 0.064 |
| FNO | 1.650 | **1.839** | 1.940 | 0.189 |

All three are **shorter** than at Nu=0.001, as predicted, so the ordering prediction passes.
The absolute error grows with the horizon length (0.005, 0.064, 0.189), which is what one
expects from linear interpolation over a coarse 0.5-wide grid at the far end of the ladder.

### What this adds

The diffusion-sorption test confirmed the rule where it is easy; this test confirms it **at
its boundary**, on data never used to fit the thresholds, with the thresholds unchanged. The
rule now has two out-of-sample confirmations, one of them quantitative to within 0.005-0.19
in forecast time.

### Remaining limits, restated

This is a parameter ladder on the same operator, solver and initial-condition generator, so
it shows the rule tracks a physical parameter; it does not by itself show the rule
generalises to a new operator class. The thresholds remain fitted on Burgers Nu0.001 plus
advection only, and were not refitted here or on diffusion-sorption. A steady elliptic
operator is still untested.
