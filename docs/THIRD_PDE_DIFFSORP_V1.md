# Third operator class: 1D diffusion-sorption — predeclared protocol

Date: 2026-09-30. This protocol is committed **before** any model is trained on this data.

## 1. Why this file, and what it can and cannot test

PDEBench `1D_diff-sorp_NA_NA.h5` is parabolic and diffusion-dominated, with no shocks and no
translation. It therefore separates two candidate drivers of the Burgers skill horizon:

* nonlinear steepening (Burgers), and
* dissipation of the fluctuation (this file),

against pure transport where nothing changes (advection, already run).

## 2. Documented data caveat: the stored initial state is spatially constant

Checked directly, not assumed. For samples `0000`, `0001`, `0123`, `5000`, `9999`:

| sample | layer 0 spatial std | layer 0 value | layer 1 std | layer 100 std |
|---|---:|---:|---:|---:|
| 0000 | 2.2e-08 | 0.10976 | 0.1159 | 0.2944 |
| 0001 | 1.5e-08 | 0.08340 | 0.1185 | 0.2959 |
| 0123 | 1.5e-08 | 0.13929 | 0.1130 | 0.2928 |
| 5000 | 3.7e-09 | 0.04581 | 0.1223 | 0.2985 |
| 9999 | 1.5e-08 | 0.16468 | 0.1107 | 0.2916 |

The `t = 0` state carries **one degree of freedom per sample** (the constant value). Taking
it as the input would make the task "generate a field from a scalar", so the fluctuation
decomposition would measure generation rather than the transport or dissipation of existing
structure, and it would not be comparable with the Burgers and advection families.

**Decision:** the input is layer **20** (`t = 100`, spatial std 0.2390), and all targets are
strictly later. This is recorded in every protocol file
(`input_index`, `input_time`, `layer0_spatial_std`, `degenerate_initial_state`).

## 3. Settings

Input: layer 20. Targets: layers 25, 40, 55, 70, 85, 100, i.e. `t = 125, 200, 275, 350, 425, 500`.
Same split as every other family: seed 20260922, sizes 1000/200/200, so the trajectory ids
match. Same grid policy: stride 4 to 256 points, no filtering. Same budget: 1500 updates,
5 seeds, 3 architectures (MLP DeepONet, Fourier DeepONet, FNO).

## 4. A-priori statistics and the predeclared prediction

Computed from the **training targets only**, before any model exists:

| t | fluctuation energy | modes for 99% | **predicted sign** mlp / fourier / fno |
|---:|---:|---:|---|
| 125 | 0.7337 | 1 | + / + / + |
| 200 | 0.7178 | 1 | + / + / + |
| 275 | 0.7009 | 1 | + / + / + |
| 350 | 0.6839 | 1 | + / + / + |
| 425 | 0.6672 | 1 | + / + / + |
| 500 | 0.6510 | 1 | + / + / + |

Prediction rule, fixed in advance and **not** refitted here: per-sample fluctuation R^2 is
positive when the fluctuation energy fraction is at least 0.2477 (MLP DeepONet), 0.2728
(Fourier DeepONet) or 0.1502 (FNO). Those thresholds were interpolated from the 14 Burgers
and advection settings and are applied unchanged.

**Falsification criterion:** if any architecture scores at or below zero at any horizon
where the rule predicts positive, the rule is falsified on an unseen operator and will be
reported as falsified. The thresholds will not be refitted on these data.

## 5. Known weakness of this test, stated in advance

`modes for 99%` equals 1 at every horizon: the target fluctuation is essentially a single
spatial mode, so this is a comparatively easy regime. A confirmation here is therefore
**weak** support for the rule. A falsification here would be strong evidence against it.
This asymmetry is recorded before the runs, not after seeing them.

## 6. Commands

```bash
python scripts/prepare_diffsorp_horizons.py
# one family per horizon, e.g.
python scripts/run_public_benchmark.py \
    --data-dir external_data/pdebench/diffsorp/in020_t025_v1 \
    --steps 1500 --out results/diffsorp_in020_t025_v1 --device cuda
python scripts/diagnose_fluctuation.py \
    --data external_data/pdebench/diffsorp/in020_t025_v1 \
    --families diffsorp_in020_t025_v1=endtoend:test,val \
    --out results/fluctuation_diffsorp_v1/t025 --device cuda
```

## 7. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
