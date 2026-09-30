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

## 7. Result (measured after the protocol was committed)

Per-sample fluctuation R^2 on the test split, 5 seeds, 1500 updates:

| t | fluctuation energy | MLP DeepONet | Fourier DeepONet | FNO |
|---:|---:|---:|---:|---:|
| 125 | 0.7333 | +0.9983 ± 0.0014 | +0.9998 ± 0.0003 | +1.0000 ± 0.0000 |
| 200 | 0.7173 | +0.9990 ± 0.0010 | +1.0000 ± 0.0000 | +1.0000 ± 0.0000 |
| 275 | 0.7004 | +0.9988 ± 0.0012 | +1.0000 ± 0.0000 | +1.0000 ± 0.0000 |
| 350 | 0.6834 | +0.9997 ± 0.0002 | +0.9999 ± 0.0001 | +1.0000 ± 0.0000 |
| 425 | 0.6667 | +0.9998 ± 0.0001 | +0.9999 ± 0.0001 | +1.0000 ± 0.0000 |
| 500 | 0.6506 | +0.9999 ± 0.0001 | +0.9999 ± 0.0001 | +1.0000 ± 0.0000 |

All recomputations reproduce their stored metrics exactly (`max_abs_metric_difference =
0.0`). Conservation holds (corr 0.999996); the oracle input-mean predictor gives 0.74001 at
t = 125, so unlike Burgers the mean-only predictor is far from competitive here.

**The prediction is confirmed: 18 of 18 settings (6 horizons x 3 architectures) are
positive, as predicted.** Applying the thresholds unchanged to these six held-out settings
gives 6/6 for every architecture, so the rule passes its first out-of-sample test.

Two readings, both recorded as stated in advance:

1. **The confirmation is weak.** As predeclared, 99% of the target fluctuation energy lies
   in a single mode, and every model scores essentially 1.000. This regime does not
   discriminate between architectures.
2. **It does discriminate between mechanisms, and that is the useful part.** The fluctuation
   energy here *does* decay with horizon (0.7333 to 0.6506) yet no skill decay appears,
   because the energy never approaches any architecture's threshold. Combined with Burgers,
   where the same decay continues past the thresholds and the skill collapses, the evidence
   supports the threshold form of the rule rather than "longer forecast is harder".

### Remaining gap

No third-PDE setting sits near a threshold, so the rule's *boundary* is still untested
out of sample. A diffusion-dominated problem whose fluctuation energy falls below 0.15
would be the decisive test; none of the downloaded files provides one at the horizons
available. This is stated rather than manufactured by refitting.
