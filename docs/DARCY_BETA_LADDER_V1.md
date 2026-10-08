# Darcy beta ladder: searching for the threshold boundary on a steady operator

Date: 2026-10-08. Committed **before** any run on these files.

## 1. What was being looked for

`REMAINING_GAPS_V1.md` identified the untested item: the threshold rule has three
out-of-sample confirmations, but only the viscosity ladder probes its boundary, and Darcy at
`beta = 1.0` sits far above every threshold. The hypothesis was that `beta`, which controls
the roughness of the coefficient field, would move the target fluctuation energy down far
enough to cross a threshold on a **steady, non-evolution** operator.

## 2. The hypothesis fails, found from a-priori statistics before training

Computed from training targets only:

| beta | fluctuation energy | modes for 99% | target std |
|---:|---:|---:|---:|
| 0.01 | **0.5638** | 5 | 0.0307 |
| 1.0 (earlier) | 0.4964 | 11 | 0.1371 |
| 10.0 | **0.4918** | 12 | 1.1850 |

The fluctuation energy barely moves across a 1000-fold range of `beta`, and it is *higher* at
the smoothest setting, not lower. Thresholds are 0.2477 (MLP DeepONet), 0.2728 (Fourier) and
0.1502 (FNO).

**Conclusion stated before the runs: this family does not contain a boundary setting.** Every
Darcy variant available lies far above every threshold, so:

* the prediction is **positive skill for all three architectures at both betas**;
* the runs will **not** test the rule's boundary, and must not be presented as if they do;
* the boundary on a steady operator remains **untestable with the data in this deposit**.

This is a negative result about the data, recorded rather than worked around by refitting.

## 3. Why run them at all

Because a 1000-fold change in coefficient roughness that leaves the prediction unchanged is
itself informative, and because it tests the rule across the whole Darcy family rather than at
one point. The expectation is low: this is a weak confirmation, like diffusion-sorption.

## 4. Protocol

Identical to `DARCY_ELLIPTIC_V1.md`: stride 2 to 64x64, split seed 20260922 with sizes
1000/200/200, Adam 1e-3, batch 32, **1500 updates**, seeds 0-4, three architectures. Thresholds
unchanged and not refitted.

## 5. Falsification

Any architecture at or below zero falsifies the rule for that architecture. Note that because
both settings are far above the thresholds, a *confirmation* here adds little; a falsification
would be decisive.

## 6. Commands

```bash
python scripts/prepare_darcy.py --source external_data/pdebench/darus/2D_DarcyFlow_beta0.01_Train.hdf5 \
    --out external_data/pdebench/darcy_beta0.01
python scripts/prepare_darcy.py --source external_data/pdebench/darus/2D_DarcyFlow_beta10.0_Train.hdf5 \
    --out external_data/pdebench/darcy_beta10.0
python scripts/run_darcy_benchmark.py --data-dir external_data/pdebench/darcy_beta0.01 \
    --steps 1500 --out results/darcy_beta001_v1 --device cuda
python scripts/run_darcy_benchmark.py --data-dir external_data/pdebench/darcy_beta10.0 \
    --steps 1500 --out results/darcy_beta10_v1 --device cuda
```

## 7. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
