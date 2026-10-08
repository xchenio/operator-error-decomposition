# Remaining gaps: what is still untested and what data would close it

Date: 2026-09-30. Downloading from this machine is not practical (see
`THIRD_PDE_DOWNLOAD_V1.md`: 34 KB/s single connection, 96 KB/s with four parallel ranges).
The files below were therefore identified from the DaRUS catalogue with exact ids, sizes and
MD5s so they can be fetched elsewhere and dropped into `external_data/pdebench/darus/`.

## Gap 1: the threshold boundary on a steady, non-evolution operator

The rule has three out-of-sample confirmations, but only the viscosity ladder probes its
boundary. Darcy at `beta = 1.0` sits far above every threshold (fluctuation energy 0.4964
against 0.2477 / 0.2728 / 0.1502), so it confirmed the rule where it is easy.

**What would close it:** another Darcy `beta`. `beta` controls the roughness of the
coefficient field, so a smoother coefficient should give a smoother solution and a lower
fluctuation energy, potentially crossing a threshold.

| File | id | bytes | MD5 | hypothesis |
|---|---:|---:|---|---|
| `2D_DarcyFlow_beta0.01_Train.hdf5` | 133217 | 1,310,724,760 | `d05c287d4c0b7d3178b0097084238251` | smoothest; most likely to cross a threshold |
| `2D_DarcyFlow_beta0.1_Train.hdf5` | 133218 | 1,310,724,488 | `294f9a03a4aa16b0e386469ca8b471be` | intermediate |
| `2D_DarcyFlow_beta10.0_Train.hdf5` | 133220 | 1,310,724,488 | `a7f23cf8011fc211b180828af39b7d1a` | rougher; expected to stay positive |

Take `beta0.01` if you fetch one, `beta0.01` and `beta10.0` if you fetch two, so the ladder
has both ends.

Ready to run once the file is present:

```bash
python scripts/download_darus.py --file 2D_DarcyFlow_beta0.01_Train.hdf5   # verifies MD5
python scripts/prepare_darcy.py --source external_data/pdebench/darus/2D_DarcyFlow_beta0.01_Train.hdf5 \
    --out external_data/pdebench/darcy_beta001
python scripts/run_darcy_benchmark.py --steps 1500 --device cuda
```

The prediction must be written down **before** those runs, from the a-priori fluctuation
energy of the new target and the existing thresholds, and the thresholds must not be
refitted. If the setting still lies far from the thresholds, say so rather than treating it
as a boundary test.

## Gap 2: a system of equations

All four operators are scalar. A system tests whether the decomposition carries over when
there are several coupled fields, and it is the one structural gap left in the study.

This deposit has no shallow-water data. The smallest system available is 1D compressible CFD
(density, velocity, pressure):

| File | id | bytes | MD5 |
|---|---:|---:|---|
| `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5` | 164672 | 12,410,888,600 | `2fdd00138f1d7abc794fb953021a9f43` |

The other CFD files are 19.9 GB and 24.8 GB; there is no smaller system in this deposit.

Open questions that must be settled before any run, because the answers change the design:

* how many components the stored tensor has and whether they share a time axis;
* whether a per-component mean/fluctuation split is the right generalisation, or whether the
  conserved quantities (mass, momentum, energy) should define the reference instead — for a
  system the conserved-mean argument is richer than for a scalar field, and this is arguably
  the more interesting version of the decomposition;
* whether the existing 1D runner can be extended or needs a multi-field variant.

No preparation script is written yet, deliberately: writing one before seeing the layout
risks the same class of silent mismatch that the diffusion-sorption check caught.

## Gap 3: carried over from earlier

* FLOP **accounting** is still incomplete: FFT work is uncounted, so equal measured FLOPs
  are not provably equal real work. The direction of the error favours FNO, so the sign of
  the conclusion is safe, but a true accounting would need a custom counter.
* The physics-informed arm uses a different batch and collocation size from the supervised
  runs, so its work per update is not matched.
* The physics-informed arm exists on one operator at one viscosity, because only Nu = 0.01
  admits a trustworthy residual here.

## What is not a gap

Update-count budget, wall-clock budget, measured FLOP budget, parameter count, seven
forecast horizons per evolution problem, two viscosities, and the trunk-span bound are all
covered. Cross-operator transfer is covered by three held-out families. The remaining items
are the boundary on a steady operator, systems, and exact FLOP accounting.
