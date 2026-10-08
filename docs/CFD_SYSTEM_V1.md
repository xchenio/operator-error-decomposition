# A system of equations: 1D compressible CFD — predeclared protocol

Date: 2026-10-08. Committed **before** any run on this data.

## 1. Why

All four operators so far are scalar. This is the first **system**, and the last structural
gap in the study. It tests whether the mean/fluctuation decomposition and the threshold rule
carry over when several coupled fields are present.

## 2. Data and layout

`1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5`, MD5 `2fdd00138f1d7abc794fb953021a9f43`,
12,410,888,600 B, verified. Three fields — `density`, `Vx`, `pressure` — each
`(10000, 101, 1024)`, plus `t-coordinate (102,)` and `x-coordinate (1024,)`, attributes
`eta = zeta = 0.01`. Target layer 100, `t = 1.0`. Stride 4 to 256 points. Split seed 20260922,
sizes 1000/200/200, so trajectory ids match every other family.

## 3. Design decision, recorded before the fact

The branch receives all three components at the initial time (768 values) and predicts
**one** component at the target time; one cache is written per output component. This keeps
every downstream script unchanged — each cache is an ordinary single-output task with
`branch_in = 768`, `trunk_in = 1` — so the existing runner and diagnostic apply without
modification, and no coordinate handling is invented.

**Cost: FNO is not run here.** `FNO1d` lifts a value channel plus a coordinate channel and
requires the input and the coordinate to have equal length, which a three-channel input
violates. `src/fno.py` is hashed into every existing run identity and was deliberately not
edited, so the FNO control is **deferred** for this family and only the two DeepONet trunks
are reported. This is a stated omission, not an oversight.

## 4. Conservation structure, measured from the data

The interest of a system is that its components have different conservation properties. Over
the training split:

| quantity | corr(input mean, target mean) | mean abs difference | relative to mean |
|---|---:|---:|---:|
| mass, `mean(rho)` | **1.000000** | 4.87e-05 | 8.3e-06 |
| momentum, `mean(rho*Vx)` | **1.000000** | 5.60e-04 | 3.7e-04 |
| component mean of `Vx` | 0.9874 | 3.15e-02 | **1.2e-01** |
| component mean of `pressure` | 0.9999 | 3.35e-01 | 5.7e-03 |

Mass and momentum are conserved, as expected on a periodic domain. The **spatial mean of `Vx`
alone is not** — momentum is conserved, velocity is not — and its mean differs by 12%.

So within a single system the input-spatial-mean predictor is a **conservation-law predictor
for density**, approximately one for pressure, and **not one for Vx**. This is the Darcy
lesson reproduced inside one operator, and it is measured rather than asserted.

## 5. A-priori statistics and the predeclared predictions

From training targets only. Thresholds unchanged from the fit on Burgers Nu0.001 plus
advection: 0.2477 (MLP DeepONet), 0.2728 (Fourier DeepONet).

| component | fluctuation energy | modes for 99% | CV of \|\|y\|\| | spread | predicted mlp / fourier |
|---|---:|---:|---:|---:|---|
| density | **0.2623** | 27 | 0.48 | 11.5 | **+ / -** |
| Vx | 0.5423 | 13 | 0.75 | 99.8 | + / + |
| pressure | **0.0823** | 13 | 0.49 | 12.5 | **- / -** |

**This is the strongest test the rule has had.** The density component sits *between* the two
thresholds (0.2623 against 0.2477 and 0.2728), so the rule makes opposite predictions for the
two trunks on the same target: positive for MLP, negative for Fourier. Pressure, at 0.0823,
is below even the FNO threshold and is predicted negative for both. Getting the density sign
split right would be evidence the thresholds carry real information rather than just ordering
a monotone curve.

All three components satisfy the scope condition established in
`DARCY_BETA_LADDER_V1.md`: coefficient of variation at most 0.75 and spread at most 99.8,
against the excluded Darcy beta 0.01 at CV 2.43 and spread 343.

## 6. Falsification, fixed in advance

Six predictions (3 components x 2 trunks). A wrong sign falsifies the rule for that trunk on
this operator class. The thresholds are applied unchanged and will not be refitted on this
data.

## 7. Commands

```bash
python scripts/prepare_cfd.py
for c in density Vx pressure; do
  python scripts/run_public_benchmark.py \
      --data-dir external_data/pdebench/cfd/${c}_t100_v1 \
      --architectures deeponet_mlp,deeponet_fourier \
      --steps 1500 --out results/cfd_${c}_v1 --device cuda
  python scripts/diagnose_fluctuation.py \
      --data external_data/pdebench/cfd/${c}_t100_v1 \
      --families cfd_${c}_v1=endtoend:test,val \
      --out results/fluctuation_cfd_v1/${c} --device cuda
done
```

## 8. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
