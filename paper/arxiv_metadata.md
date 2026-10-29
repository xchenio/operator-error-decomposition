# arXiv submission metadata

Prepared 2026-10-28.

The abstract below is written to **JCP's 250-word cap** so the same text can be pasted into the
journal submission without editing. arXiv's own limit is 1920 characters, which this also
satisfies.

## Title

An Exact A Posteriori Error Decomposition for Operator Learning: Conservation, Fluctuation, and
the Limit of Branch Capacity

## Primary category

`cs.LG` (Machine Learning)

## Cross-list

`math.NA` (Numerical Analysis)

Rationale: the object contributed is a decomposition and a bound, which is numerical analysis, but
the systems diagnosed are learned operators and the intended readership is the scientific
machine-learning community. Do not additionally cross-list to `physics.flu-dyn`: it dilutes the
listing without adding a distinct readership.

## Abstract

On periodic Burgers the spatial mean is conserved, so relative L2 can hide that a model predicts
nothing beyond a conserved scalar. We decompose squared prediction error exactly into a
conserved-mean part and a fluctuation part and prove a hard upper bound on fluctuation skill
that depends only on the trunk span.

Against a defined zero-fluctuation reference on 1D Burgers (Nu = 0.001, t = 1.0), MLP DeepONet
scores -0.0042 ± 0.0072 and Fourier DeepONet -0.0934 ± 0.0886 in per-sample fluctuation R^2,
while FNO scores +0.6503 ± 0.0222 on identical data, split and budget.

The deficit is horizon-dependent rather than absolute. Across seven horizons every architecture
decays monotonically, giving an interpolated skill horizon near 0.98 (MLP), 0.86 (Fourier) and
1.94 (FNO). The decay tracks nonlinear spectral evolution rather than elapsed time, and is
absent on linear advection.

An a-priori fluctuation-energy threshold, fitted on two scalar families and never refitted,
predicts the sign of skill on held-out settings. The record is stratified since the 55 are not
equally informative: 27 of 30 discriminating, 5 of 6 at the boundary, against 16 of 30 for
always positive.

Physics-informed training likewise fails, its deficit again localising in the conserved mean,
and the dissociation replicates on a second disjoint split. We claim a diagnostic decomposition
with two proved propositions and a reproducible negative result, not a theorem for the empirical
rule.

## Keywords (6)

operator learning; error decomposition; conservation laws; spectral bias; scientific machine
learning; reproducibility

## Highlights (5, each within 85 characters)

- Exact orthogonal split of squared error into conserved-mean and fluctuation
- Hard upper bound on fluctuation skill that depends only on the trunk span
- MLP DeepONet sits at the zero-fluctuation reference; FNO reaches R2 = +0.65
- Skill horizon 0.98 MLP, 0.86 Fourier, 1.94 FNO on identical budgets
- Dissociation replicates on a second split sharing no trajectory with it

## Files to include

- LaTeX source compiled with the `elsarticle` class
- `experiments/prediction_ledger.json` and `experiments/prediction_ledger_rs2.json`
- Any figures

## Notes for the author before submitting

1. The Data availability statement still carries the marker
   `[fill: Zenodo concept DOI once the deposit is public]`. Replace it with the concept DOI.
2. arXiv requires preprints to be labelled as such. The arXiv-only references already carry
   `https://arxiv.org/abs/...`; confirm the surrounding text says "preprint".
3. JCP imposes no page limit, so the ~30-page length is not a defect in itself, but the handling
   Editor may request a shorter version. Compressing to about 22 pages reduces that risk.
