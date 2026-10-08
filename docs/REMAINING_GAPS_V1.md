# Remaining gaps, and the two that were closed

Date: 2026-10-08. Supersedes the earlier version of this file.

## Gap 1: the threshold boundary on a steady operator — CLOSED AS UNTESTABLE

Both additional Darcy `beta` files were fetched and prepared. The a-priori statistics, taken
before any run, refuted the hypothesis: across a 1000-fold change in `beta` the target
fluctuation energy only moves between 0.492 and 0.564, always far above every threshold, and
the smoothest setting has the *highest* fluctuation energy.

`beta = 10.0` was run and confirmed the prediction (3 of 3). `beta = 0.01` failed outright and
was **excluded** for two measured reasons: its validation curves are non-monotone, and its
target norms have a coefficient of variation of 2.43 with a 343-fold spread, against at most
0.66 and 100-fold in every reported family, which makes the per-sample metric uninterpretable
there. Full detail and the scope condition this established are in
`DARCY_BETA_LADDER_V1.md`.

**Conclusion: no Darcy setting in this deposit approaches a threshold, so the boundary on a
steady operator is untestable with the available data.** This is reported as a data limitation.
It was not worked around by refitting the thresholds or by redefining the metric.

## Gap 2: a system of equations — CLOSED, AND THE RULE FAILED

`1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5` (12.4 GB, three coupled fields) was
fetched, prepared and run. Sign accuracy was **3 of 6**, with every prediction of positive
skill wrong, and the thresholds were not refitted. Full detail in `CFD_SYSTEM_V1.md`.

**The rule therefore has a stated boundary of validity: scalar operators, not systems of
coupled fields.** The manuscript's claim was narrowed rather than defended.

This is arguably the most useful single result of the last round: a predictive rule that had
passed three out-of-sample tests was finally broken by a structurally different setting, and
the break was recorded instead of repaired.

## What remains open

1. **FLOP accounting is still incomplete.** FFT work is not counted by
   `torch.utils.flop_counter`, so equal measured FLOPs are not provably equal real work. The
   uncounted part belongs to FNO, so the error favours FNO and the sign of the conclusion is
   safe, but a true accounting needs a custom counter.
2. **No FNO control on the system.** `FNO1d` requires the input and the coordinate to have
   equal length, which the three-channel input violates. `src/fno.py` is hashed into every
   existing run identity, so a multi-input variant would have to be a new module.
3. **The physics-informed arm is narrow**: one operator, one viscosity, and only Nu = 0.01
   admits a trustworthy residual because Nu = 0.001 is under-resolved. Its work per update is
   also not matched to the supervised runs.
4. **No shallow-water or multi-dimensional system.** Only the 1D CFD system was tested, and
   the rule already fails there, so it is unclear whether any system would support it.
5. **The threshold boundary was tested on evolution problems only** (the viscosity ladder).

## What is not a gap

Update-count, wall-clock and measured FLOP budgets; parameter counts; seven forecast horizons
per evolution problem; two viscosities; the trunk-span bound; the physics-informed comparison;
and cross-operator transfer across four scalar operators are all covered.
