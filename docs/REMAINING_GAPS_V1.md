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

## Closed since the previous version

1. **FLOP accounting — closed to an estimate.** An analytic FFT term (~`2.5 n log2(n)` per
   real transform, four per spectral layer for forward and backward, because
   `torch.utils.flop_counter` registers no FFT op) raises FNO from 2.68e8 to 3.52e8 FLOPs per
   step. The matched DeepONet budgets rise to 6744 and 5559 updates, and re-running there
   gives -0.0892 and -0.2139 against FNO's +0.6503: unchanged, with a larger margin. The FFT
   term remains an **estimate**, reported separately from the measured dense count, and it
   belongs to FNO so any error favours FNO.
2. **FNO control on the system — closed via `src/fno1d_multi.py`.** FNO scores -0.0628,
   -0.8215 and -0.2501 per component, **worse than both DeepONet trunks on all three**,
   against +0.65 to +0.99 on every scalar problem. The FNO advantage is specific to
   single-field inputs and reverses on a system, which also explains why the threshold rule
   fails there.
3. **Physics-informed arm extended to a second horizon (T = 0.5)** on the same operator. The
   preregistered prediction — that it would not reach positive skill even where supervised
   training does — was **falsified**: it reaches +0.0088 and +0.0516 against supervised
   +0.1500 and +0.1878. The deficit remains in the mean share (0.071 and 0.157 against 0.017
   and 0.033).

## What remains open, with the evidence for why

4. **No multi-dimensional system, and no shallow water anywhere in the deposit.** A full
   listing of the 336 files in `doi:10.18419/darus-2986` was queried rather than assumed:

   * there is **no** shallow-water, SWE or similar dataset in this deposit at all;
   * the smallest multi-dimensional *system* is `2D_diff-react_NA_NA.h5`, **13.2 GB**
     (id 133017, MD5 `b8d0b86064193195ddc30c33be5dc949`), a two-component reaction-diffusion
     system;
   * everything larger is impractical: 2D CFD starts at 55.1 GB, 3D CFD at 66.1 GB, and
     several files exceed 88 GB.

   So the only realistic way to test a second, multi-dimensional system is
   `2D_diff-react_NA_NA.h5`, already added to `scripts/download_darus.py`. Its value is that
   it would show whether the rule's failure on systems generalises across a second and
   structurally different system, or was specific to compressible flow. Note that the study
   does already contain a **2D scalar** operator (Darcy) and a **1D system** (CFD); only the
   combination is missing.
5. **The threshold boundary remains tested on evolution problems only.** This was
   investigated rather than asserted: across a 1000-fold change in Darcy `beta` the target
   fluctuation energy stays between 0.492 and 0.564, so no steady-operator setting approaches
   a threshold (detail in `DARCY_BETA_LADDER_V1.md`). One direction does carry over to the
   system: pressure, with fluctuation energy 0.0744 below every threshold, was predicted
   negative and is negative for both trunks, so the rule keeps its *negative* predictive
   content on a system while losing its *positive* content.
6. **The physics-informed arm is still narrow**: one operator (Burgers), one viscosity
   (Nu = 0.01, the only one here with a trustworthy residual), two horizons. Its work per
   update is not matched to the supervised runs.

## What is not a gap

Update-count, wall-clock and measured FLOP budgets; parameter counts; seven forecast horizons
per evolution problem; two viscosities; the trunk-span bound; the physics-informed comparison;
and cross-operator transfer across four scalar operators are all covered.
