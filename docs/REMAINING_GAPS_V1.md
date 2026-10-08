# Remaining gaps: status after the last round

Date: 2026-10-08.

## Closed since the first version of this file

1. **FLOP accounting — closed to an estimate.** An analytic FFT term (~`2.5 n log2(n)` per
   real transform, four per spectral layer for forward and backward, because
   `torch.utils.flop_counter` registers no FFT op) raises FNO from 2.68e8 to 3.52e8 FLOPs per
   step. Matched DeepONet budgets rise to 6744 and 5559 updates, giving -0.0892 and -0.2139
   against FNO's +0.6503: unchanged, with a larger margin. The FFT term remains an **estimate**,
   reported separately from the measured dense count, and it belongs to FNO so any error
   favours FNO.
2. **FNO control on the 1D system — closed via `src/fno1d_multi.py`.** FNO scores -0.0628,
   -0.8215 and -0.2501 per component, worse than both DeepONet trunks on all three.
3. **FNO control on the 2D system — closed via `src/fno2d_multi.py`.** FNO scores +0.9869 and
   +0.9879 against Fourier DeepONet +0.2377 and +0.3734 and MLP +0.0310 and +0.0910. So FNO
   dominates here exactly as on the scalar families, and the 1D CFD system is the single
   exception in the study on every architecture — not a consequence of coupling.
4. **Physics-informed arm extended** to a second horizon (Burgers T = 0.5) and a second
   operator (linear advection). It recovers at most 0.38 of supervised fluctuation skill.
5. **A second, multi-dimensional system tested**, which **withdrew an over-strong narrowing**
   ("scalar operators only") that had been introduced after the 1D CFD failure.
6. **The leading explanation for the CFD failure was eliminated without new data**: Darcy
   beta 1.0 at fluctuation energy 0.4964 holds while CFD Vx at 0.4659 fails, so fluctuation
   energy level is not the discriminator; coupling is not either, because the 2D system holds.

## What remains open

1. **Why 1D compressible CFD fails is not established.** This is now a sharper open question
   than before, because three candidates have been eliminated and the one substantial miss has
   been localised: Vx is the only setting in the study where the mean share dominates the error
   (0.598 and 0.747 against at most 0.017 everywhere else) and where the spatial mean is not
   conserved. The other two misses are inside |R^2| < 0.03, indistinguishable from the
   zero-fluctuation reference. Closing this would need a compressible system at a different
   parameter set; none of the files in this deposit is small enough (2D CFD starts at 55 GB).
2. **FLOP accounting is an estimate, not a measurement.** A true count needs a custom counter
   that registers FFT work. The direction of the error favours FNO, so the sign of the
   conclusion is safe.
3. **No shallow-water data anywhere in the deposit** (verified by listing all 336 files), so
   there is no incompressible multi-dimensional system available.
4. **The physics-informed arm covers two operators and three horizons**, not more, because
   only Burgers Nu = 0.01 and linear advection admit a trustworthy residual with this data
   (Nu = 0.001 does not fit, R^2 = 0.79, and diff-sorp's stored initial state is degenerate).
   Its work per update is also not matched to the supervised runs.

## What is not a gap

Six settings (linear advection, Burgers at two viscosities, diffusion-sorption, steady 2D
Darcy, a 1D three-field system, a 2D two-field system), a physics-informed arm, update-count,
wall-clock and measured-plus-estimated FLOP budgets, parameter counts, seven forecast horizons
per evolution problem, the trunk-span bound, and 55 preregistered held-out sign predictions of
which 52 were correct.
