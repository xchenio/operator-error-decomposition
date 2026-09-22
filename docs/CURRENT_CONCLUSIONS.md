# Current conclusions

**Authority.** This document and `paper/main.md` supersede older entries in `README.md` and
`paper/short_paper.md`. Historical claims remain recorded in `docs/EXPERIMENT_LEDGER.md`, but
are not evidence for the current paper.

## Supported by the committed results

1. A shared-grid evaluator used with per-sample coordinates can silently broadcast a target from
   `(B, N)` to `(B, B, N)`. In this project it affected Burgers initial conditions,
   diffusion--reaction forcing, and Darcy coefficients. It can produce plausible losses and a
   coherent but false experimental story.
2. A supervised model's autograd PDE residual, compared with the trivial-solution residual, and
   explicit sampler shape assertions detect this failure mode. Finite-difference residuals of
   stored reference solutions do not detect it.
3. For a fixed trunk feature matrix, `LB = sqrt(1 - captured_energy)` is the projection-error
   lower bound for any linear DeepONet branch readout. Together with train/test error it is used
   here to separate representation, optimization, and generalization gaps.
4. In the corrected 1D Burgers experiments, representation-side alignment improves the mean in
   all ten reported cells, but is not the preferred practical intervention: random Fourier trunk
   features are cheaper and competitive or better.
5. In corrected 2D Burgers at width 32, the MLP baseline is representation-limited and collapses
   near the zero solution. Fourier features repair substantially more of this failure than the
   alignment loss. The analytic prior is not better than a random smooth target subspace.
6. The reported alignment mechanism is empirical: output-NTK effective rank increases and the
   measured loss sharpness decreases. It is evidence consistent with regularization, not a proof
   of a universal causal mechanism.

## Claims explicitly retracted or unsupported

- Trunk-basis collapse is the universal root cause of PI-DeepONet failure.
- Subspace alignment is necessary, optimal, or complementary to Fourier features.
- Alignment gain follows a linear law determined by baseline cross-seed standard deviation.
- Any single-seed 20k/50k result establishes a statistically reliable method ranking.

## Scope and next falsification tests

The evidence currently covers synthetic Burgers (1D/2D), diffusion--reaction, and Darcy
implementations in this repository. It does not establish transfer to public benchmarks, FNO/PINO
architectures, real-world problems, or high-accuracy regimes. The next protocol therefore tests:

1. the sampler-health checks on every in-repository PDE;
2. the lower-bound attribution on a non-DeepONet architecture; and
3. a public PDEBench dataset using a resumable, manifest-driven run layout.
