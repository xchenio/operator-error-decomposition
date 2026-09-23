# Current conclusions

> 2026-09-23 fixed-trunk update: all 40 predeclared conditioning / mean-lift runs
> completed. Whitening gives small average paired validation gains, not a successful
> performance repair: every arm remains worse than the initial-mean baseline. See
> PUBLIC_COEFFICIENTS_V1.md. Prioritize small-sample branch fit and zero-residual
> initialization checks; conditioning alone is not supported as the primary solution.

> Public-data update (2026-09-22): read `PUBLIC_BENCHMARK_V1.md` for the completed
> 15-run matched supervised pilot. Test mean relative L2: MLP DeepONet 0.39357,
> Fourier DeepONet 0.41231, FNO 0.22835; initial spatial-mean baseline 0.37577.
> Fourier has a better offline projection fit but not better learned predictions.
> The older Fourier recommendation below does not generalize to this public-data protocol.
> Prioritize controlled coefficient-learning / conditioning experiments, not model scaling.

> 2026-09-22 audit update: the legacy LB and optimization-gap claims below are not
> certified by their original implementation. Read `AUDIT_V2.md` first. Full-grid
> recomputation supports a large 2D projection component, but not pure causal attribution
> to architecture capacity. Historical p-values and variance-floor claims require regrouping.

**Authority.** The dated audit and public-data updates above override conflicting legacy claims
below and in the paper. The historical text of this document and `paper/main.md` superseded older entries in `README.md` and
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
