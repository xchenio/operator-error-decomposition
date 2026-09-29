# Current conclusions

> Latest2026-09-29 UNCERTAINTY_MODAL_V1 confirmation: the continuous uncertainty-penalized
> shrinkage FAILED its primary endpoint on four sealed 500-ID sets. 0 of 12 comparisons
> against strong controls (old continuous shrinkage / all-mode / matched-energy) pass; the
> new rule is worse than old continuous shrinkage on both Burgers splits (0.419982 vs
> 0.419806; 0.384878 vs 0.382451). Allen-Cahn is degenerate (all arms identical); Advection
> differences are ~1e-7. The only two passing comparisons are versus no correction, which
> restate an earlier effect. The modal-shrinkage line is CLOSED under its own predeclared
> rule; 2000 targets consumed, no retuning and no sixth variant. See UNCERTAINTY_MODAL_V1.md.
>
> Latest2026-09-29 E1 completed: 175 stored runs across five public families were
> recomputed into results/fluctuation_v1 by scripts/diagnose_fluctuation.py; every
> recomputed mean_relative_l2 reproduces its stored value exactly (max difference
> 0.00e+00). **Every DeepONet arm has fluctuation R^2 <= 0** (best -0.0009+-0.0377 at
> MLP/n=1000/cosine, i.e. indistinguishable from the zero-fluctuation reference), while
> FNO is +0.6282+-0.0167 on identical data and budget. Training the branch harder moves
> R^2 from ~0 to negative (down to -1.24 and -15.3), never positive. Whitening and mean
> lifting do not change this. E0 (step-budget ladder) is still required before any
> architectural claim. See docs/FLUCTUATION_DIAGNOSIS_V1.md section E1.
>
> Latest2026-09-29 mean/fluctuation decomposition (new): on the public fixed-t1 protocol the
> target spatial mean is exactly the input spatial mean (Burgers conservation; corr 1.000000,
> mean |dy-dx| 3.18e-05), so the "input-mean baseline" is a conservation-law predictor, and
> fluctuation carries only 24.4% of target energy. Recomputed from stored checkpoints:
> fluctuation R^2 is -0.0056+-0.0084 (MLP DeepONet), -0.0124+-0.0319 (Fourier DeepONet) and
> +0.6282+-0.0167 (FNO). DeepONet therefore has zero or negative fluctuation skill, not merely
> a weak baseline. This reframes the dissociation as an architectural bottleneck in the
> input-to-coefficient map. Step-budget sufficiency is NOT yet established (all public runs
> use 1500 updates) and is the next required control. See docs/FLUCTUATION_DIAGNOSIS_V1.md.

> Latest2026-09-29 FIELD_MODAL_V1: whole-field relevance (including projection error)
> keeps original Burgers modes1/2/4, independent1/2, Advection none, synthetic Allen-Cahn
> 1–15. Sourceb2d554c, selectionseal8a4020b BEFORE four new500-ID confirmation sets.
> New confirmation gains versus tuned linear:3.0636%,3.1939%,21.6550% for the three
> nonlinear cases, passing predeclared effect and12-comparison adjusted intervals.
> Advection rejects correction. Allen-Cahn is custom numerically validated synthetic
> data, NOT a downloaded public benchmark. Eleven new fits; old public models reused.
> No new rule superiority over energy or old selector is confirmed. Allen-Cahn is
> slightly worse than old all-mode correction (0.0521%, difference interval above0).
> Low contribution does not imply a coefficient should be deleted. Prefer retaining
> old strong controls; investigate continuous field-aware shrinkage/uncertainty only
> under a new protocol, not by retuning on these2000 now-consumed confirmation targets.

> Latest2026-09-29: INCREMENTAL_MODAL_V1.md supersedes the next-step proposals below.
> Nested training-only tuning strengthens linear maps;33 neural fits compare identical
> residual predictions under none/all/stable/energy/shrinkage. Candidates committed at
> ad94e69 BEFORE three500-ID confirmation sets were opened. Original Burgers stable
> improves0.387273→0.375353 (3.0779%); independent Burgers shrink improves0.379517→0.368605
> (2.8752%). Both pass>=1% gain and Bonferroni-adjusted paired bootstrap criteria.
> This confirms incremental correction over the tested tuned linear comparator, not
> global linear optimality or superiority over FNO. Hard masks still equal energy order.
> Rule superiority over energy FAILS: original identical; independent gain0.2013% with
> interval crossing zero. Advection falls back to none, L2=0.024182229260. All three
> confirmation sets are now CONSUMED; no adaptive reuse as untouched holdouts.
> Next research: field-risk relevance/rejection with matched energy/shrinkage controls,
> and a genuinely second nonlinear PDE. These are proposals, not validated innovations.

> 2026-09-28 objective/modal follow-up: see OBJECTIVE_MODAL_V1.md and the dated paper
> addendum. Guarded normalized squared loss improves full-data development relative L2
> from0.366400 to0.358258 on original Burgers and0.407464 to0.398468 on a completely
> disjoint training/development split (five paired seeds improve on each). This is not
> the nonsquared primary metric itself, nor a fresh confirmation of the old model.
> Strict train-internal CV selects leading4 and2 modes respectively, identical to energy
> truncation. No incremental selection-rule novelty or superiority is established.
> The earlier500-ID confirmation failure below remains valid. New reserved labels stay
> unopened until fixed models, comparisons and success criteria are committed.
> Final status:123/123 records complete,48 ridge grid refits. Advection is a linear
> negative control: normalized loss worsens0.169283→0.175051; tuned pure ridge0.015378
> is far better. All16 modes are selected, making mask controls degenerate. The rule
> detects correctable error relative to a weak anchor, not uniquely nonlinear benefit.
> Next priority: cross-fit incremental benefit beyond tuned linear correction, with
> no-correction and shrinkage controls. No cross-PDE universal improvement is supported.

> 2026-09-28 frozen-ridge residual historical update: see RIDGE_RESIDUAL_V1.md.
> 40 runs +8 ridge refits completed; development selects width32/penalty1 with L2=0.366400
> versus frozen ridge0.373656. Selection was committed BEFORE opening500 fresh trajectories.
> Confirmation L2=0.405135 versus frozen ridge0.405354 (only0.054% gain; paired95% interval
> for difference[-0.007498,0.008923]). Predeclared primary success FAILS. Expanded ridge
> and initial mean are slightly better in confirmation L2. Secondary RMSE improves5/5
> seeds, mean0.147701 versus0.150931. Do not replace the failed primary endpoint with RMSE.
> Next: development-only objective/metric alignment diagnostics, then modal learnability
> shrinkage with cross-fitting. Both are proposals; reserve a NEW confirmation set before
> adaptive follow-up. The500 current confirmation IDs are now consumed, not reusable holdout.

> 2026-09-28 same-POD update (latest): 30/30 readout runs completed; see
> PUBLIC_POD_READOUT_V1.md. Identical POD bases / input normalization / mean offsets:
> nonlinear MLP improves train L2 and RMSE in 15/15 pairs versus Adam affine control,
> but worsens both validation metrics in 15/15. MLP validation means at ranks16/32/64
> are 0.398761 / 0.401114 / 0.402642, all worse than ridge and mean baseline.
> Coefficient mismatch contributes roughly 93–99% of MLP validation squared error,
> but train coefficients are also not fully fit. This localizes error to coefficients,
> not uniquely to insufficient nonlinearity or optimization. Next: restrained nonlinear
> correction to a frozen ridge map, regularization/early stopping, fresh confirmation split.

> 2026-09-28 follow-up: PUBLIC_SCALE1000_V1.md reports 20 completed n=1000 controls
> and 18 POD ridge candidates. Cosine improves validation relative L2 in 10/10 pairs,
> unlike n=128; full training fit remains unmet. MLP cosine mean 0.374807 is slightly
> worse than baseline 0.374509. Selected POD ridge is 0.373656, a tiny validation-tuned
> difference, not confirmed test superiority. Next: nonlinear readout on identical POD bases.

> 2026-09-24 stability update: 40/40 n=128 zero-init runs completed;
> PUBLIC_STABILITY_V1.md is the latest evidence. Cosine decay achieves stable final-tail
> projection fitting in 20/20 runs; constant LR achieves 0/10 (batch32), 1/10 (batch128).
> Validation does not generally improve, including RMSE: convergence is not generalization.
> Next: matched stable-training controls on 1000 samples before larger architectures.

> 2026-09-24 smallfit update: 60/60 runs complete; see PUBLIC_SMALLFIT_V1.md.
> Zero-residual initialization improves final validation in 29/30 matched pairs and all
> six group means. At n=32 all ten zero-initialized models reach their own projection-fit
> threshold; n=128 and n=1000 do not reliably do so. All group validation means still
> trail the initial-mean baseline. Keep zero initialization, then test optimization
> stability; these are pretrained-trunk diagnostics, not few-shot generalization claims.

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
