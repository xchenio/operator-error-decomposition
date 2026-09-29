# Whole-field relevance: useful rejection, unconfirmed selection advantage

2026-09-29 addendum. Source/protocol b2d554c; candidate/model seal8a4020b precedes all
new target extraction/generation. Four NEW500-trajectory confirmation sets exclude all
previously consumed public splits. They too are now consumed. No adaptive tuning or
retraining follows confirmation. Historical positive/negative results remain unchanged.

## Method and second nonlinear problem

The proposed gate measures a mode's CV coefficient squared-risk reduction as a fraction
of the entire weighted baseline FIELD squared risk, including out-of-basis projection
error. It retains an old-stable mode only when this fraction is>=0.1% in each fold.
Hard and shrunken joint masks additionally require>=1% mean nonsquared relative-L2
improvement in each training fold. Internal gate evaluation follows mask selection on
the same folds and is not an unbiased final assessment. Thresholds are fixed before
development inspection; no sweep is conducted. No-correction, old stable/shrink rules
and matched-count energy truncation reuse the same five branch predictions. There is
no inference-compute reduction merely from masking this fixed-size branch.

Development selects hard versus field-shrink, with>=1% gain over linear or fallback
to none. The old stable/shrink selector is independently locked on development. Twelve
confirmation comparisons (four tasks versus linear, energy and old selector) use
Bonferroni two-sided99.5833% paired trajectory bootstrap intervals,10000 resamples.
Success requires>=1% relative mean gain and an interval wholly below zero. The estimand
averages seed-specific per-trajectory errors, not ensemble predictions. Intervals
condition on fitted seeds and do not establish broad PDE-distribution generalization.

The second nonlinear PDE is a CUSTOM SYNTHETIC Allen-Cahn operator:
u_t=0.01u_xx+5u(1-u^2), periodic x in[0,2), t=0→0.5. It is not a downloaded benchmark
reproduction. There are1000 training,200 development and500 newly generated confirmation
trajectories from random12-mode smooth Fourier initial conditions; fixed256-point grid.
Representation400/correction600, rank16, nested linear tuning and residual training
follow the prior incremental-modal protocol. Only11 new neural fits are needed;
the33 public-task fits are reused unchanged. Masks do not constitute independent fits.

Fourier diffusion/analytic reaction Strang splitting uses dt0.001. Separate QA IDs
test constant exact flow, equilibria, pure diffusion, step halving, spatial refinement
and independent DOP853 integration. Maximum step-refinement relative error2.471e-7,
spatial error5.721e-14, independent-reference error3.091e-7 pass preset tolerances.
This is finite numerical QA, not a proof of solver accuracy for all possible inputs.
The nonlinear family follows the [PINNs authors' equation documentation](https://github.com/maziarraissi/PINNs/blob/master/docs/index.md),
with our explicitly different diffusion coefficient and random initial distribution.
The public1D reaction-diffusion candidate was avoided due to an unresolved
[upstream consistency concern](https://github.com/pdebench/PDEBench/issues/94), not a
claim that we independently proved every released file incorrect.

## Results

Retained mode orders: original Burgers1/2/4 (mode3 removed); independent Burgers1/2;
Advection none; Allen-Cahn1–15 (mode16 removed). Different bases do not identify the
same physical modes merely because their ranks match. Field and field-shrink yield
identical predictions here; the locked candidates are field on nonlinear tasks and
none on Advection. Only original Burgers differs from same-count energy truncation.

| Development | Linear | Field candidate | Energy control | Old selected rule |
|---|---:|---:|---:|---:|
| Original Burgers | 0.373252 | 0.362597 | 0.364460 | 0.362262 |
| Independent Burgers | 0.409382 | 0.402368 | 0.402368 | 0.402073 |
| Advection | 0.019756125100 | 0.019756125100 | 0.019756125100 | 0.019756125100 |
| Synthetic Allen-Cahn | 0.458885 | 0.359218 | 0.359218 | 0.359048 |

Mean per-sample relative L2 on new confirmation trajectories (lower is better):

| Confirmation | Linear | Field candidate | Energy control | Old selected rule |
|---|---:|---:|---:|---:|
| Original Burgers | 0.405879 | 0.393444 | 0.394195 | 0.393909 |
| Independent Burgers | 0.402369 | 0.389518 | 0.389518 | 0.388486 |
| Advection | 0.020532878061 | 0.020532878061 | 0.020532878061 | 0.020532878061 |
| Synthetic Allen-Cahn | 0.467579 | 0.366325 | 0.366325 | 0.366134 |

| Locked candidate versus linear | Relative gain | Adjusted difference interval | Success |
|---|---:|---|---|
| Original Burgers | 3.0636% | [-0.019278,-0.005834] | Yes |
| Independent Burgers | 3.1939% | [-0.018854,-0.007294] | Yes |
| Advection | 0% | [0,0] | No gain; rejection |
| Synthetic Allen-Cahn | 21.6550% | [-0.107219,-0.095285] | Yes |

No superiority comparison versus energy or the old selected rule meets the predeclared
success criteria. Original Burgers versus energy gains0.1904%, interval
[-0.003182,0.001841]; versus old stable gains0.1179%, interval[-0.001975,0.001270].
Independent Burgers matches energy exactly and is0.2657% worse than old shrinkage,
interval for absolute difference[-0.000242,0.002512]. Advection is unchanged.
Allen-Cahn equals energy and is0.0521% worse than the old all-mode rule, with interval
[0.000153,0.000233]. The latter is a small but statistically resolved deterioration
under the declared comparison family, not a practical improvement. No effect threshold
was declared for claiming harmfulness; report its magnitude alongside the interval.

## Consequence for the research hypothesis

The new rule rejects negligible Advection corrections directly at the training-internal
field-risk stage. It also produces a non-prefix mask on one Burgers split. Those are
mechanistic properties, not evidence of a better predictive selection algorithm.
Hard removal of a low-contribution but helpful mode can hurt, as Allen-Cahn illustrates.
The confirmed cross-task finding is restrained nonlinear correction over the tested
linear comparator, including a second nonlinear SYNTHETIC PDE; the original ambition
of a superior modal selection rule remains unconfirmed. This is not state-of-the-art
evidence against matched FNO/other nonlinear baselines, nor a public cross-PDE benchmark.

Preserve old continuous shrinkage/all-mode correction as strong comparators. Do not
search new thresholds on these consumed confirmation sets. A future hypothesis could
use continuous field-aware shrinkage and uncertainty rather than mandatory deletion,
and distinguish a prediction-error objective from an explicit complexity/latency objective.
It requires new prospective data and genuinely matched costs. This extension has not
been implemented or validated here. Full evidence: [FIELD_MODAL_V1](../docs/FIELD_MODAL_V1.md).
