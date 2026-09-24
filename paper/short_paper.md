# Better Projection and Stable Fitting Do Not Guarantee Generalization in DeepONet

**Short research draft — revised 2026-09-24. Not submitted; no venue acceptance or page-limit compliance implied.**

## Abstract

We study whether offline basis quality and stable training fit predict better operator
generalization. In a supervised, fixed-time PDEBench Burgers case study, a Fourier
trunk has a better projection oracle than an MLP trunk without better initial
prediction accuracy. Zero-residual initialization improves final validation error in
29/30 matched comparisons. At 128 training functions, cosine decay gives stable
projection-level fitting in 20/20 runs, while constant learning rates do so in 0/10
small-batch and 1/10 full-batch runs. Yet at small batch size, decay worsens validation
error in every seed of both trunk families. This persists under RMSE. These results
support separating representation diagnostics, training convergence and validation;
they do not establish a new optimizer or universal mechanism.

## 1. Scope and prior work

Two-step DeepONet training and orthonormalization already exist (Lee and Shin,
https://arxiv.org/abs/2309.01020). Learned PI-DeepONet basis diagnostics are also studied
by Williams et al. (https://arxiv.org/abs/2411.18459). Our contribution candidate is a
reproducible intervention-based case study, not the invention of these tools.
This paper's current public-data experiments are supervised, not physics-informed.

## 2. Matched diagnostic

For target fields S, prediction Y, known offset L and a retained numerical feature
space with orthogonal projector P, the affine oracle is L+P(S-L). If Y-L is in this
space, squared Frobenius error splits into projection residual and in-space discrepancy.
The latter is not uniquely a pure optimization error. Mean relative L2 is reported
separately; subtracting unmatched norms or different splits is not an error identity.

Our protocol uses 1000/200/200 train/validation/test trajectories from PDEBench Burgers
Nu0.001, t=0 to t=1, on a 256-point subsampled grid. Five seeds share fixed data.
Initial comparisons use 1500 updates; branch interventions use 3000.
Smaller branch-fit subsets inherit trunks and normalization trained on 1000 functions,
so this is not few-shot learning.

## 3. Findings

Initial test mean relative L2 is 0.39357 (MLP), 0.41231 (Fourier) and 0.22835 (FNO);
the input-mean baseline is 0.37577. Fourier's offline projection fit is better but its
learned predictions are not. Conditioning alone does not repair performance.

At n=32, all ten zero-initialized runs fit to within the predeclared excess-Frobenius
threshold of 0.01. Larger subsets do not reliably fit under constant learning rates.
We then fix n=128 and zero initialization, comparing constant 1e-3 against cosine
decay to 1e-5 and batches 32/128. Stable fitting requires the final four logged points
to meet the threshold, not merely the final point.

| Trunk | Stable, constant b32 | Stable, constant b128 | Stable, cosine b32 | Stable, cosine b128 |
|---|---:|---:|---:|---:|
| MLP | 0/5 | 0/5 | 5/5 | 5/5 |
| Fourier | 0/5 | 1/5 | 5/5 | 5/5 |

For batch 32, validation relative L2 changes from 0.44657 to 0.45708 (MLP) and
0.55232 to 0.58755 (Fourier), worsening in every paired seed. Validation RMSE also
worsens. MLP full-batch is an exception: decay improves validation relative to constant
full-batch. Every group still trails the input-mean baseline.

## 4. Limits and conclusion

The four experiment families contain 155 run records, including ten repeated controls,
not 155 independent confirmations. This is one PDE, time, parameter and split; later
experiments evaluate validation only. Full batches process four times as many samples
per update. The early test set is exploratory, and final confirmation needs new data.

The result is an empirically supported separation of representability, fitting and
generalization within this protocol. Stronger publication claims require faithful
two-step and POD baselines, data-scaling controls, a second PDE, and a locked fresh
holdout. Historical causal and variance-floor claims are withdrawn.

See [the current full manuscript](main.md) for complete methods, references, tables,
provenance and withdrawals, and [the stability report](../docs/PUBLIC_STABILITY_V1.md)
for per-seed artifacts.
