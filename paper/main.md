# Better Projection and Stable Fitting Do Not Guarantee Generalization in DeepONet

**Evidence-audited research draft v2 — 2026-09-24. Not a submitted manuscript.**

**Latest 2026-09-29 follow-up:** [incremental correction beyond nested-tuned linear maps](FOLLOWUP_INCREMENTAL_MODAL_20260929.md).
Sealed candidates improve relative L2 by3.0779% and2.8752% on two new Burgers confirmation
sets, passing predeclared effect and adjusted-interval criteria. Superiority over energy
truncation is not confirmed; linear Advection selects no correction. All new holdouts
are now consumed. Earlier failed confirmation results below are preserved.

**2026-09-28 follow-up:** [objective alignment and cross-fitted modal correction](FOLLOWUP_OBJECTIVE_MODAL_20260928.md).
The new development study includes a disjoint Burgers training split and linear Advection.
Normalized loss is not universally better; the CV rule does not beat energy selection.
These findings do not overturn the earlier failed confirmation or establish a new method.

**Earlier 2026-09-28 follow-up:** [regularized ridge residual with fresh confirmation](FOLLOWUP_RIDGE_RESIDUAL_20260928.md).
Forty additional neural runs selected a development improvement, but the500-trajectory
sealed confirmation did not meet the primary relative-L2 success criterion. Secondary
RMSE improved; this does not replace the failed primary endpoint or validate a new method.

**2026-09-28 follow-up:** the core results below are preserved; see the
[larger-sample and POD ridge addendum](FOLLOWUP_20260928.md) for 20 additional neural
records and 18 deterministic validation-tuned candidates. Schedule effects change
direction at n=1000; no universal deterioration claim is intended.

**Same-POD follow-up:** [matched linear/nonlinear readout results](FOLLOWUP_POD_READOUT_20260928.md)
add 30 neural runs. Nonlinear branches improve training in all 15 paired comparisons,
but worsen validation in all 15, even with the same POD decoder. Most validation squared
error is coefficient mismatch; this does not uniquely identify an optimization bottleneck.

This is the current manuscript. Earlier drafts are retained in
[the historical archive](archive/pre_audit_20260924/main.md) and are superseded.
The new public-data evidence is supervised operator learning, not a demonstrated
improvement to physics-informed training.

## Abstract

A learned output basis can approximate observed solution fields well even when the
operator mapping into that basis generalizes poorly. We investigate this distinction
through a provenance-tracked case study of DeepONet on a fixed-time subset of PDEBench
Burgers. We first replace an inconsistent historical attribution proxy with same-grid,
same-split, bias-aware orthogonal projection diagnostics. We then conduct matched
interventions on conditioning, residual initialization, learning-rate schedules and
batch size. The study contains 155 completed run records across four sequential
experiment families; ten are exact repetitions of earlier controls, not additional
independent evidence. A Fourier trunk has a lower offline projection error than an MLP
trunk without yielding a lower initial operator-prediction error. Zero-residual
initialization improves final validation error in 29 of 30 paired comparisons.
For 128 fixed training functions, cosine decay yields stable projection-level fitting
in all 20 decay runs, whereas constant learning rates do so in zero of ten small-batch
and one of ten full-batch runs. Nevertheless, validation error does not generally
improve; at batch size 32 it worsens for every paired seed of both trunk families,
including under RMSE. These findings distinguish representation, sampled training
convergence and validation behavior within the studied protocol. They do not establish
a novel optimizer, a universal failure mechanism, or state-of-the-art performance.

## 1. Research question and contribution

Our question is not whether a new network wins a benchmark. It is whether improving
a trunk's offline representability or the branch's training convergence reliably
improves predictions on unseen inputs. The empirical answer in the present study is no.

The contribution candidate is an auditable sequence of controlled diagnostic experiments:

1. Evaluate full target fields and predictions on identical points and samples, with
   explicit affine offsets and numerical rank.
2. Compare conditioning transformations within the same retained feature span.
3. Test whether the branch can fit small training subsets before inferring capacity limits.
4. Intervene on initialization and optimization while separately evaluating validation error.

The linear-algebra identities below are standard, not new approximation theory.
Neither orthonormalization, zero-residual initialization nor cosine decay is claimed
as a new method. The potential novelty lies in the combination of matched interventions,
negative results and explicit attribution boundaries; breadth and confirmatory evidence
remain insufficient for strong general claims.

## 2. Related work and overlap

Lee and Shin introduce two-step DeepONet training with trunk-first learning followed
by branch learning and orthonormalization [1]. Consequently, fixed-trunk training and
basis orthogonalization are necessary comparison points, not original contributions
of this draft. Our intervention freezes previously end-to-end-trained trunks; it is
not a reproduction of their complete trunk-training procedure.

Williams et al. examine learned PI-DeepONet bases, singular-value/coefficient behavior
and transfer learning [2]. This is directly relevant prior diagnostic work. Our
supervised case study cannot refute conclusions for their physics-informed protocols.

Lu et al. compare DeepONet and FNO with practical extensions, including POD-DeepONet,
across multiple benchmarks [3]. A credible extension of our work must include a
training-only POD baseline and a faithful two-step reference, rather than only our
own weak initial DeepONet configurations. PDEBench supplies the public data [4].

Zero-initialized residual mechanisms precede this study, for example ReZero [5].
Our zero final readout around an input-mean baseline is not identical to ReZero's
residual gates, and no dynamical-isometry claim is made. Cosine scheduling is also
established [6]; we use Adam with one decay segment and no restarts. Prior large-batch
generalization studies [7] further preclude presenting convergence-versus-generalization
as a new general discovery. We have not established a sharpness mechanism here.

## 3. Projection diagnostics and their limits

Let S and Y in R^(N x m) be target and predicted fields for the same m functions on
the same N-point grid. Let L contain known per-input offsets. Let A be the numerical
feature matrix, including a constant column where the model relaxation allows it.
Write Q for an orthonormal basis of its retained span and P = QQ^T.

Define the affine oracle S_star = L + P(S-L). If Y-L lies in span(A), orthogonality gives

\[
 \frac{\|S-Y\|_F^2}{\|S\|_F^2}
 =
 \underbrace{\frac{\|(I-P)(S-L)\|_F^2}{\|S\|_F^2}}_{E_{\rm repr}^2}
 +
 \underbrace{\frac{\|P(S-L)-(Y-L)\|_F^2}{\|S\|_F^2}}_{E_{\rm in}^2}.
\]

This is a **squared-error identity**, not an additive decomposition of relative L2.
When Y-L is not in the retained span, the identity above need not close. We therefore
record outside-span prediction energy and closure error in the projection audit.
Rank truncation makes the diagnostic conditional on the retained numerical space.

The oracle chooses coefficients independently for each known target; it uses labels
and does not produce a deployable input-to-output operator. A free samplewise constant
is a relaxation relative to a single shared model bias. The in-space discrepancy may
reflect optimization, branch capacity, training-data coverage or generalization.
It is not a uniquely identified causal "optimization error."

We separately report mean per-function relative L2,
m^(-1) sum_j ||s_j-y_j||_2 / ||s_j||_2, relative Frobenius error, and RMSE.
These quantities are not interchangeable. A train-test error difference is descriptive;
it is not an exact causal decomposition across different sample sets.

For fixed-trunk fitting, the reported excess is
sqrt(max(E_total^2 - E_repr^2, 0)), using the corresponding affine oracle and subset.
Tiny values are subject to floating-point cancellation. The predeclared fit criterion
is excess <= 0.01; attaining it does not imply a total prediction error below 1%.

### Numerical rank sensitivity

In the initial public experiment, test projection mean relative L2 is 0.19177 for
MLP and 0.04770 for Fourier at rcond=1e-12. At rcond=1e-6 these become 0.25818 and
0.05937. MLP numerical rank changes from 65 to 4–5; Fourier changes from 65 to 34–44.
Thus numerical near-null directions cannot be ignored when interpreting an oracle.
Subsequent conditioning and fitting experiments use rcond=1e-6.

## 4. Data, controls and sequential study design

We use PDEBench's 1D Burgers Nu0.001 file, verified by SHA-256, with 10000 trajectories,
201 stored time layers and 1024 cell-centered spatial points. The task maps t=0 to
t=1.0 (index 100), with stride-four point sampling to 256 points and no anti-alias filter.
One unused extra time-coordinate entry is excluded. The generator uses epsilon/pi
as the diffusion coefficient; we do not silently substitute the filename parameter
into our earlier PDE residual implementation.

The initial split is 1000 training, 200 validation and 200 test trajectories, sampled
with split seed 20260922. Input/target scalar normalization uses training arrays only.
DeepONet uses 64 trunk features and three-layer, width-128 branch/trunk MLPs;
the alternative trunk uses fixed random Fourier input features. The FNO reference
has four blocks, width 32 and 16 spectral modes. All training seeds are 0–4.

| Family | Run records | Main intervention | Evaluation status |
|---|---:|---|---|
| public_fixed_t1_v1 | 15 | MLP/Fourier DeepONet and FNO, 1500 updates | Train/validation/test; exploratory subset |
| public_coefficients_v1 | 40 | Conditioning x mean offset, 3000 branch updates | Same exploratory split |
| public_smallfit_v1 | 60 | n=32/128/1000 x random/zero readout | Train/validation only |
| public_stability_v1 | 40 | Constant/cosine LR x batch 32/128, n=128 | Train/validation only |

These families were designed sequentially after inspecting preceding results.
Protocols were fixed before their respective launches, not preregistered as one
independent confirmatory study. Ten stability controls reproduce earlier results
exactly and are not independent replicates.

For branch interventions, augment the trunk with a constant, truncate its float64 SVD
at rcond=1e-6, and compare U_r S_r V_r^T against U_r V_r^T after matching elementwise RMS.
Both retain 65 readout columns and the same retained span. New branches have three
width-128 tanh layers. Input-mean lifting adds each input's spatial mean to the predicted
residual; zero initialization sets the final readout weights and bias to zero.

Subsets of 32 and 128 rows are nested within a fixed permutation of the original
1000 training rows (seed 20260923). Trunks and normalization already saw those 1000
functions: this is a branch-fit diagnostic, **not few-shot training from scratch**.

The stability factorial holds zero initialization and n=128 fixed. Adam uses either
constant 1e-3 or cosine decay from 1e-3 to 1e-5 over 3000 updates, with batch 32 or 128.
All trials use their final prescribed checkpoint. Every 250 steps we log full training
and validation metrics; every 50 steps we atomically checkpoint model, optimizer,
random states, step and history. "Stable tail" requires the final four logged points
to meet the fit threshold. It does not establish stability at unobserved updates.

## 5. Results

### 5.1 Representability is not operator accuracy

| Initial method | Test mean relative L2 (mean ± seed SD) |
|---|---:|
| Input spatial-mean baseline | 0.37577 |
| MLP DeepONet | 0.39357 ± 0.01014 |
| Fourier DeepONet | 0.41231 ± 0.01344 |
| FNO | 0.22835 ± 0.00680 |

Fourier has a substantially smaller offline projection residual but not a better
learned prediction. Both initial DeepONets fail to beat the input-mean baseline.
FNO is the best tested model in this limited protocol, not universally superior.
Equal update counts do not match parameter counts or compute.

Conditioning interventions yield small average validation improvements but no
baseline-beating predictions. All eight final test group means lie between 0.41594
and 0.43832. This rules out presenting orthonormalization alone as a successful
performance repair in the tested regime; it does not invalidate two-step training
methods with different basis learning or optimization.

### 5.2 Zero initialization helps, but fitting and generalization remain distinct

Zero initialization improves final validation relative L2 in 29/30 paired comparisons;
all six group means and seed standard deviations improve. The exception is Fourier
at n=1000, seed 2. Shared inputs and related seeds do not constitute 30 independent
datasets, so we make no aggregate significance claim.

At n=32, all ten zero-initialized runs meet the final fit criterion. MLP random runs
also meet it, whereas Fourier random runs reach the threshold during training but
all finish above it. At n=128, only one of ten zero-initialized runs meets the criterion;
at n=1000, none does under the then-current constant-LR budget.

The best zero-init validation group is MLP/n=1000 at 0.38981, still worse than the
validation input-mean baseline 0.374509. A good starting predictor and better final
performance relative to random initialization do not guarantee a useful trained operator.

### 5.3 Learning-rate decay stabilizes fitting without generally improving validation

| Trunk | LR | Batch | Final train excess | Stable seeds | Validation relative L2 (mean ± SD) |
|---|---|---:|---:|---:|---:|
| MLP | constant | 32 | 0.034534 | 0/5 | 0.44657 ± 0.00981 |
| MLP | constant | 128 | 0.004113 | 0/5 | 0.50846 ± 0.02313 |
| MLP | cosine | 32 | 0.001117 | 5/5 | 0.45708 ± 0.00689 |
| MLP | cosine | 128 | 0.000061 | 5/5 | 0.47377 ± 0.01201 |
| Fourier | constant | 32 | 0.030453 | 0/5 | 0.55232 ± 0.00913 |
| Fourier | constant | 128 | 0.008727 | 1/5 | 0.56900 ± 0.01195 |
| Fourier | cosine | 32 | 0.005602 | 5/5 | 0.58755 ± 0.01125 |
| Fourier | cosine | 128 | 0.000809 | 5/5 | 0.59345 ± 0.01332 |

Cosine decay yields stable final-tail fitting in 20/20 runs. Constant LR yields
0/10 with batch 32 and 1/10 with batch 128. Every constant/full-batch run crosses the
fit threshold and later rebounds. Three MLP cosine/full-batch seeds rebound earlier
but stabilize in the final tail; "stable tail" must not be read as "never fluctuates."

At batch 32, cosine worsens final validation relative L2 for every paired seed:
the mean increase is 0.010512 for MLP and 0.035227 for Fourier. Validation RMSE also
increases, from 0.171690 to 0.176732 and from 0.201126 to 0.216168 respectively.
Thus metric mismatch alone does not explain the failure to generalize.
At batch 128, cosine improves MLP validation relative to constant/full-batch but
worsens Fourier validation. We do not claim that decay always harms validation.

![Training fit and validation behavior](../results/public_stability_v1/stability_curves.png)

Full-batch trials process four times as many examples at equal updates. The summaries
also compare full-batch 750 updates with mini-batch 3000 updates (96000 examples each),
but their update counts and decay phases differ. This is descriptive, not equal-compute
causal evidence. Recorded update times exclude evaluation/checkpointing and partly
overlap regression checks; they are not isolated throughput measurements.

## 6. Interpretation and limitations

The supported finding is **a within-protocol dissociation**: improved basis projection,
improved training fit and improved validation are distinct outcomes. The interventions
demonstrate that n=128 fitting can be stabilized without increasing branch size, while
poor validation persists. They do not identify a unique generalization mechanism.

The study has one public PDE, one coefficient, one forecast time and one split; no
resolution transfer, rollout, noisy-input or second-PDE confirmation. The test split
was exposed during early exploratory development, and validation guided later study
choices. Fresh unused trajectories are required for final confirmation.

Other limitations include weak initial DeepONet performance, fixed rather than
architecture-specific tuning, reused supervised trunks, numerical-rank sensitivity,
unequal sample exposure across batches, and only five seeds. Training-only POD,
faithful two-step training and better-tuned operator baselines remain missing.
A deployment claim would additionally require performance beyond simple baselines.

## 7. Historical correctness audit and withdrawn claims

The project previously found shared-grid helpers called on per-sample coordinates,
producing (B,B,N) instead of (B,N). Current sampler tests compare batched evaluation
against an independent per-sample oracle. This is a useful implementation lesson,
not evidence that other published systems contain the same bug.

We withdraw the old residual assertion as a mathematical test: small value error
does not ensure small derivative error or PDE residual. A supervised residual check
can motivate inspection, but is neither a necessary nor sufficient proof of sampler
correctness. Explicit shape and per-sample-value contracts are stronger direct checks.

Earlier claims that test_relL2 minus a truncated captured-energy proxy is pure
optimization error, that alignment has a problem-independent variance floor, and
that output-NTK/sharpness correlations establish a causal mechanism are not maintained.
Historical unknown source fingerprints remain unknown; historical p-values are not
reused as confirmatory evidence. The old "85% of error" norm-ratio interpretation
is replaced by matched squared-error audits. Full records remain archived.

## 8. Reproducibility and next confirmation

The four versioned result directories contain manifests, events, final metrics and
summaries. Dataset SHA-256, exact row identities, code hashes and environment records
are retained. Large downloaded data and resumable checkpoints remain local rather than
in Git. A redistributable release and author/affiliation details are not yet prepared.

The test suite passed 18 tests after revision, including manuscript evidence checks; CPU and CUDA forced-resume
tests verify parameter, metric and schedule consistency. One real interrupted
smallfit trial resumed at step 1350. These tests do not certify all hardware
power-loss scenarios. AI assistance was used for code and prose; numerical results
come from executed scripts. Authors must review claims, references and disclosures
before any submission.

Next experiments should distinguish data coverage from optimization: transfer the
matched constant/cosine comparison to 1000 training functions, add training-only
POD/ridge and faithful two-step controls, then test a second public PDE and a fresh
holdout under a locked protocol. A prospective diagnostic decision rule is a possible
future contribution, but has not been demonstrated.

## References

Primary sources checked 2026-09-24; versions are given without guessing later journal metadata.

1. Sanghyun Lee and Yeonjong Shin. *On the training and generalization of deep operator
   networks*. 2023. https://arxiv.org/abs/2309.01020
2. Emily Williams, Amanda Howard, Brek Meuris and Panos Stinis. *What do physics-informed
   DeepONets learn? Understanding and improving training for scientific computing
   applications*. 2024. https://arxiv.org/abs/2411.18459
3. Lu Lu et al. *A comprehensive and fair comparison of two neural operators (with
   practical extensions) based on FAIR data*. 2021 preprint; journal DOI
   https://doi.org/10.1016/j.cma.2022.114778 ; https://arxiv.org/abs/2111.05512
4. Makoto Takamoto et al. *PDEBench: An Extensive Benchmark for Scientific Machine
   Learning*. NeurIPS Datasets and Benchmarks, 2022. https://arxiv.org/abs/2210.07182
5. Thomas Bachlechner et al. *ReZero is all you need: fast convergence at large depth*.
   UAI / PMLR 161, 2021. https://proceedings.mlr.press/v161/bachlechner21a.html
6. Ilya Loshchilov and Frank Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts*.
   ICLR 2017. https://arxiv.org/abs/1608.03983
7. Nitish Shirish Keskar et al. *On Large-Batch Training for Deep Learning:
   Generalization Gap and Sharp Minima*. ICLR 2017. https://arxiv.org/abs/1609.04836
