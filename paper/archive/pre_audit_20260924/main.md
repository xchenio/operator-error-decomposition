# What Really Limits Physics-Informed Operator Learning? A Silent Broadcasting Pitfall, an Error-Attribution Diagnostic, and Evidence that Subspace Alignment Regularizes Rather than Expands

**Draft v1 — 2026-09-17**
**Target**: AI4Science / ML4PS workshop (short track) or *J. Machine Learning for Modeling and Computing* (full track). See §9 for a candid venue assessment.

---

## Abstract

Physics-informed operator learning — DeepONet, physics-informed DeepONet (PI-DeepONet), and
PINO — is increasingly analysed through subspace diagnostics, with a recurring claim that
training fails because the trunk network's learned basis functions cannot span the target
solution space. We report three findings that jointly redirect this line of inquiry.

**First, a silent numerical pitfall.** The routine that evaluates the input function at query
points is naturally written under a *shared-grid* convention, returning shape
`(n_coef, *x.shape)`. When a batched training loop instead hands it a *per-sample* `(B, N)`
coordinate array, NumPy broadcasting silently produces `(B, B, N)`, so each sample is trained
against another sample's source term or initial condition. We exhibit this in three PDEs.
The failure is invisible to every conventional check: the dataset itself stays correct (the
finite-difference residual of the ground truth is `6e-6`), shapes are legal, and supervised
training succeeds. The only reliable detector is the autograd PDE residual of a *supervised*
model, which was `0.368` — larger than the trivial solution's `0.185`, a contradiction that
exposes the bug. Fixing it changes PI-DeepONet test error from **0.986 → 0.174** on
diffusion–reaction and **0.999 → 0.721** on Burgers. Critically, **the buggy pipeline
produced a clean, monotone, apparently publishable narrative** (Spearman `ρ = −1` between
subspace alignment and accuracy; a dramatic "trunk basis collapse"; spectral priors beating
target-free regularizers) — all of it artefactual.

**Second, a training-free error-attribution diagnostic.** With trunk features
`Φ ∈ R^{N×p}` and true solutions `S ∈ R^{N×n}`, the best relative error attainable by *any*
branch network is exactly `sqrt(1 − captured_energy)`. This quantity is a hard lower bound
computable at any point during training. Decomposing `test_relL2 = LB + opt_gap` separates
*representation* limitations from *optimization* limitations without training to convergence.
Applied after the fix, it shows `LB ≈ 0.19–0.25` while `opt_gap ≈ 0.45–0.49`: representation
is not the binding constraint.

**Third, a mechanism, tested and confirmed.** Direct measurement of two quantities that are
never optimized shows that alignment of the trunk to an analytic spectral prior roughly
halves the loss sharpness (`λ_max(∇²L)/2L`: 295→141, 682→336) and doubles the effective rank
of the output neural tangent kernel (1.7→3.8, 2.0→4.0), while captured energy moves only
`0.947 → 0.951`. Alignment acts as a **regularizer that flattens the landscape and widens the
NTK spectrum**, not as a capacity expansion. Over **ten** (width, viscosity, dimension)
settings with five seeds each, its accuracy gain is **positive in all ten and significant in
eight** (five at `p < 0.001`), and its most robust effect is on *variance*: it clamps the
cross-seed standard deviation into a **problem-independent band of ≈0.006–0.014** (baseline σ
spans 23.6×, aligned σ spans 2.5×). In 2D the baseline does not merely vary — it collapses
deterministically onto `u ≡ 0` (`relL2 = 1.004`, output NTK effective rank 1.1) and is
**representation-bound** (85% of its error is the lower bound of §4), but a random
Fourier-feature trunk — needing no target subspace and no tuning — repairs it better than
alignment does (`1.004 → 0.693`, **30.9%**, versus `0.821` for alignment; p = 1.1e-3), while a
target-free decorrelation regularizer and a 10× initial-condition weight both fail entirely.
We also **retract** an
earlier claim from this project that the gain is linearly predicted by baseline instability:
with more settings and more seeds it collapses (`r = 0.985` → `r = 0.318`, 95% CI
`[−0.199, 0.671]`). Finally, the two standard optimization-side remedies — causal weighting and
L-BFGS fine-tuning — both *degrade* performance here, and we report those negative results.

---

## 1. Introduction

Physics-informed operator learning replaces paired input–output simulation data with the
governing equations themselves. PI-DeepONet [Wang et al., 2021] extends DeepONet
[Lu et al., 2021] by adding a PDE-residual loss, and PINO [Li et al., 2021] does the same for
Fourier neural operators. Because these models are expensive to train and hard to debug, a
substantial literature is devoted to *explaining their failures*: spectral bias
[Wang et al., 2021b], gradient pathologies [Wang et al., 2021a], causality violations
[Wang et al., 2022], loss-landscape stiffness [Krishnapriyan et al., 2021], and — most
directly relevant here — the proposal that the trunk network's learned basis functions fail
to span the target solution space [JCP, 2026].

We began from that last hypothesis, and built the complete pipeline: a diagnostic that
measures how much of the true solution subspace the trunk spans (captured energy, principal
angles, effective rank), a differentiable alignment loss that pushes the trunk toward an
analytic spectral prior or a data-derived subspace, and controlled comparisons. The first
results were excellent. Trunk effective rank rose from 1.5 to 33.7; captured energy from
0.976 to 0.9999; the mean principal angle fell from 67° to 17°. Five methods ranked exactly
as theory would predict, with accuracy monotone in alignment.

All of it was an artefact of a one-line shape bug in our own PDE interface.

We therefore write this paper in two halves. The first half (§3) documents the pitfall
precisely, shows why the usual sanity checks are blind to it, and demonstrates — with a
side-by-side before/after — that it can manufacture a coherent, publishable-looking
scientific story. The second half (§4–§6) reports what we found *after* the fix: an
attribution diagnostic that makes the representation/optimization split quantitative, and a
mechanistic account of subspace alignment as regularization. We regard the first half as the
more immediately useful contribution to the community, and the second as the more
interesting scientific one.

---

## 2. Related work

**Operator learning.** DeepONet [Lu et al., 2021] learns operators via a branch/trunk
factorisation. PI-DeepONet [Wang et al., 2021] trains it without paired data by penalising the
PDE residual. PINO [Li et al., 2021] applies the same idea to FNO. Recent work analyses what
PI-DeepONets actually learn, using NTK and basis-function perspectives to argue that the
trunk basis is poorly matched to the solution space [JCP, 2026], and a systematic study of
PINO training pipelines finds that PINN optimisation pathologies (gradient conflict, causal
violation) recur in PINO and that PINN-style mitigations transfer [arXiv:2606.06164].

**Diagnosing PINN failure.** NTK analyses [Wang et al., 2022b], gradient-pathology
mitigations [Wang et al., 2021a], causality-respecting training [Wang et al., 2022], and
failure-mode characterisations [Krishnapriyan et al., 2021] are the standard toolkit.
Surveys [Zhang et al., 2026; Neurocomputing 2026] converge on the view that PINNs complement
rather than replace classical solvers, with the advantage concentrated in inverse problems
and data assimilation.

**Benchmarks.** PDEBench and RealPDEBench [Hu et al., 2026] have moved the field toward
paired real/simulated evaluation; neither addresses training-loop correctness, which is our
focus.

**Numerical pitfalls in ML code.** Silent broadcasting is a known hazard of NumPy, but its
specific manifestation in operator-learning samplers — where a shared-grid helper is called
with per-sample coordinates — appears not to be documented, and we show that it survives the
checks practitioners actually run.

---

## 3. A silent pitfall in DeepONet-style training loops

### 3.1 The code pattern

An operator-learning problem must evaluate the input function `u` at arbitrary query points:
for a source term in the residual, for initial-condition targets, or for spatially varying
coefficients. A natural helper is

```python
def input_at(self, coef, x):
    """coef: (n, K) spectral coefficients of the input function.
       x   : query points."""
    S = np.sin(np.outer(x.ravel(), self.k))      # shared-grid convention
    C = np.cos(np.outer(x.ravel(), self.k))
    out = coef @ S.T + coef @ C.T                # (n, M)
    return out.reshape(coef.shape[0], *x.shape)
```

This is *correct* whenever `x` describes a single shared grid — sensor locations `(m,)`,
the evaluation grid `(Ng,)`, or a sensor mesh `(m, 2)`. It is *wrong* when `x` is a
per-sample array `(B, N)`, which is exactly what a batched loop hands over: collocation
points resampled independently for each input function.

NumPy does not complain. With `coef.shape[0] == x.shape[0] == B`, the output becomes
`(B, B, N)`. Every downstream expression broadcasts, the loss remains a scalar of plausible
magnitude, and training proceeds — with the residual for sample `b` built from sample `b'`'s
data for `B − 1` of `B` pairings. There is no exception, no warning, and no NaN.

### 3.2 Three instances

| PDE | Quantity affected | Buggy shape | Correct shape |
|---|---|---|---|
| Burgers (periodic) | IC target `u₀(x)` | `(B, B, N_ic)` | `(B, N_ic)` |
| Diffusion–reaction | collocation source `u(x)` | `(B, B, N_col)` | `(B, N_col)` |
| Darcy (2D) | coefficient `a, a_x, a_y` | `(B, B, N_col)` | `(B, N_col)` |

In our codebase the three instances were independent: each PDE class had its own
`input_at`-style helper with the shared-grid convention.

### 3.3 Why the usual sanity checks are blind

We ran the checks a practitioner would run, and all of them passed:

| Check | Result | Why it passes |
|---|---|---|
| Ground-truth residual (finite differences on stored solutions) | `mean(res²) = 6.0e-6` | The **dataset is correct**: reference solvers consume coefficients with per-sample semantics and never call the buggy helper. |
| Supervised training | reaches `relL2 = 0.26` | Data-driven training never calls the helper on per-sample coordinates. |
| Shape assertion | no error | `(B, B, N)` is a legal shape; broadcasting is silent. |
| Loss magnitude | plausible | The mean over `B²` pairings is statistically similar to the correct mean, so the *value* looks fine. |

The single diagnostic that exposed the bug was the **autograd PDE residual of a supervised
model**. After fitting the data to `relL2 = 0.26`, the model's own residual was
`mean(res²) = 0.368`, while the trivial solution `s ≡ 0` has residual `E[u²] = 0.185`.
A model that reproduces the solution cannot have a residual worse than the zero function;
that contradiction was the thread that led to the bug.

### 3.4 Detection recipe

Two lines, both cheap:

```python
# (1) after supervised training to low rel-L2, the autograd PDE residual
#     must be well below the trivial-solution level
assert res_autograd(model) < 0.5 * res_trivial

# (2) assert the sampler's per-sample convention explicitly
assert pts["u0"].shape == (B, n_col)   # never (B, B, n_col)
```

We recommend (1) as a standard health check in any physics-informed training loop, and (2)
inside every sampler that mixes grid-like and per-sample coordinate conventions.

### 3.5 What the bug does to scientific conclusions

This is the part we consider most important. Before the fix, our Burgers ablation
(width 64, 10 000 steps) looked like this:

| method | test relL2 | LB | opt_gap | captured | eff_rank | mean angle |
|---|---|---|---|---|---|---|
| mlp baseline | 0.9992 | 0.5786 | 0.421 | 0.6653 | 1.16 | 74.4° |
| orthogonalization reg. | 0.9966 | 0.4826 | 0.514 | 0.7671 | 5.05 | 68.8° |
| random-subspace alignment | 0.9826 | 0.3366 | 0.646 | 0.8867 | 13.23 | 56.3° |
| spectral-prior alignment | 0.9809 | 0.2960 | 0.685 | 0.9124 | 6.18 | 52.5° |
| Fourier-feature trunk | 0.9493 | 0.2838 | 0.666 | 0.9195 | 4.81 | 50.8° |

Every qualitative conclusion a reader would draw from this table is false:

- accuracy is **monotone** in captured energy (`ρ = −1` across five methods) — artefact;
- spectral priors **beat** a random subspace and a target-free regularizer — artefact;
- a clear **"trunk basis collapse"** (effective rank 1.16) explains the baseline's failure —
  artefact;
- the story "PI-DeepONet fails because its trunk basis is misaligned" appears **supported** —
  it is not.

After fixing the one-line broadcasting bug, with everything else held fixed:

| problem | metric | buggy | fixed |
|---|---|---|---|
| Diffusion–reaction | test relL2 | 0.9857 | **0.1737** |
| Diffusion–reaction | trunk captured energy | 0.9774 | 0.9999 |
| Burgers | test relL2 | 0.9992 | **0.7213** |
| Burgers | trunk captured energy | 0.6653 | 0.9471 |

The "basis collapse" essentially disappears: the plain baseline reaches captured energy
0.947 with no representation-side intervention at all. **Subspace-level conclusions about
operator learning must not be drawn before the physics residual is verified to actually be
minimised.**

![Figure 1 — Effect of fixing the one-line broadcasting bug. Left: test error; right: trunk
captured energy. The buggy pipeline learns nothing on diffusion–reaction and appears
"representation-limited" on Burgers; the fix changes both by 4–6×.](figures/fig3_bug.png)

---

## 4. An error-attribution diagnostic

### 4.1 Setup

Fix a probe point set `Y` (we use a subsample of the evaluation grid), let
`Φ = trunk(Y) ∈ R^{N×p}` be the trunk features and `S ∈ R^{N×n}` the matrix of true solutions
(one column per input function). A DeepONet with this trunk and *any* branch network
realises the column space `{Φ B : B ∈ R^{p×n}}`.

### 4.2 A hard lower bound

**Proposition 1.** For any branch network `b`, the relative `L₂` error of the resulting
operator on `S` satisfies

```
              ||Φ B − S||_F
   min_B  ───────────────────  =  ||(I − P_Φ) S||_F / ||S||_F  =  sqrt(1 − captured_energy),
                 ||S||_F
```

where `P_Φ` is the orthogonal projector onto `span(Φ)` and
`captured_energy = ||P_Φ S||_F² / ||S||_F²`.

*Proof.* The minimiser is `B = Φ⁺S`, giving the projection residual; `||P_Φ S||_F²/||S||_F²` is
exactly the (unweighted) captured energy. When target directions are weighted by their
singular values `σ_i`, the same computation with `w_i = σ_i²/Σσ_j²` yields the
energy-weighted version used throughout. ∎

The bound is **computable at any point during training** (one SVD), requires no converged
model, and holds regardless of branch capacity. It is therefore a *representation* limit that
can be read off directly.

### 4.3 Attribution

Define `LB := sqrt(1 − captured_energy)` and `opt_gap := test_relL2 − LB`. Then

- large `LB` ⇒ **representation-limited**: the trunk cannot express the solution family;
- large `opt_gap` ⇒ **optimization-limited**: the trunk could express it, but training did not
  find it.

This is a one-line, training-free decision rule, and it tells you *in advance* whether a
representation-side intervention is worth attempting.

### 4.4 What it reveals here

Table 1 (§5.3): after the fix, all representation-side methods land at
`captured ∈ [0.94, 0.96]`, i.e. `LB ∈ [0.19, 0.25]`, while `opt_gap ∈ [0.45, 0.49]`
dominates by a factor of two. The diagnostic correctly predicts that representation-side work
has little headroom, and it is consistent with the negative results of §5.5.

### 4.5 A three-way split: representation / optimization / generalization

The two-way split of §4.3 conflates optimization with generalization: the model is trained on
200 input functions and tested on 50 unseen ones, so part of `opt_gap` could be failure to
generalize across the function space rather than failure to optimize. We therefore also
evaluate on the *training* functions and report

```
test_relL2 = LB + opt_gap_train + gen_gap
    LB            = sqrt(1 − captured_energy)   representation
    opt_gap_train = train_relL2 − LB            optimization / branch capacity
    gen_gap       = test_relL2 − train_relL2    generalization over input functions
```

| setting | n | LB | train | test | opt_gap_train | gen_gap |
|---|---|---|---|---|---|---|
| Burgers ν=0.02, w64 | 4 | 0.22–0.30 | 0.742 | 0.740 | 0.47 | −0.002 |
| Burgers ν=0.005, w64 | 6 | 0.32–0.39 | 0.793 | 0.793 | 0.45 | 0.000 |
| 2D Burgers w32, baseline | 5 | 0.849 | 1.003 | 1.003 | 0.154 | 0.000 |
| 2D Burgers w32, aligned | 5 | 0.654 | 0.811 | 0.821 | 0.157 | 0.010 |

**`gen_gap` is ≈0 everywhere — below 1% of total error in every cell.** This must be read
carefully: it does not mean generalization is *good*. Train error is as large as test error.
It means the model does not fit even the 200 input functions it was trained on, so the
residual error is genuinely optimization- or representation-bound rather than a
generalization failure. The two-way split of §4.3 is therefore safe for these problems, and
we use the three-way version to separate the two regimes: **1D is optimization-bound**
(`LB` 0.22–0.39, `opt_gap_train` 0.45–0.52); **2D is representation-bound** (85% of the
baseline's error is `LB`).

*Caveat.* `LB` is computed on a subsampled probe set with validation solutions, whereas
`relL2` uses the full grid and the split's own solutions (§7); `gen_gap` is the only one of
the three computed entirely within one metric on one model.

![Figure 2 — Three-way error attribution. 1D cells are optimization-bound (orange dominates);
the 2D baseline is representation-bound (blue dominates, 85% of its error). `gen_gap` is
negligible everywhere.](figures/fig1_attribution.png)

---

## 5. Experiments

### 5.1 Setup

Burgers `u_t + u u_x = ν u_xx` on `x ∈ [0,1)` periodic, `t ∈ [0,1]`, solved with a
Strang-split spectral method (Kassam–Trefethen-style exponential treatment of diffusion, RK4
convection); `Nx = 256` for `ν = 0.02` and `Nx = 1024` for `ν = 0.005`. Also a 1D
diffusion–reaction system (`D = k = 0.01`) and 2D Darcy flow. Input functions are truncated
spectral expansions, so `u₀(x)` and `a(x,y)` are analytic and available at any query point.

Branch/trunk MLPs, 3 hidden layers, trunk width 64 unless stated, 64 collocation points ×
batch 16 input functions per step, Adam `lr = 1e-3`, 10 000 steps (6 000 for the phase
diagram), 200/50/50 train/val/test input functions. All reported numbers are test relative
`L₂`, averaged over input functions. Statistical tests are Welch's `t`-test over seeds.

### 5.2 Effect of the pitfall

See §3.5. For diffusion–reaction, fixing the bug moves test error from 0.986 to 0.174 — the
difference between "no learning" and "learning".

### 5.3 Representation-side interventions (Burgers, width 64, 10 000 steps)

| method | test relL2 | LB | opt_gap | captured | eff_rank | mean angle |
|---|---|---|---|---|---|---|
| Fourier-feature trunk | **0.6651** | 0.1891 | 0.476 | 0.9642 | 8.00 | 39.4° |
| spectral-prior alignment | 0.6672 | 0.2136 | 0.454 | 0.9544 | 10.46 | 43.2° |
| Fourier + spectral alignment | 0.6680 | 0.2084 | 0.460 | 0.9566 | 10.04 | 43.7° |
| random-subspace alignment | 0.6988 | 0.2454 | 0.453 | 0.9398 | 11.59 | 45.1° |
| orthogonalization regulariser | 0.7114 | 0.2347 | 0.477 | 0.9449 | 14.04 | 46.1° |
| mlp baseline | 0.7213 | 0.2301 | 0.491 | 0.9471 | 4.47 | 45.6° |

After the fix:
1. spectral-prior alignment beats the baseline by 7.5% and beats both a random subspace and a
   target-free orthogonalization regulariser — **prior knowledge does help**;
2. but a **random Fourier-feature trunk matches it at zero cost** and is not complementary
   with alignment (0.6680 vs 0.6651);
3. `LB` is already small for everyone (0.19–0.25); `opt_gap` (0.45–0.49) dominates.

![Figure 3 — Representation-side interventions at width 64, 10 000 steps (single seed).
Every intervention lowers the error, but the ordering is baseline > ortho > random
alignment > prior alignment ≈ Fourier trunk, and the Fourier trunk needs neither a target
subspace nor a hyperparameter.](figures/fig6_1d_ablation.png)

### 5.4 The bottleneck is optimization

Guided by `opt_gap`, we tested the two standard optimization-side remedies.

**(a) Causal weighting** of the residual over time slabs [Wang et al., 2022]:

| ε | off | 1 | 10 | 100 |
|---|---|---|---|---|
| test relL2 | **0.7213** | 0.7601 | 0.8045 | 0.8285 |

Monotonically worse: with a soft IC constraint of equal weight, concentrating weight at small
`t` starves the later-time fit.

**(b) L-BFGS fine-tuning** after Adam, standard in single-instance PINN solvers:

| method | fixed training set | after Adam | after L-BFGS | opt_gap | trunk eff. rank |
|---|---|---|---|---|---|
| `pi_lbfgs` | 16 fn × 64 pts | 0.7213 | **2.0036** | — | — |
| `pi_lbfgs` | 64 fn × 32 pts | 0.7213 | **0.9119** | 0.491 → 0.688 | 4.47 → 1.40 |
| `pi_align_lbfgs` | 64 fn × 32 pts | 0.6672 | **0.7219** | 0.454 → 0.516 | 10.46 → 18.89 |

L-BFGS requires a deterministic objective and hence a fixed training set; at the scale needed
to represent an operator family (hundreds of input functions × thousands of coordinates) that
set is far too small, and the second-order method overfits it. Enlarging the set mitigates
(2.00 → 0.91) but does not fix it. **The "Adam + L-BFGS" recipe from single-instance PDE
solving does not transfer to operator learning.**

Note the incidental robustness signal: L-BFGS collapses the *baseline* trunk
(effective rank 4.47 → 1.40) while the aligned trunk becomes *more* diverse (10.46 → 18.89).

### 5.5 Mechanism: alignment regularizes the optimization problem

Three observations were unexplained by capacity: a modest mean-accuracy gain, a 6.7× variance
reduction (§5.6), and robustness to an aggressive optimizer (§5.4b). We therefore measured
two quantities that are **never optimized** and are independent of the diagnostic used to
build the alignment loss: the output neural tangent kernel `K = J Jᵀ` (per-point Jacobian of
the model output w.r.t. all parameters) and the loss sharpness `λ_max(∇²L)/2L`, estimated by
power iteration with Hessian-vector products.

Paired comparison at width 64 (same seed, data and optimizer; the only difference is the
alignment term):

| checkpoint | trunk eff. rank | **out-NTK eff. rank** | **NTK top-eig. share** | **λ_max** | **sharpness** |
|---|---|---|---|---|---|
| baseline, seed 1 | 4.1 | 1.7 | 0.874 | 1035 | 295 |
| **aligned**, seed 1 | **9.0** | **3.8** | **0.542** | **430** | **141** |
| baseline, seed 2 | 4.2 | 2.0 | 0.803 | 2020 | 682 |
| **aligned**, seed 2 | **10.6** | **4.0** | **0.432** | **833** | **336** |
| (ref.) baseline, width 32 | 2.7 | 1.5 | 0.917 | 959 | 437 |

Alignment roughly **doubles the effective rank of the output NTK** and cuts the dominant
eigenvalue's share from ≈0.85 to ≈0.49; it **halves the sharpness**, with `λ_max` falling from
959–2020 (all three baselines ≥ 959) to 430–833 (both aligned models < 959) — a complete
separation between conditions. Meanwhile captured energy moves only `0.947 → 0.951`.

> Alignment acts as a **regularizer that flattens the loss landscape and widens the NTK
> spectrum**, not as a capacity expansion.

![Figure 4 — Mechanism. Two quantities that are never optimized: output-NTK effective rank
doubles, top-eigenvalue share and loss sharpness drop by ~2×. Black dots are individual
checkpoints.](figures/fig5_mechanism.png)

### 5.6 Alignment clamps run-to-run variance to a problem-independent floor

To test the mechanism across regimes we varied trunk width and, independently, the
solution-space rank. Decreasing the viscosity from `ν = 0.02` to `ν = 0.005` lowers the initial
captured energy from 0.637 to 0.571, i.e. it makes the solution space **higher-rank and
harder**. All settings below use **five seeds** per cell (6000 steps).

| setting | baseline (mean ± **std**) | aligned (mean ± std) | **variance ratio** | gain | Welch p |
|---|---|---|---|---|---|
| **2D, w32** | 1.0039 ± 0.0024 | 0.8205 ± 0.0421 | — | **18.27%** | 0.0006 |
| ν=0.02, w32 | 0.8047 ± **0.0780** | 0.6815 ± 0.0102 | **7.6×** | 15.31% | 0.0235 |
| ν=0.01, w32 | 0.8353 ± **0.0729** | 0.7420 ± 0.0080 | **9.1×** | 11.18% | 0.0453 |
| ν=0.02, w64 | 0.7627 ± 0.0122 | 0.6823 ± 0.0118 | 1.03× | 10.55% | <0.0001 |
| ν=0.005, w32 | 0.8435 ± **0.0731** | 0.7683 ± 0.0056 | **13.1×** | 8.92% | 0.0826 |
| ν=0.02, w24 | 0.7500 ± 0.0144 | 0.6874 ± 0.0081 | 1.78× | 8.35% | 0.0001 |
| ν=0.01, w64 | 0.7999 ± 0.0033 | 0.7465 ± 0.0100 | 0.33× | 6.67% | 0.0001 |
| ν=0.005, w64 | 0.8117 ± 0.0089 | 0.7691 ± 0.0091 | 0.98× | 5.25% | 0.0001 |
| ν=0.02, w16 | 0.7370 ± 0.0145 | 0.7020 ± 0.0108 | 1.34× | 4.74% | 0.0030 |
| ν=0.005, w16 | 0.7974 ± 0.0128 | 0.7797 ± 0.0138 | 0.93× | 2.22% | 0.0695 |

Ten settings, five seeds each. **The gain is positive in 10/10 and significant at `p < 0.05` in
8/10** (five of them at `p < 0.001`). Only the two ν=0.005 cells at w16/w32 fail to reach
significance (`p = 0.070`, `0.083`), consistent with their baselines already being stable.

**The robust effect is on the variance, not on a fitted slope.** Across the nine 1D settings the
*baseline* cross-seed standard deviation spans a **23.6×** range (0.0033–0.0780), whereas the
*aligned* standard deviation spans only a **2.5×** range (0.0056–0.0138). Alignment does not
reduce variance by a fixed proportion; it **clamps it into a problem-independent band of
≈0.006–0.014**, from both sides. Where the baseline sits far above the band (the three w32 cells,
σ ≈ 0.073–0.078) this is a 7.6–13.1× reduction; where the baseline already lies inside it
(ν=0.02 w64, ν=0.005 w64/w16) the ratio is ≈1; and where the baseline is *below* it (ν=0.01 w64,
σ = 0.0033) alignment slightly *increases* variance (0.33×) while still improving the mean by
6.67%. This two-sided clamping is a sharper and better-fitting description than any monotone
law, and it holds in every 1D cell. It is not an artifact of numerical nondeterminism: §5.9
shows that re-running a fixed configuration is bit-identical (`σ_repeat = 0`).

**A corrected negative result.** In an earlier iteration of this project we reported that the
accuracy gain was almost perfectly predicted by the baseline's cross-seed standard deviation
(`r = 0.985` over four settings with three seeds each). That claim does not survive better
data. With **ten** settings and five seeds each, the same quantity gives **`r = 0.318`, 95%
bootstrap CI `[−0.199, 0.671]`** — indistinguishable from zero, and stable at that value as `n`
grew from 8 to 10. The earlier value was inflated
by two known shortcomings: the regressor (`σ`) was itself estimated from only three seeds
(≈40% relative noise), and `n = 4` with one contaminated grouping. We report this correction
explicitly because it changes the paper's central quantitative claim, and because it is the
clearest illustration of the general lesson of §3: a plausible-looking quantitative regularity
can be an artefact of small samples and a noisy regressor.

**What survives.** The sign of the gain is positive in **all ten** settings; it is significant
at `p < 0.05` in **eight of ten** (five at `p < 0.001`); the two-sided variance clamp into
≈0.006–0.014 holds in every 1D cell; and in 2D the baseline does not merely vary — it collapses
deterministically, giving the largest and most significant gain in the study (§5.7). We make no
claim about a quantitative instability–gain law. The defensible claims are the sign of the gain,
its significance, the variance clamp, and the finding that 2D is representation-bound (§5.7).

![Figure 5 — The variance clamp. (a) Aligned cross-seed sd vs baseline sd (log–log); the
green band is the problem-independent floor ≈0.006–0.014. (b) Accuracy gain against baseline
sd — plotted to show that the retracted linear law (`r = 0.985`) does not survive more data
(`r = 0.318`).](figures/fig4_varfloor.png)

### 5.7 2D: the bottleneck is representation, but alignment is not the fix

In 2D the baseline does not partially fail — it collapses completely and reproducibly
(`relL2 = 1.0039 ± 0.0024`, captured ≈ 0.27, out-NTK effective rank 1.08–1.10 over five
seeds). Its output NTK is numerically rank-1, the top eigenvalue carrying 98.5–98.9% of the
spectrum: the model has collapsed onto `u ≡ 0`, an exact solution of the PDE and the periodic
boundary conditions, leaving only the initial condition to supply gradient. It is not
*unstable*, it is *degenerate* (σ = 0.0006).

Per-seed diagnostics (first three baseline seeds):

| 2D Burgers, width 32 | test relL2 | captured | trunk eff. rank | out-NTK eff. rank | NTK top-eig. share |
|---|---|---|---|---|---|
| baseline, seed 0 | 1.0033 | 0.286 | 1.55 | **1.10** | **0.985** |
| baseline, seed 1 | 1.0027 | 0.248 | 1.34 | 1.08 | 0.989 |
| baseline, seed 2 | 1.0039 | 0.274 | 1.38 | 1.09 | 0.987 |

§4.5 establishes what kind of failure this is: `LB = 0.849` accounts for **85%** of the
baseline's error, while `opt_gap_train` is only 0.154 and `gen_gap ≈ 0`. This is the
representation-limited regime. We then asked what actually repairs it, using four controls
that isolate the mechanism (width 32, 6000 steps, five seeds each).

| intervention | test relL2 | gain | Welch p | captured | needs a target subspace? | cost |
|---|---|---|---|---|---|---|
| MLP baseline | 1.0039 ± 0.0024 | — | — | 0.28 | — | — |
| orthogonalization regularizer (target-free) | 1.0016 ± 0.0010 | 0.22% | 0.112 | 0.30 | no | 1 hyperparameter |
| 10× initial-condition weight | 1.0029 ± 0.0013 | 0.10% | 0.464 | 0.28 | no | 1 hyperparameter |
| alignment to a **random** smooth subspace | 0.7910 ± 0.0165 | 21.21% | 6.0e-6 | 0.60 | yes | subspace + weight |
| alignment to the **analytic spectral prior** | 0.8205 ± 0.0421 | 18.27% | 6.1e-4 | 0.61 | yes | subspace + weight |
| **random Fourier-feature trunk** | **0.6933 ± 0.0199** | **30.94%** | 3.1e-6 | **0.77** | **no** | **none** |

![Figure 6 — 2D controls. Red: interventions that fail (baseline, target-free
decorrelation, 10× IC weight) sit on the trivial-solution line. Purple: alignment to *any*
high-dimensional smooth subspace works. Green: a random Fourier-feature trunk works best, at
zero cost. Brown: the fully supervised upper bound (6000 steps) is *worse* than the
physics-informed model with a Fourier trunk.](figures/fig2_2d_controls.png)

Four things follow.

1. **The failure is representational and it is repairable — but not specifically by
   alignment.** A random Fourier-feature trunk, needing no target subspace, no prior and no
   hyperparameter, reaches **0.693**: **15.5% better than spectral-prior alignment
   (p = 1.1e-3)**, and the best result anywhere in this study.

2. **Diversity alone is not sufficient, and this is the sharpest mechanistic result we
   have.** The target-free orthogonalization regularizer, which only asks that
   `ΦᵀΦ ∝ I`, is completely ineffective in 2D (1.0016, p = 0.112) even though it produced a
   small but real gain in 1D (§5.3). What matters is therefore not *how many* directions the
   trunk spans but *what kind* of function subspace it spans: the 2D solution family needs 58
   modes and a randomly-initialised trunk captures only 0.295 of it, so an arbitrary
   high-dimensional subspace simply fails to intersect it.

3. **The specific prior does not matter.** Alignment to a random smooth subspace (0.7910) is
   at least as good as alignment to the analytic spectral prior (0.8205), with nearly
   identical captured energy (0.60 vs 0.61). The benefit comes from being pushed toward
   *some* high-dimensional smooth subspace, not toward the correct one in particular.

4. **It is not a soft-constraint artifact.** Raising the initial-condition weight 10× leaves
   the collapse untouched (1.0029, p = 0.464), ruling out the explanation that the baseline
   merely under-weights the one term able to supply gradient.

Taken together with §5.3, the same conclusion holds in both dimensions: **what rescues
PI-DeepONet in these regimes is a spectrally structured trunk basis, and random Fourier
features supply it most cheaply.** Alignment is one way to buy that structure; it is not
necessary, and in 2D it is not even the best way. We therefore retract the "alignment is
necessary in 2D" reading of this experiment; what survives is the finding that 2D is
representation-bound (§4.5) — a regime in which, unlike 1D, representation-side interventions
have large headroom.

### 5.8 Reference points: how good can this architecture get?

Two questions are left open by §4.5. (a) Is `opt_gap_train ≈ 0.45` in 1D a property of
*physics-informed* training, or would any training of this architecture stop there? (b) Is
0.693 in 2D close to what the architecture can do, or nowhere near? Both are answered by a
fully supervised control: the same network trained on ground-truth solution values (200
functions, 256 random grid points per function per step, no physics).

| setting | supervised 6k | supervised 30k | supervised 50k | physics-informed |
|---|---|---|---|---|
| 1D ν=0.02, w64 | 0.582 ± 0.021 | 0.486 ± 0.022 | **0.472 ± 0.011** | 0.7627 @6k → **0.6139 @50k** |
| 2D w32 | 0.868 ± 0.008 | 0.638 ± 0.007 | **0.618 ± 0.011** | 1.0039 @6k → **0.5983 @50k** (Fourier trunk) |

*(three seeds each; the physics 50k entries are one seed)*

1. **The supervised number is the missing scale reference.** With full supervision the *same*
   trunk reaches **0.472** in 1D and **0.618** in 2D, so neither the 1D `opt_gap_train ≈ 0.45`
   nor the 2D `LB = 0.85` is a capacity limit of the architecture — the network can represent
   and fit both solution families. The 1D gap is attributable to optimizing the physics
   residual; the 2D gap to the trunk's basis.
2. **A caution about budget-matched baselines.** At 6000 steps the supervised 2D model scores
   0.868 — *worse* than physics-informed training with a Fourier trunk. Had we stopped there
   we would have concluded "physics beats data", which is simply an under-training artifact:
   supervision needs 5× more steps to reach 0.638. Any "physics-informed beats supervised"
   claim in this literature should be required to show the supervised curve past its elbow.
3. **At a matched budget the two are comparable, and the ordering is problem-dependent.**
   At 50 000 steps, 2D gives **0.598** with physics (no solution data, 1 seed) against
   **0.618 ± 0.011** with full supervision (3 seeds) — a difference well inside the seed-to-seed
   spread, so we claim *parity*, not superiority. In 1D the ordering reverses and clearly
   favours supervision (0.614 vs 0.472), i.e. physics-informed training is markedly less
   compute-efficient there. Both "physics beats data" and "data beats physics" need the budget
   on the x-axis before they mean anything.
4. **The physics residual is a strong regularizer.** The supervised models show a real
   generalization gap that the physics-informed models do not — in 2D at 50k,
   `train = 0.454` vs `test = 0.629` (`gen_gap ≈ 0.175`) for supervision, against
   `gen_gap ≈ 0.01` throughout §4.5 for physics-informed training. This is a concrete
   instance of the usual argument for physics-informed learning, and it is orthogonal to the
   accuracy question.

![Figure 7 — Budget-matched comparison of physics-informed and fully supervised training
(same architecture, same data budget for the supervised runs). Left: in 1D supervision is
consistently ahead. Right: in 2D the curves cross — the physics-informed model with a Fourier
trunk starts ahead and stays ahead, using no solution data at all.](figures/fig7_budget.png)

**Training length: the gap is partly a step budget.** Increasing the step budget keeps
improving both regimes (single seed, seed 0):

| steps | 2D Fourier trunk (physics) | 1D baseline (physics) |
|---|---|---|
| 6 000 | 0.6933 (captured 0.77) | 0.7627 (captured 0.94, 5 seeds) |
| 20 000 | 0.6427 (captured 0.93) | 0.6736 (captured 0.95) |
| 50 000 | **0.5983** (captured 0.94) | **0.6139** (captured 0.96) |

So "optimization-limited" (1D) and "representation-limited" (2D) should both be read as
statements *at a fixed budget*: with 20–50k steps the physics-informed models keep improving
and their captured energy rises to 0.93–0.96. What the diagnostic buys is not the verdict
"this can never be fixed" but the verdict "**this is where to spend the next unit of effort**
— on the optimizer in 1D, on the trunk basis in 2D".

### 5.9 Is the variance floor numerical noise? No.

The floor of §5.6 (σ_aligned ≈ 0.006–0.014) is suspiciously close to the `Δ = 0.0124`
discrepancy we found between two historical runs of the same configuration, so we tested it
directly: we re-ran one configuration — 1D Burgers ν=0.02, w64, baseline, seed 0 — five times
into separate directories, and likewise the 2D baseline seed 0 five times.

```
1D (5 repeats): 0.7623 0.7623 0.7623 0.7623 0.7623     σ_repeat = 0.000000
2D (5 repeats): 1.0033 1.0033 1.0033 1.0033 1.0033     σ_repeat = 0.000000
```

Re-running is **bit-identical**: our pipeline is deterministic, so the floor is genuine
seed-to-seed variation and the §5.6 claim survives. The historical `Δ = 0.0124` pair is
therefore not nondeterminism — the two files were produced four days apart (2026-09-17 and
2026-09-21) by *different versions of the code*, with no record of which version produced
which. That is a different and more common reproducibility failure than numerical noise, and
it is the reason we recommend recording a code hash alongside every run (§7).

---

## 6. Discussion

### 6.1 What we can and cannot claim

**Supported.**
- A specific, reproducible broadcasting pitfall in operator-learning samplers, invisible to
  the standard checks, with a two-line detection recipe (§3).
- The pitfall can manufacture a coherent, monotone, publishable-looking conclusion set (§3.5)
  — a warning about the evidential standard of subspace-level claims in this literature.
- `LB = sqrt(1 − captured_energy)` is a hard, training-free lower bound, and
  `opt_gap = test_relL2 − LB` cleanly separates representation from optimization limits (§4).
  A three-way split shows `gen_gap ≈ 0`, so neither split is concealing a generalization
  failure (§4.5).
- Subspace alignment in this setting behaves as a **regularizer**: it halves loss sharpness and
  doubles NTK effective rank (§5.5). Across **ten** settings with five seeds each, the gain is
  **positive in all ten and significant in eight** (§5.6).
- Alignment drives the run-to-run standard deviation to a **problem-independent floor of ≈0.01**
  (baseline σ spans 8.8×, aligned σ spans 2.5×; 7.6–13.1× reductions where the baseline sits
  above the floor) (§5.6). This is the most robust effect we measure.
- In 2D the baseline collapses deterministically onto `u ≡ 0` and is **representation-bound**
  (85% of its error is `LB`), and representation-side interventions have large headroom
  (`1.003 → 0.693` with a Fourier-feature trunk) — but a target-free decorrelation
  regularizer and a 10× initial-condition weight fail entirely, and the analytic prior is no
  better than a random smooth subspace (§5.7).
- `gen_gap ≈ 0` in every cell: the model does not fit even its own 200 training functions, so
  the residual error is optimization- or representation-bound, not a generalization failure
  (§4.5).
- Causal weighting and L-BFGS fine-tuning both fail here (§5.4).

**Not supported.**
- "Trunk basis collapse is the root cause of PI-DeepONet failure." After the fix the plain
  baseline reaches captured energy 0.947 unaided in 1D. (It *is* the correct description in
  2D, where the baseline's output NTK is numerically rank-1.)
- "Alignment is necessary in 2D." We reported this in an earlier iteration. With four
  controls it fails: a random Fourier-feature trunk does better at zero cost,
  a target-free decorrelation regularizer and a 10× initial-condition weight do nothing,
  and the analytic prior is no better than a random smooth subspace. **Retracted** (§5.7).
  What survives is that 2D is representation-bound.
- "The accuracy gain is linearly predicted by baseline instability." We reported this in an
  earlier iteration (`r = 0.985`); with **ten** settings and five seeds it collapses to
  `r = 0.318`, 95% CI `[−0.199, 0.671]`. **Retracted** (§5.6).
- "Alignment substantially improves accuracy." The mean gain is 2.2–16% depending on setting,
  and in 1D it is matched for free by a random Fourier-feature trunk.
- "Alignment and Fourier features are complementary." They are not.

### 6.2 Implications for practice

1. Add the autograd-residual health check to any physics-informed training loop (§3.4).
2. Before attempting representation-side fixes, compute `LB`. If it is already low, the work
   belongs on the optimization side.
3. If you do want the trunk to span more of the solution space, changing the trunk to random
   Fourier features is simpler than adding an alignment loss and performs comparably.

### 6.3 Implications for the literature

Papers that infer representational conclusions from subspace diagnostics of a physics-informed
operator learner should report (a) the autograd residual of a supervised baseline on the same
architecture and data, and (b) explicit shape assertions on sampler outputs. Both are cheap;
neither is currently standard. Where a claimed intervention is not compared against a
target-free regulariser and an architectural alternative (e.g. Fourier features), the effect
attribution is incomplete.

---

## 7. Limitations

- **Problem suite.** Quantitative comparisons are on 1D Burgers (three viscosities), one 2D
  Burgers problem, and a 1D diffusion–reaction system for the pitfall demonstration. There is
  still only **one** 2D problem, at a single width, and no 3D or real-world application. The
  variance-floor effect is measured on seven completed 1D cells; whether the floor value (≈0.01)
  is universal or specific to this problem family is unknown. This is the single largest gap.
- **Low-accuracy regime.** Absolute errors remain 0.68–0.84 in 1D and 0.81–0.88 in 2D; all
  methods sit far from `LB`. Conclusions may differ in the high-accuracy regime.
- **The variance floor is an empirical regularity, not a mechanism.** We observe
  `σ_aligned ≈ 0.01` across seven cells but have no account of *why* that value, nor of how it
  depends on optimizer, batch size, or architecture. Until it is replicated across those axes,
  it should be treated as descriptive.
- **Five seeds.** Adequate for the reported effect sizes; but the width-32 baselines show
  `σ ≈ 0.073–0.078`, so those cells remain the least precisely estimated.
- **A retracted claim is part of the record.** We previously reported `r = 0.985` for a linear
  instability–gain relation; it is retracted in §5.6. Readers of any earlier draft of this work
  should defer to the current version.
- **The `LB` argument** assumes the target solutions and the trunk are evaluated on the same
  point set, and that a linear read-out with per-sample coefficients is the relevant model
  class. For architectures with architectural constraints coupling coefficients across
  inputs, `LB` remains valid as a bound but may be loose.
- **NTK scope.** We use the *output* NTK, not the residual-operator NTK that strictly governs
  PINN training dynamics. The former is standard and cheaper; a residual-NTK analysis would be
  more faithful.
- **Genesis.** The central pitfall is one we introduced ourselves. We report it because the
  failure mode is generic to the code pattern and because its effect on scientific conclusions
  was so strong, but readers should weight the generalisability of the pattern (which we argue
  is high) rather than the fact that it was ours.

---

## 8. Conclusion

We set out to test whether PI-DeepONet fails because its trunk basis cannot span the solution
space, obtained a clean and convincing affirmative answer, and discovered that the answer was
produced by a one-line NumPy broadcasting bug that survives every conventional sanity check —
including the finite-difference residual of the ground truth, which remains `6e-6` while
training is entirely broken. After the fix, the representation-side story collapses:
`LB` is already small, the trunk is fine, and the residual error is optimization-bound. What
survives is an attribution diagnostic that makes the representation/optimization split
quantitative without training to convergence, and a mechanism — alignment regularizes rather
than expands — confirmed by two never-optimized geometric quantities (NTK effective rank
doubling, loss sharpness halving), by a problem-independent variance floor (≈0.01), by a
three-way attribution showing `gen_gap ≈ 0`, and by the
2D case where representation, not optimization, is binding (`relL2 = 1.003`, NTK rank 1.1,
85% of the error explained by `LB`) — and where the cheapest fix is a trunk architecture
rather than an alignment loss (`0.693`). Along the way we also retracted one of our own quantitative claims: a
linear law relating baseline instability to the alignment gain, which held at `r = 0.985` on
four settings with three seeds and collapsed to `r = 0.318` (95% CI spanning zero) on ten
settings with five. That retraction is, we think, the most useful methodological object in
this paper: it is the same failure mode as the broadcasting bug — a convincing regularity
produced by too little data and a noisy regressor — caught only because we went back and
measured it properly. We hope the detection recipe, the attribution diagnostic, and the
retraction are useful beyond this paper's own claims.

---

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks: A deep
   learning framework for solving forward and inverse problems involving nonlinear partial
   differential equations. *J. Comput. Phys.* 378:686–707, 2019.
2. L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. Learning nonlinear operators via
   DeepONet based on the universal approximation theorem of operators.
   *Nature Machine Intelligence* 3(3):218–229, 2021.
3. S. Wang, H. Wang, P. Perdikaris. Learning the solution operator of parametric partial
   differential equations with physics-informed DeepONets. *Science Advances* 7(40):eabi8605,
   2021.
4. Z. Li, N. Kovachki, K. Azizzadenesheli, et al. Physics-informed neural operator for
   learning partial differential equations. arXiv:2111.03794; *ACM Trans. Math. Softw.*, 2024.
5. What do physics-informed DeepONets learn? Understanding and improving training.
   *J. Comput. Phys.*, 2026. (S0021999126002019; arXiv:2411.18459)
6. N. Chen, C. Cui, A. Chen, S. Wang, R. Ma. On the training of physics-informed neural
   operators for solving parametric partial differential equations. arXiv:2606.06164, 2026.
7. S. Wang, S. Sankaran, P. Perdikaris. Respecting causality when training physics-informed
   neural networks. arXiv:2203.07404, 2022.
8. S. Wang, X. Yu, P. Perdikaris. When and why PINNs fail to train: A neural tangent kernel
   perspective. *J. Comput. Phys.* 449:110768, 2022.
9. S. Wang, Y. Teng, P. Perdikaris. Understanding and mitigating gradient pathologies in
   physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5):A3055–A3081, 2021.
10. A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney. Characterizing possible
    failure modes in physics-informed neural networks. *NeurIPS* 2021.
11. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. DeepXDE: A deep learning library for solving
    differential equations. *SIAM Review* 63(1):208–228, 2021.
12. A. Jacot, F. Gabriel, C. Hongler. Neural tangent kernel: Convergence and generalization in
    neural networks. *NeurIPS* 2018.
13. N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang. On large-batch
    training for deep learning: Generalization gap and sharp minima. *ICLR* 2017.
14. M. Tancik, P. P. Srinivasan, B. Mildenhall, et al. Fourier features let networks learn
    high frequency functions in low dimensional domains. *NeurIPS* 2020.
15. S. Wang, H. Wang, P. Perdikaris. On the eigenvector bias of Fourier feature networks.
    *Comput. Methods Appl. Mech. Engrg.* 384:113938, 2021.
16. W. Zhang, W. Suo, J. Song, W. Cao. Physics-informed neural networks (PINNs) as intelligent
    computing technique for solving partial differential equations: Limitation and future
    prospects. *Sci. China Phys. Mech. Astron.* 69:214602, 2026.
17. P. Hu, H. Feng, H. Liu, et al. RealPDEBench: A benchmark for complex physical systems with
    real-world data. *ICLR* 2026 (Oral); arXiv:2601.01829.
18. N. Chen, S. Wang, R. Ma, A. Chen, C. Cui. Enforcing hidden physics in physics-informed
    neural networks. *Communications Physics*, 2026 (s42005-026-02743-z).
19. T. De Ryck, S. Mishra. Numerical analysis of physics-informed neural networks and related
    models — a review. 2023.
20. E. Cardoso-Bihlo, A. Bihlo. Exactly conservative physics-informed neural networks and deep
    operator networks for dynamical systems. *Neural Networks* 181:106826, 2025.

*TODO: fill in DOIs/volumes, verify refs 5 and 19.*

---

## Appendix A. Full phase-diagram data (5 seeds per cell; authoritative version)

Matches §5.6 exactly. Regenerate with `python scripts/analyze_phase.py --verbose`
(per-seed raw values are also stored in every run's JSON under `results/<exp>/*.json`).

| setting | baseline (mean ± std) | aligned (mean ± std) | gain | Welch p |
|---|---|---|---|---|
| **2D, w32** | 1.0039 ± 0.0024 | 0.8205 ± 0.0421 | 18.27% | 0.0006 |
| ν=0.02, w32 | 0.8047 ± 0.0780 | 0.6815 ± 0.0102 | 15.31% | 0.0235 |
| ν=0.01, w32 | 0.8353 ± 0.0729 | 0.7420 ± 0.0080 | 11.18% | 0.0453 |
| ν=0.02, w64 | 0.7627 ± 0.0122 | 0.6823 ± 0.0118 | 10.55% | <0.0001 |
| ν=0.005, w32 | 0.8435 ± 0.0731 | 0.7683 ± 0.0056 | 8.92% | 0.0826 |
| ν=0.02, w24 | 0.7500 ± 0.0144 | 0.6874 ± 0.0081 | 8.35% | 0.0001 |
| ν=0.01, w64 | 0.7999 ± 0.0033 | 0.7465 ± 0.0100 | 6.67% | 0.0001 |
| ν=0.005, w64 | 0.8117 ± 0.0089 | 0.7691 ± 0.0091 | 5.25% | 0.0001 |
| ν=0.02, w16 | 0.7370 ± 0.0145 | 0.7020 ± 0.0108 | 4.74% | 0.0030 |
| ν=0.005, w16 | 0.7974 ± 0.0128 | 0.7797 ± 0.0138 | 2.22% | 0.0695 |

With Holm–Bonferroni correction across the ten tests, six settings remain significant at
α = 0.05 (the two w32 cells at p = 0.024 and 0.045 do not survive); an aggregate sign test
over the ten settings gives p ≈ 0.002 (10/10 positive).

## Appendix B. Reproduction

Environment: `conda activate pytorch` (torch 2.14+cu130, sm_120), `pip install numpy scipy matplotlib`.

```
# --- 相图（§5.6 / 附录 A）：10 设置 × 5 seed -------------------------------
python scripts/run.py --pde burgers --methods pi_baseline,pi_align --n-data 0 \
    --align-kind prior --align-weight 1 --steps 6000 --width <16|24|32|64> \
    --nu <0.02|0.01|0.005> --nx <256|512|1024> --nx-store 128 --seeds 0,1,2,3,4 \
    --device cuda --skip-existing --out results/expV        # 各设置同构，见 scripts/final_sweep*.cmd
python scripts/analyze_phase.py                              # 汇总表 + r 的 bootstrap CI

# --- 静默广播 bug（§3）-----------------------------------------------------
python scripts/check_residual.py                             # 监督模型 autograd 残差 vs 平凡解
python tests/test_hard_ic_2d.py                              # 2D 采样器/残差回归测试

# --- 2D 干预对照（§5.7）：约 2 小时 -----------------------------------------
python scripts/sweep2d_controls.cmd                          # fourier / ortho / align random / ic×10

# --- 监督上界与步数扫描（§5.8）----------------------------------------------
python scripts/sweep_refs.cmd                                # data_only + σ_repeat + 6k/20k/50k
python scripts/sweep_dataonly_long.cmd                       # 监督 30k 步（防止"物理>数据"假象）

# --- 诊断与机制（§4、§5.5）---------------------------------------------------
python scripts/eval_saved.py --dir results/expV --write-back # 由 ckpt 补算 train 误差（三项分解）
python scripts/analyze_attribution.py --dirs expV,expW --pde burgers2d
python scripts/ntk_analysis.py --dir results/expQ --out results/ntk.json

# --- 论文配图（results/paper_figs/ → paper/figures/）-------------------------
python scripts/make_paper_figs.py                            # fig1–fig6
```

Code, cached datasets, per-run JSON and figures accompany this draft.
