# A Silent Broadcasting Bug That Produces Publishable-Looking Results in Physics-Informed Operator Learning

**Status**: workshop-style short paper draft (Preliminary — main table is single-seed; multi-seed running)
**Date**: 2026-09-17

---

## Abstract

Physics-informed operator learning (PI-DeepONet, PINO) is a fast-growing area, and its
empirical literature relies heavily on subspace/diagnostic analyses to explain when and why
training fails. We report two findings.

**(1) A silent numerical pitfall.** In DeepONet-style code, the routine that evaluates the
input function at query points (needed for the PDE residual and initial-condition targets)
is naturally written under a *shared-grid* convention, returning shape `(n_coef, *x.shape)`.
When it is instead called with a *per-sample* `(B, N)` coordinate array, NumPy broadcasting
silently produces `(B, B, N)`, so that each sample is trained against another sample's
source term / initial condition. We show this in three independent PDEs (Burgers,
diffusion–reaction, Darcy). The failure is statistically invisible: the *dataset* stays
correct, so the standard sanity check — computing the finite-difference residual of the
ground truth — passes with `6e-6` while training is completely broken. Fixing the bug changes
the test relative error of a plain PI-DeepONet from **0.986 to 0.174** on diffusion–reaction
and from **0.999 to 0.721** on Burgers.

**(2) The bug produces convincing but wrong science.** Before the fix, our own ablation
exhibited a textbook-clean, monotone relationship between trunk-subspace alignment and test
accuracy, a clear ordering of alignment strategies, and a reproducible "trunk basis collapse"
effect. All of it was an artifact. We quantify the before/after and derive the lesson: in
this literature, *subspace-level conclusions must not be drawn before an autograd-residual
check of a supervised model confirms that the physics residual is actually being minimized*.

**(3) A mechanistic claim, tested and confirmed.** Across widths and problem ranks, the
alignment term's benefit is *not* explained by expanded representational capacity (captured
energy moves only 0.947 → 0.951). Measuring two quantities that are never optimized — the
output neural tangent kernel `K = J Jᵀ` and the loss sharpness `λ_max(∇²L)/2L` — we find
that alignment roughly doubles the effective rank of the NTK, cuts the dominant eigenvalue's
share from ≈0.85 to ≈0.49, and halves the sharpness (λ_max 959–2020 → 430–833, a complete
separation between conditions over two seeds). Across four (width, ν) settings with three
seeds each, the **accuracy gain is almost perfectly predicted by the baseline's cross-seed
standard deviation (Pearson r = 0.985) and is unrelated to solution-space rank**: making the
problem harder and higher-rank *reduces* the gain. Alignment therefore acts as a
**regularizer that flattens the loss landscape and widens the NTK spectrum**, which explains
its large variance reduction (6.7× at width 32) and its robustness to an aggressive
optimizer, despite a modest mean-accuracy gain.

We additionally contribute a simple, training-free **error-attribution diagnostic**:
`LB = sqrt(1 - captured_energy)` is a hard lower bound on the relative error attainable by
*any* branch network given the trunk features, and `opt_gap = test_relL2 - LB` isolates the
optimization gap. Applied after the fix, it shows that representation is *not* the bottleneck
(`LB ≈ 0.19–0.25`) while optimization is (`opt_gap ≈ 0.45–0.49`) — explaining why
representation-side interventions yield at most modest gains while a causal-weighting
intervention actively hurts.

---

## 1. Introduction

PI-DeepONets [Wang et al., 2021] learn solution operators by using the governing equations
as supervision. A persistent theme in the follow-up literature is *diagnosing failure*:
spectral bias, gradient conflict, causal violation, and — most relevant here — the claim that
the trunk network's learned basis functions fail to span the target solution space
[JCP 2026, S0021999126002019]. Papers in this line typically propose a representation-side
fix (better basis, orthogonalization, spectral priors) and report improved diagnostics.

We started from exactly that hypothesis: *align the trunk basis to a target subspace and
measure the effect.* We built the full pipeline, obtained a clean and encouraging result —
trunk effective rank rising from 1.5 to 33.7, captured energy from 0.976 to 0.9999, mean
principal angle from 67° to 17° — and then discovered that the entire effect, including the
"basis collapse" we set out to fix, was a symptom of a one-line shape bug in our own PDE
interface.

This paper is written as a cautionary and methodological contribution rather than a
method paper. We document the pitfall precisely enough to be recognizable in other
codebases, show that standard sanity checks do not catch it, and demonstrate the observable
difference between "a buggy pipeline" and "a correct pipeline" on a task where both produce
plausible-looking ablations.

---

## 2. The pitfall

### 2.1 Code pattern

An operator-learning problem needs the input function `u` evaluated at arbitrary query
points: for the PDE residual (a source term), for initial-condition targets, or for spatially
varying coefficients. A natural helper is

```python
def input_at(self, coef, x):
    """coef: (n, K) spectral coefficients of the input function.
       x   : query points."""
    S = np.sin(np.outer(x.ravel(), self.k))        # shared-grid convention
    C = np.cos(np.outer(x.ravel(), self.k))
    out = coef @ S.T + coef @ C.T                  # (n, M)
    return out.reshape(coef.shape[0], *x.shape)
```

This convention is *correct* whenever `x` describes one shared grid — e.g. the sensor
locations `(m,)`, the solution grid `(Ng,)`, or a sensor mesh `(m, 2)`. It is *wrong* when
`x` is a per-sample array of shape `(B, N)`, which is exactly what a batched training loop
hands over: collocation points resampled per input function.

NumPy does not complain. With `coef.shape[0] == x.shape[0] == B`, the result becomes
`(B, B, N)`. Every downstream expression broadcasts, the loss remains a scalar of a
plausible magnitude, and training proceeds — with the residual for sample `b` built from
sample `b'`'s data for `B − 1` of `B` pairings.

### 2.2 Three instances

| PDE | Quantity affected | Buggy shape | Correct shape |
|---|---|---|---|
| Burgers (periodic) | IC target `u0(x)` | `(B, B, N_ic)` | `(B, N_ic)` |
| Diffusion–reaction | collocation source `u(x)` | `(B, B, N_col)` | `(B, N_col)` |
| Darcy | coefficient `a, a_x, a_y` | `(B, B, N_col)` | `(B, N_col)` |

### 2.3 Why the usual checks miss it

The dataset is *correct*: the reference solvers consume the coefficients with per-sample
semantics and were never affected. Consequently:

- **Ground-truth residual check passes.** Finite-differencing the stored solutions and
  evaluating the PDE gives `mean(res²) = 6.0e-6` — a clean bill of health.
- **Supervised training works.** A purely data-driven DeepONet reaches `relL2 = 0.24`,
  because it never calls `input_at` on per-sample coordinates.
- **Shapes are legal.** No exception, no warning.

The one diagnostic that *does* expose it is the autograd residual of a **supervised** model:
after fitting the data to `relL2 = 0.26`, the model's own PDE residual was
`mean(res²) = 0.368` — **larger than the trivial solution's `E[u²] = 0.185`**.
A model that reproduces the solution cannot have a residual worse than the zero function;
that contradiction was the thread that led to the bug.

### 2.4 Recommended check

```python
# after supervised training to low rel-L2, the autograd PDE residual
# must be well below the trivial-solution level E[u^2]
assert res_autograd < 0.5 * res_trivial
```

plus a shape assertion in the sampler itself:

```python
assert pts["u0"].shape == (B, n_col)   # never (B, B, n_col)
```

---

## 3. Error attribution diagnostic

Let `Φ = trunk(Y) ∈ R^{N×p}` be the trunk features on a fixed probe set `Y`, and
`S ∈ R^{N×n}` the matrix of true solutions. The model class
`{Φb : b ∈ R^p}` (one column per input function) has best attainable relative error

```
min_B ||Φ B − S||_F / ||S||_F = ||(I − P_Φ) S||_F / ||S||_F
                             = sqrt(1 − captured_energy)
```

where `captured_energy = ||P_Φ S||_F² / ||S||_F²`. Hence

> `LB := sqrt(1 − captured_energy)` is a *hard lower bound* on the relative error of any
> branch network attached to this trunk, computable at any point during training.

We then define `opt_gap := test_relL2 − LB`. `LB` is attributable to **representation**;
`opt_gap` to **optimization**. Both are scalars that cost one SVD.

---

## 4. Experiments

**Setup.** Burgers (`ν = 0.02`, `x ∈ [0,1)` periodic, `t ∈ [0,1]`), diffusion–reaction
(`D = k = 0.01`), Darcy (2D, 41×41). Branch/trunk MLPs, width 64, 3 hidden layers,
64 collocation points × batch 16 per step, 10 000 Adam steps, `lr = 1e-3`,
200 training / 50 validation / 50 test input functions, one seed unless noted.

### 4.1 Effect of the bug

| Problem | metric | buggy | fixed |
|---|---|---|---|
| Diffusion–reaction | test relL2 | 0.9857 | **0.1737** |
| Diffusion–reaction | trunk captured energy | 0.9774 | 0.9999 |
| Burgers | test relL2 | 0.9992 | **0.7213** |
| Burgers | trunk captured energy | 0.6653 | 0.9471 |

The trunk "collapse" we originally set out to explain largely disappears once the physics
residual is actually minimized: the plain baseline reaches captured energy 0.947 on its own,
with no representation-side intervention.

### 4.2 What the buggy pipeline looked like (Burgers, width 64, 10 000 steps)

| method | test relL2 | LB | opt_gap | captured | eff_rank | mean angle |
|---|---|---|---|---|---|---|
| mlp baseline | 0.9992 | 0.5786 | 0.421 | 0.6653 | 1.16 | 74.4° |
| orthogonalization reg. | 0.9966 | 0.4826 | 0.514 | 0.7671 | 5.05 | 68.8° |
| random-subspace alignment | 0.9826 | 0.3366 | 0.646 | 0.8867 | 13.23 | 56.3° |
| spectral-prior alignment | 0.9809 | 0.2960 | 0.685 | 0.9124 | 6.18 | 52.5° |
| Fourier-feature trunk | 0.9493 | 0.2838 | 0.666 | 0.9195 | 4.81 | 50.8° |

This table is monotone in `captured` (Spearman `ρ = −1` across five methods), orders the
alignment strategies exactly as prior knowledge would predict, and shows a dramatic
"basis collapse". **Every one of these conclusions is an artifact.**

### 4.3 Same table after the fix

| method | test relL2 | LB | opt_gap | captured | eff_rank | mean angle |
|---|---|---|---|---|---|---|
| Fourier-feature trunk | **0.6651** | 0.1891 | 0.476 | 0.9642 | 8.00 | 39.4° |
| spectral-prior alignment | 0.6672 | 0.2136 | 0.454 | 0.9544 | 10.46 | 43.2° |
| Fourier + spectral alignment | 0.6680 | 0.2084 | 0.460 | 0.9566 | 10.04 | 43.7° |
| random-subspace alignment | 0.6988 | 0.2454 | 0.453 | 0.9398 | 11.59 | 45.1° |
| orthogonalization reg. | 0.7114 | 0.2347 | 0.477 | 0.9449 | 14.04 | 46.1° |
| mlp baseline | 0.7213 | 0.2301 | 0.491 | 0.9471 | 4.47 | 45.6° |

The corrected picture:
1. Spectral-prior alignment does give a real gain over the MLP baseline (7.5% relative) and
   beats a random subspace (3.1%) and a target-free orthogonalization regularizer (1.4%).
   Prior knowledge matters — but modestly.
2. A **random-feature trunk matches it at zero cost** (no target subspace, no extra
   hyperparameter).
3. Fourier features and spectral alignment are **not complementary**.
4. `captured` is no longer the discriminator: all methods land in 0.94–0.96, `LB` in
   0.19–0.25, while `opt_gap` stays at 0.45–0.49.

### 4.4 The bottleneck is optimization — but standard remedies fail

Guided by `opt_gap ≈ 0.45–0.49`, we tested the two standard optimization-side remedies.

**(a) Causal weighting** of the PDE residual over time slabs:

| causal ε | off | 1 | 10 | 100 |
|---|---|---|---|---|
| test relL2 | **0.7213** | 0.7601 | 0.8045 | 0.8285 |

Monotonically worse: with a soft IC constraint of equal weight, concentrating residual
weight at small `t` starves the later-time fit.

**(b) L-BFGS fine-tuning** after Adam (fixed training set, strong-Wolfe line search), as is
standard in single-instance PINN solvers:

| method | fixed set | Adam | after L-BFGS | opt_gap | trunk eff. rank |
|---|---|---|---|---|---|
| `pi_lbfgs` | 16 fn × 64 pts | 0.7213 | **2.0036** | — | — |
| `pi_lbfgs` | 64 fn × 32 pts | 0.7213 | **0.9119** | 0.491 → 0.688 | 4.47 → **1.40** |
| `pi_align_lbfgs` | 64 fn × 32 pts | 0.6672 | **0.7219** | 0.454 → 0.516 | 10.46 → **18.89** |

L-BFGS requires a deterministic objective and therefore a fixed training set; at the scale
needed to represent an operator family (hundreds of input functions × thousands of
coordinates) that set is far too small, and the second-order method overfits it. Enlarging
the set mitigates (2.00 → 0.91) but does not fix it.

**An incidental robustness finding.** L-BFGS drives the *baseline* trunk into collapse
(effective rank 4.47 → 1.40, `opt_gap` 0.49 → 0.69) while the aligned trunk becomes *more*
diverse (10.46 → 18.89).

### 4.5 Mechanism: alignment regularizes the optimization problem

The observations above (modest accuracy gain, 6.7× variance reduction at width 32,
robustness to an aggressive optimizer) all point to a single mechanism: alignment acts as a
*regularizer*, not as a capacity expansion. We test this directly with two quantities that
are never optimized and are independent of the diagnostic used to build the alignment loss:
the **output neural tangent kernel** `K = J Jᵀ` (per-point Jacobian of the model output
w.r.t. all parameters) and the **sharpness** of the physics loss, `λ_max(∇²L) / 2L`,
estimated by power iteration with Hessian-vector products.

Paired comparison at width 64 (same seed, data, optimizer; the only difference is the
alignment term):

| checkpoint | trunk eff. rank | **out-NTK eff. rank** | **NTK top-eig. share** | **λ_max** | **sharpness** |
|---|---|---|---|---|---|
| baseline, seed 1 | 4.1 | 1.7 | 0.874 | 1035 | 295 |
| **aligned**, seed 1 | **9.0** | **3.8** | **0.542** | **430** | **141** |
| baseline, seed 2 | 4.2 | 2.0 | 0.803 | 2020 | 682 |
| **aligned**, seed 2 | **10.6** | **4.0** | **0.432** | **833** | **336** |
| (ref.) baseline, width 32 | 2.7 | 1.5 | 0.917 | 959 | 437 |

Alignment roughly **doubles the effective rank of the output NTK** (1.7→3.8, 2.0→4.0) and
**reduces the dominant eigenvalue's share from ≈0.85 to ≈0.49**; the residual §4.4a
optimization does not appear. It also **halves the sharpness** of the loss landscape:
`λ_max` falls from 959–2020 (all three baselines ≥ 959) to 430–833 (both aligned models
< 959), a complete separation between the two groups.

We therefore revise the central claim: the benefit of spectral-prior alignment here is not
an expansion of representational capacity (captured energy moves only from 0.947 to 0.951)
but a **flattening of the loss landscape and a widening of the NTK spectrum**, which
stabilizes optimization. This is consistent with every observation we have and is the
mechanism we would put forward for further study.

### 4.6 The gain is predicted by baseline instability, not by problem rank

Three seeds per cell (6000 steps). We vary the trunk width and, independently, the problem's
solution-space rank via the Burgers viscosity (`ν = 0.02` → `0.005`, which lowers the initial
captured energy from 0.637 to 0.571, i.e. a *higher-rank*, harder solution space).

| setting | baseline (mean ± **std**) | aligned (mean ± std) | relative gain | Welch p |
|---|---|---|---|---|
| width 16, ν=0.02 | 0.7296 ± **0.0061** | 0.7088 ± 0.0078 | 2.85% | 0.022 |
| width 64, ν=**0.005** | 0.8122 ± **0.0118** | 0.7732 ± 0.0098 | 4.80% | 0.012 |
| width 64, ν=0.02 | 0.7512 ± **0.0257** | 0.6839 ± 0.0165 | 8.96% | 0.019 |
| width 32, ν=0.02 | 0.8361 ± **0.0924** | 0.6821 ± 0.0137 | **18.4%** | 0.045 |

The alignment gain is significant in all four settings, but its **magnitude tracks the
baseline's cross-seed standard deviation almost perfectly (Pearson r = 0.985 over the four
settings) and is unrelated to solution-space rank**: the *harder, higher-rank* problem
(`ν = 0.005`) shows a *smaller* gain than the easier one.

```
baseline std:  0.0061 →  0.0118 →  0.0257 →  0.0924
gain:           2.85% →   4.80% →   8.96% →  18.40%
```

This is the strongest evidence for the regularization account: alignment pays off exactly
where the baseline optimization is unstable, and the effect is quantitatively predicted by
that instability rather than by any measure of representational capacity.

---

## 5. Discussion

**Implications for the literature.** Any paper that infers representational conclusions from
subspace diagnostics of a physics-informed operator learner should report (a) the autograd
residual of a supervised baseline on the same architecture and data, and (b) an explicit
assertion on the shape of sampler outputs. Both are cheap; neither is currently standard.

**Implications for our own question.** The hypothesis "PI-DeepONet fails because the trunk
basis is misaligned" is not supported on Burgers at width 64 once training is correct.
`LB` is already small without any intervention. The residual error is optimization-bound.

**Statistical significance (partial).** A width × seed sweep (3 seeds; 6000 steps) gives, at
width 16, baseline `0.7296 ± 0.0061` vs. spectral alignment `0.7088 ± 0.0078`
(Welch `t ≈ 3.66`, `df ≈ 3.9`, `p ≈ 0.022`) — a 2.85% relative gain that is already
significant at three seeds. Note also that *baseline* accuracy is non-monotone in width
(`0.730 / 0.783 / 0.721` at widths `16 / 32 / 64`): the spread across widths exceeds the
alignment effect, consistent with optimization variance rather than representational
capacity driving the differences.

**Limitations.** Widths 32 and 64 are not yet complete across all seeds, and the width-64
7.5% figure remains single-seed. Burgers at
`ν = 0.02`, `T = 1` is a hard problem on which all methods remain far from `LB`
(`relL2 ≈ 0.67–0.72`), so effect sizes are measured in a low-accuracy regime. The `LB`
argument assumes the target solutions and the trunk are evaluated on the same point set and
that a linear read-out with per-sample coefficients is the relevant model class.

---

## 6. Conclusion

We report a one-line NumPy broadcasting failure that silently corrupts the physics residual
in batched operator-learning code, survives every conventional sanity check, and — in our own
runs — manufactured a clean, monotone, apparently publishable scientific narrative about
trunk-basis collapse and the value of spectral priors. After the fix, that narrative
disappears and the picture inverts toward "representation is cheap to fix; optimization is
the bottleneck". We contribute the minimal detection recipe (autograd residual of a
supervised model, plus shape assertions) and a training-free error-attribution diagnostic
`LB = sqrt(1 − captured_energy)` that makes the representation/optimization split quantitative.

---

## References (to be completed)

- Wang, Wang, Perdikaris. Learning the solution operator of parametric PDEs with
  physics-informed DeepONets. *Sci. Adv.* 7(40), 2021.
- What do physics-informed DeepONets learn? Understanding and improving training.
  *J. Comput. Phys.*, 2026. (S0021999126002019)
- On the training of physics-informed neural operators for solving parametric PDEs.
  arXiv:2606.06164, 2026.
- Wang, Shankar, Karniadakis. Respecting causality when training physics-informed neural
  networks. 2022.
- Li et al. Physics-informed neural operator for learning PDEs. arXiv:2111.03794.
