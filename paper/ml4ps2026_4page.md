# What Really Limits Physics-Informed Operator Learning?
*A silent broadcasting pitfall, a training-free error-attribution diagnostic, and evidence that subspace alignment regularizes rather than expands*

> **Target**: ML4PS 2026 (9th Workshop on Machine Learning and the Physical Sciences), Research Track.
> 4 pages max excluding references. NeurIPS 2026 LaTeX template. **Double-blind — fully anonymized.**
> Footer: *Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute.*

---

## Abstract

Physics-informed operator learning is increasingly analysed through subspace diagnostics, with a
recurring claim that training fails because the trunk network's learned basis functions cannot span
the target solution space. We report three findings that redirect this line of inquiry. **First**, a
silent numerical pitfall: the routine that evaluates the input function at query points is naturally
written under a shared-grid convention returning shape `(n_coef, *x.shape)`; when a batched loop
hands it per-sample `(B, N)` coordinates, NumPy broadcasting silently produces `(B, B, N)`, so each
sample is trained against another sample's source term or initial condition. It survives every
conventional check — the dataset stays correct (ground-truth finite-difference residual `6e-6`),
shapes are legal, supervised training succeeds. Fixing it changes PI-DeepONet test error from
**0.986 → 0.174** (diffusion–reaction) and **0.999 → 0.721** (Burgers). Critically, the buggy
pipeline produced a clean, monotone, publishable-looking narrative (Spearman `ρ = −1` between
subspace alignment and accuracy, a dramatic "trunk collapse") — all artefactual. **Second**, a
training-free attribution diagnostic: with trunk features `Φ` and true solutions `S`, the best
relative error of *any* branch network is exactly `sqrt(1 − captured_energy)`, so
`test_relL2 = LB + opt_gap` separates representation from optimization limits without training to
convergence. After the fix, `LB ≈ 0.19–0.25` while `opt_gap ≈ 0.45–0.49`: representation is not
binding. **Third**, a mechanism: alignment of the trunk to an analytic spectral prior halves loss
sharpness and doubles output-NTK effective rank while captured energy moves only `0.947 → 0.951`;
over **ten settings × five seeds** the gain is positive in 10/10 and significant in 8/10; and in 2D,
where the baseline collapses deterministically onto `u ≡ 0` (`relL2 = 1.004`, NTK rank 1.1) and is
**representation-bound** (85% of its error is our lower bound), a random Fourier-feature trunk —
no target subspace, no tuning — repairs it better than alignment does (`1.004 → 0.693`, **30.9%**,
vs `0.821`; p = 1.1e-3), while a target-free decorrelation regularizer and a 10× IC weight both fail. We also **retract** an
earlier claim of ours that the gain is linearly predicted by baseline instability: it collapses from
`r = 0.985` to `r = 0.318` (95% CI `[−0.199, 0.671]`) with better data.

---

## 1. Introduction

PI-DeepONet [3] and PINO [4] train operator networks from governing equations rather than paired
simulation data. Because they are expensive and hard to debug, a substantial literature explains
their failures: spectral bias, gradient pathologies, causality violations [5], landscape stiffness
[6], and — most relevant here — the proposal that the trunk basis cannot span the target solution
space [7].

We began from that last hypothesis, built the full pipeline (a subspace diagnostic, a differentiable
alignment loss toward an analytic spectral prior, and controlled comparisons), and obtained
excellent results: trunk effective rank 1.5 → 33.7, captured energy 0.976 → 0.9999, mean principal
angle 67° → 17°, with five methods ranked exactly as theory predicted. All of it was an artefact of
a one-line shape bug in our own PDE interface.

We therefore write the paper in two halves. §3 documents the pitfall precisely, shows why standard
sanity checks are blind to it, and demonstrates it can manufacture a coherent publishable story.
§4–§6 report what survives after the fix: an attribution diagnostic, and a mechanistic account of
subspace alignment as regularization. We regard the first half as the more immediately useful
contribution to the community.

---

## 2. Setup

Burgers `u_t + u u_x = ν u_xx` on `x ∈ [0,1)` periodic, `t ∈ [0,1]`, solved by a Strang-split
spectral method; also 2D Burgers (`u_t + u·∇u = ν Δu`, 2D spectral splitting) and a 1D
diffusion–reaction system. Input functions are truncated spectral expansions, so the input field is
analytic and available at any query point. Branch/trunk MLPs, 3 hidden layers, Adam `lr = 1e-3`,
6000 steps (10000 for §3.5), 64 collocation points × 16 input functions per step, 200/50/50
train/val/test input functions. Test relative `L₂`; Welch's `t`-test over seeds.

---

## 3. A silent pitfall in DeepONet-style training loops

### 3.1 The code pattern

```python
def input_at(self, coef, x):
    """coef: (n, K) coefficients of the input function.  x: query points."""
    S = np.sin(np.outer(x.ravel(), self.k))      # shared-grid convention
    C = np.cos(np.outer(x.ravel(), self.k))
    out = coef @ S.T + coef @ C.T                # (n, M)
    return out.reshape(coef.shape[0], *x.shape)
```

Correct whenever `x` is one shared grid — sensor locations `(m,)`, evaluation grid `(Ng,)`, sensor
mesh `(m, 2)`. **Wrong** when `x` is a per-sample array `(B, N)`, which is exactly what a batched
loop hands over. With `coef.shape[0] == x.shape[0] == B` the output becomes `(B, B, N)`; every
downstream expression broadcasts, the loss stays a scalar of plausible magnitude, and sample `b` is
trained against sample `b'`'s data for `B − 1` of `B` pairings. No exception, no warning, no NaN.
We found this in three independent PDEs (Burgers IC target, diffusion–reaction collocation source,
Darcy coefficient and its gradients).

### 3.2 Why the usual checks are blind

| Check | Result | Why it passes |
|---|---|---|
| Ground-truth residual (FD on stored solutions) | `mean(res²) = 6.0e-6` | The **dataset is correct**: solvers consume coefficients with per-sample semantics and never call the buggy helper. |
| Supervised training | reaches `relL2 = 0.26` | Data-driven training never calls the helper on per-sample coordinates. |
| Shape assertion | no error | `(B, B, N)` is legal; broadcasting is silent. |
| Loss magnitude | plausible | The mean over `B²` pairings is statistically similar to the correct mean. |

The only diagnostic that exposed it was the **autograd PDE residual of a supervised model**: after
fitting the data to `relL2 = 0.26`, the model's own residual was `mean(res²) = 0.368`, while the
trivial solution `s ≡ 0` has residual `E[u²] = 0.185`. A model that reproduces the solution cannot
have a residual worse than the zero function; that contradiction was the thread.

**Detection recipe** (two lines, both cheap):

```python
assert res_autograd(model) < 0.5 * res_trivial          # after supervised training
assert pts["u0"].shape == (B, n_col)                     # never (B, B, n_col)
```

### 3.3 What the bug does to scientific conclusions

Before the fix, our Burgers ablation (width 64, 10 000 steps) looked like this:

| method | test relL2 | LB | opt_gap | captured | eff. rank | mean angle |
|---|---|---|---|---|---|---|
| mlp baseline | 0.9992 | 0.5786 | 0.421 | 0.6653 | 1.16 | 74.4° |
| orthogonalization reg. | 0.9966 | 0.4826 | 0.514 | 0.7671 | 5.05 | 68.8° |
| random-subspace alignment | 0.9826 | 0.3366 | 0.646 | 0.8867 | 13.23 | 56.3° |
| spectral-prior alignment | 0.9809 | 0.2960 | 0.685 | 0.9124 | 6.18 | 52.5° |
| Fourier-feature trunk | 0.9493 | 0.2838 | 0.666 | 0.9195 | 4.81 | 50.8° |

Accuracy is monotone in captured energy (`ρ = −1`); spectral priors beat a random subspace and a
target-free regularizer; a clear "trunk collapse" explains the baseline's failure; the story
"PI-DeepONet fails because its basis is misaligned" appears supported. **Every one of these is
false.** After fixing the one-line broadcasting bug, with everything else held fixed:

| problem | metric | buggy | fixed |
|---|---|---|---|
| diffusion–reaction | test relL2 | 0.9857 | **0.1737** |
| diffusion–reaction | captured energy | 0.9774 | 0.9999 |
| Burgers | test relL2 | 0.9992 | **0.7213** |
| Burgers | captured energy | 0.6653 | 0.9471 |

The "collapse" essentially disappears: the plain baseline reaches captured energy 0.947 with no
representation-side intervention.

---

## 4. An error-attribution diagnostic

With probe trunk features `Φ ∈ R^{N×p}` and true solutions `S ∈ R^{N×n}`, the model class
`{ΦB}` realises column space `span(Φ)`.

**Proposition 1.** `min_B ||ΦB − S||_F / ||S||_F = ||(I − P_Φ)S||_F / ||S||_F = sqrt(1 − captured_energy)`.
*Proof.* The minimiser is `B = Φ⁺S`, giving the projection residual; `||P_Φ S||_F²/||S||_F²` is the
captured energy (weighted by `σ_i²` in the energy-weighted version used here). ∎

Hence **`LB := sqrt(1 − captured_energy)`** is a hard, computable-at-any-time lower bound on the
error of *any* branch network on this trunk, and **`opt_gap := test_relL2 − LB`** isolates the
optimization gap. Large `LB` ⇒ representation-limited; large `opt_gap` ⇒ optimization-limited.
After the fix all representation-side methods land at `captured ∈ [0.94, 0.96]`, i.e.
`LB ∈ [0.19, 0.25]`, while `opt_gap ∈ [0.45, 0.49]` dominates — correctly predicting that
representation-side work has little headroom. Consistently, the two standard optimization-side
remedies both fail here: **causal weighting** [5] degrades monotonically
(`0.721 → 0.760 → 0.805 → 0.829` for `ε = 1, 10, 100`), and **L-BFGS fine-tuning** after Adam
degrades too (`0.721 → 0.912`; with a smaller fixed set, `→ 2.004`), because L-BFGS needs a
deterministic objective over a fixed training set that is far too small to represent an operator
family.

---

## 5. Mechanism: alignment regularizes rather than expands

We measure two quantities that are **never optimized** and independent of the diagnostic used to
build the alignment loss: the output neural tangent kernel `K = J Jᵀ` (per-point Jacobian of the
model output w.r.t. all parameters) and loss sharpness `λ_max(∇²L)/2L` (power iteration with
Hessian-vector products). Paired comparison at width 64 (same seed, data, optimizer):

| checkpoint | trunk eff. rank | out-NTK eff. rank | NTK top-eig. share | λ_max | sharpness |
|---|---|---|---|---|---|
| baseline, seed 1 | 4.1 | 1.7 | 0.874 | 1035 | 295 |
| **aligned**, seed 1 | **9.0** | **3.8** | **0.542** | **430** | **141** |
| baseline, seed 2 | 4.2 | 2.0 | 0.803 | 2020 | 682 |
| **aligned**, seed 2 | **10.6** | **4.0** | **0.432** | **833** | **336** |

Alignment roughly **doubles the output-NTK effective rank**, cuts the dominant eigenvalue share from
≈0.85 to ≈0.49, and **halves sharpness**, with `λ_max` falling from 959–2020 (all baselines ≥ 959)
to 430–833 (both aligned < 959) — a complete separation. Captured energy moves only
`0.947 → 0.951`: **not a capacity expansion.**

### 5.1 Ten settings, five seeds

| setting | baseline (mean ± **std**) | aligned (mean ± std) | gain | Welch p |
|---|---|---|---|---|
| **2D, w32** | 1.0039 ± 0.0024 | 0.8205 ± 0.0421 | **18.27%** | 0.0006 |
| ν=0.02, w32 | 0.8047 ± **0.0780** | 0.6815 ± 0.0102 | 15.31% | 0.024 |
| ν=0.01, w32 | 0.8353 ± **0.0729** | 0.7420 ± 0.0080 | 11.18% | 0.045 |
| ν=0.02, w64 | 0.7627 ± 0.0122 | 0.6823 ± 0.0118 | 10.55% | <0.0001 |
| ν=0.005, w32 | 0.8435 ± **0.0731** | 0.7683 ± 0.0056 | 8.92% | 0.083 |
| ν=0.02, w24 | 0.7500 ± 0.0144 | 0.6874 ± 0.0081 | 8.35% | 0.0001 |
| ν=0.01, w64 | 0.7999 ± 0.0033 | 0.7465 ± 0.0100 | 6.67% | 0.0001 |
| ν=0.005, w64 | 0.8117 ± 0.0089 | 0.7691 ± 0.0091 | 5.25% | 0.0001 |
| ν=0.02, w16 | 0.7370 ± 0.0145 | 0.7020 ± 0.0108 | 4.74% | 0.003 |
| ν=0.005, w16 | 0.7974 ± 0.0128 | 0.7797 ± 0.0138 | 2.22% | 0.070 |

The gain is **positive in 10/10 and significant in 8/10** (five at `p < 0.001`). Its most robust
effect is on variance: across the nine 1D settings the baseline σ spans **23.6×** (0.0033–0.0780)
while the aligned σ spans only **2.5×** (0.0056–0.0138) — alignment **clamps σ into a
problem-independent band ≈0.006–0.014, from both sides** (7.6–13.1× reductions where the baseline
is far above it; ≈1 where the baseline is inside it; and 0.33× where the baseline, at 0.0033, lies
below it, though the mean still improves by 6.67%).

**Retraction.** In an earlier iteration we reported the gain was almost perfectly predicted by
baseline σ (`r = 0.985`, four settings, three seeds). With better data this collapses:
`r = 0.318`, 95% seed-level bootstrap CI `[−0.199, 0.671]`, stable as `n` grew 8 → 10. The regressor
is itself estimated from few seeds, and fitting a noisy regressor inflates `|r|`. **We retract the
linear law**; the defensible claims are the sign of the gain, its significance, the variance clamp,
and §6.

---

## 6. 2D: representation-bound — but alignment is not the fix

In 2D the baseline does not partially fail — it **collapses completely and reproducibly**:
`1.0039 ± 0.0024` over five seeds (captured ≈0.27, out-NTK effective rank 1.08–1.10, top eigenvalue
98.5–98.9%). It has collapsed onto `u ≡ 0`, an exact solution of the PDE and the periodic boundary
conditions, leaving only the initial condition to supply gradient; it is not *unstable*, it is
*degenerate*. A three-way split (`test = LB + opt_gap_train + gen_gap`, §4) shows `LB = 0.849`
accounts for **85%** of its error while `gen_gap ≈ 0`: this is the representation-limited regime.

We then isolated what repairs it (width 32, 6000 steps, 5 seeds):

| intervention | test relL2 | gain | p | captured |
|---|---|---|---|---|
| baseline | 1.0039 ± 0.0024 | — | — | 0.28 |
| orthogonalization regularizer (target-free) | 1.0016 ± 0.0010 | 0.22% | 0.112 | 0.30 |
| 10× initial-condition weight | 1.0029 ± 0.0013 | 0.10% | 0.464 | 0.28 |
| alignment, random smooth subspace | 0.7910 ± 0.0165 | 21.21% | 6.0e-6 | 0.60 |
| alignment, analytic spectral prior | 0.8205 ± 0.0421 | 18.27% | 6.1e-4 | 0.61 |
| **random Fourier-feature trunk** | **0.6933 ± 0.0199** | **30.94%** | 3.1e-6 | **0.77** |

**(i)** Diversity alone is *not* enough: the target-free decorrelation regularizer, which only asks
`ΦᵀΦ ∝ I`, is ineffective in 2D (p = 0.112). What matters is not how many directions the trunk spans
but what kind of function subspace it spans. **(ii)** The specific prior does not matter: a random
smooth subspace is at least as good as the analytic spectral prior (0.791 vs 0.821). **(iii)** The
best fix is free: random Fourier features beat alignment by 15.5% (p = 1.1e-3). So 2D is
representation-bound, but what rescues it is a **spectrally structured trunk**, which architecture
supplies more cheaply than an alignment loss.

---

## 7. Conclusion

We set out to test whether PI-DeepONet fails because its trunk basis cannot span the solution space,
obtained a clean affirmative answer, and found it was produced by a one-line NumPy broadcasting bug
that survives every conventional sanity check. After the fix the representation-side story
collapses: `LB` is already small, the trunk is fine, and the residual error is optimization-bound —
and the two standard optimization-side remedies both make things worse. What survives is a
training-free attribution diagnostic, and a mechanism (alignment regularizes, and in 2D is
necessary) confirmed by two never-optimized geometric quantities. Along the way we also retracted
one of our own quantitative claims, which is the same failure mode as the bug — a convincing
regularity produced by too little data and a noisy regressor — caught only because we went back and
measured it properly. We hope the detection recipe and the attribution diagnostic are useful beyond
this paper's own claims.

**Reproducibility.** Code, cached datasets, per-run JSON and all figures accompany this submission
(anonymous release; DOI to be added upon acceptance).

---

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *J. Comput. Phys.* 378:686–707, 2019.
2. L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. Learning nonlinear operators via DeepONet. *Nat. Mach. Intell.* 3(3):218–229, 2021.
3. S. Wang, H. Wang, P. Perdikaris. Learning the solution operator of parametric PDEs with physics-informed DeepONets. *Sci. Adv.* 7(40):eabi8605, 2021.
4. Z. Li, N. Kovachki, K. Azizzadenesheli, et al. Physics-informed neural operator for learning PDEs. arXiv:2111.03794; *ACM TOMS*, 2024.
5. S. Wang, S. Sankaran, P. Perdikaris. Respecting causality when training physics-informed neural networks. arXiv:2203.07404, 2022.
6. A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney. Characterizing possible failure modes in physics-informed neural networks. *NeurIPS* 2021.
7. What do physics-informed DeepONets learn? Understanding and improving training. *J. Comput. Phys.*, 2026 (S0021999126002019).
8. N. Chen, C. Cui, A. Chen, S. Wang, R. Ma. On the training of physics-informed neural operators. arXiv:2606.06164, 2026.
9. S. Wang, X. Yu, P. Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. *J. Comput. Phys.* 449:110768, 2022.
10. S. Wang, Y. Teng, P. Perdikaris. Understanding and mitigating gradient pathologies in PINNs. *SIAM J. Sci. Comput.* 43(5):A3055–A3081, 2021.
11. A. Jacot, F. Gabriel, C. Hongler. Neural tangent kernel. *NeurIPS* 2018.
12. N. S. Keskar, D. Mudigere, J. Nocedal, et al. On large-batch training: Generalization gap and sharp minima. *ICLR* 2017.
13. M. Tancik, P. P. Srinivasan, B. Mildenhall, et al. Fourier features let networks learn high frequency functions. *NeurIPS* 2020.
14. P. Hu, H. Feng, H. Liu, et al. RealPDEBench. *ICLR* 2026 (Oral); arXiv:2601.01829.
15. W. Zhang, W. Suo, J. Song, W. Cao. PINNs as intelligent computing technique for PDEs: Limitation and future prospects. *Sci. China Phys. Mech. Astron.* 69:214602, 2026.
16. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. DeepXDE. *SIAM Rev.* 63(1):208–228, 2021.
