# Physics-informed training on the public data — predeclared protocol

Date: 2026-09-30. Committed **before** any physics-informed run.

## 1. Why, and why only now

Every result in the manuscript so far is **supervised**. That is the most likely objection at
review, and the project is named for physics-informed operator learning. This adds a
physics-informed training arm so the decomposition can be applied to it directly.

It could not be done earlier because a PDE residual requires a diffusion coefficient that the
data actually satisfies. `scripts/estimate_burgers_nu.py` settled that first
(results below), which is why no residual was written before this point.

## 2. Prerequisite: the coefficient, measured

Least squares on spectral derivatives, `u_t + u u_x - nu u_xx = 0`:

| file | fitted convection | fitted diffusion | `nu_file / pi` | R^2 |
|---|---:|---:|---:|---:|
| Nu = 0.01 | 0.9951 | **3.140e-03** | 3.183e-03 | **0.9999** |
| Nu = 0.001 | 0.6764 | 2.129e-04 | 3.183e-04 | 0.7944 |

So on **Nu = 0.01** the stored trajectories satisfy Burgers with `nu` equal to the filename
value divided by `pi`, to within 1.3%. On **Nu = 0.001** the fit is poor because shocks are
under-resolved at 1024 points, so a physics loss there would enforce an equation the
reference solution does not satisfy.

**Decision: physics-informed runs use Nu = 0.01 only.** The coefficient is preregistered as
**`nu = 3.083e-3`**, the pooled least-squares fit over layers 20, 40, 60, 80, 100 reported by
`scripts/estimate_burgers_nu.py` on that file. It is not tuned on any model output.

(A sign error in the spectral second derivative was found and fixed during this measurement;
see the commit and the analytic sanity check in the script's docstring.)

## 3. Setting

Input: `u(x, 0)` on the same 256-point grid and the same trajectory ids (split seed 20260922,
sizes 1000/200/200) as the supervised Nu=0.01 family, so the comparison is paired.
Target time `T = 1.0`. Model: DeepONet with `trunk_in = 2` taking `(t, x)`, width 64, three
layers of width 128 in both nets, MLP or fixed random Fourier trunk. Derivatives come from
the existing `src/derivatives.py` trunk-derivative utility, already benchmarked in this
repository.

Losses, all weight 1, preregistered and not tuned:

* initial condition `u(0, x_k) = u0(x_k)` at the 256 sensor points;
* PDE residual `u_t + u u_x - nu u_xx` at collocation points with `t` uniform in `(0, 1]` and
  `x` uniform in `[0, 1)`;
* periodicity `u(t, 0) - u(t, 1) = 0`.

Budget: Adam 1e-3, 1500 updates, batch 8 functions, 512 collocation points per function,
seeds 0-4. **No supervision at the target time is used**, so this is genuinely
physics-informed rather than a hybrid.

## 4. Prediction, tied to the manuscript's own mechanism claim

Section 5.7 shows the binding constraint is the branch (the input-to-coefficient map), not
the trunk span, and section 5.4 shows that more supervised updates make per-sample
fluctuation skill worse rather than better. A physics residual changes *what* the branch is
asked to fit, but it does not change the branch's ability to represent the input-to-coefficient
map. The study therefore predicts:

**Physics-informed training will not produce positive per-sample fluctuation skill here.**
The supervised comparison points at the same setting are -0.0273 (MLP) and -0.2443 (Fourier).

Falsification: if physics-informed training yields clearly positive per-sample fluctuation
R^2 (say above +0.2) where supervised training is negative, the branch-bottleneck claim in
section 5.7 is contradicted and must be revisited. That outcome would be reported as such,
not reconciled by reweighting the losses.

## 5. Secondary reporting

Relative L2 at `T = 1.0` and the mean/fluctuation split are reported alongside the supervised
numbers. Loss-component histories are recorded so that an unconverged residual can be
distinguished from a converged one.

## 6. Commands

```bash
python scripts/run_physics_public.py --steps 1500 --device cuda
python scripts/diagnose_fluctuation.py \
    --data external_data/pdebench/physics_nu001_t100 \
    --families physics_nu001_t100_v1=endtoend:test,val \
    --out results/fluctuation_physics_v1 --device cuda
```

## 7. Result

Filled in after the runs; see also `docs/RUN_LOG.md`.
