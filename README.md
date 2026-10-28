# Conserved Means and Lost Fluctuations

A mean/fluctuation diagnosis of operator learning across six PDEBench operator families.

> **Status.** Evidence-audited research draft, not a submitted manuscript.
> **The authoritative text is [`paper/main.md`](paper/main.md);** this file is only a map of
> the repository. The current scientific position is summarised in
> [`docs/CURRENT_CONCLUSIONS.md`](docs/CURRENT_CONCLUSIONS.md).

---

## 1. What this repository is

On periodic Burgers the spatial mean of the solution is conserved, so a model can look
competitive on relative L2 while predicting nothing beyond a conserved scalar. This project
separates operator-learning prediction error into a **conserved-mean part** and a
**fluctuation part**, measures skill on each against a **zero-fluctuation reference**, and
asks which part each architecture actually predicts.

Headline finding: on PDEBench 1D Burgers (`Nu0.001`, `t = 0 → 1.0`), MLP DeepONet scores
`-0.0042 ± 0.0072` and Fourier DeepONet `-0.0934 ± 0.0886` in per-sample fluctuation R²,
while FNO scores `+0.6503 ± 0.0222` on identical data, split and update budget. Across 175
reconstructed runs spanning trunk architecture, whitening, mean lifting, zero initialization,
learning-rate schedule, batch size and training-subset size, no DeepONet configuration scores
above zero.

**What is claimed:** a diagnostic decomposition and a reproducible negative result.
**What is not claimed:** a new architecture, a new optimizer, a theory, or a causal mechanism.

Six operator families — linear advection, Burgers at two viscosities, diffusion-sorption,
steady 2D Darcy, a three-field compressible system and a two-field reaction-diffusion system
— plus one physics-informed arm. See `paper/main.md` §5 for results and §6 for limitations.

---

## 2. ⚠️ An earlier programme in this repository is withdrawn

This repository began as **Basis-Align**, a study of trunk-basis alignment in PI-DeepONet on
synthetic PDEs. That line is **retracted** and is not evidence for the current paper:

- the "silent broadcasting bug" that dominated its results was fixed, after which the baseline
  trained normally;
- the central claim (*trunk-basis collapse is the root cause of PI-DeepONet failure*) does not
  survive that fix — see `paper/main.md` §7 and [`docs/AUDIT_V2.md`](docs/AUDIT_V2.md);
- the derived claims — a problem-independent variance floor, a linear law between alignment
  gain and baseline cross-seed standard deviation, and a modal-shrinkage rule — were each
  withdrawn or failed their predeclared confirmation.

The corresponding code is retained **for provenance only**:

| Legacy path | Status |
|---|---|
| `src/config.py`, `src/trainer.py`, `src/basis.py`, `src/diagnostics.py`, `src/data.py`, `src/pdes/` | synthetic-PDE basis-align pipeline, not used by the current study |
| `scripts/run.py`, `scripts/analyze*.py`, `scripts/ntk_analysis.py`, `scripts/inspect_subspace.py`, `scripts/make_paper_figs.py`, `scripts/*.cmd` | legacy drivers |
| `scripts/run_*_modal.py`, `scripts/audit_*_modal.py` | closed modal-shrinkage line (`docs/UNCERTAINTY_MODAL_V1.md`) |

Do not read numbers from those results directories as support for the current manuscript.
See `paper/main.md` §7 for the full withdrawal list.

---

## 3. Environment

```powershell
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Validated: Python 3.11.16, torch 2.14.0+cu130, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0.
`environment.yml` reproduces it; see also `requirements.txt` for a pip-only install.

> **CUDA note.** An environment built as `torch 2.14.0+cu126` is compiled up to `sm_90` and
> fails on `sm_120` (RTX 50 series) with
> `CUDA error: no kernel image is available for execution on the device`.
> Use the `cu130` environment, or pass `--device cpu`.

---

## 4. Repository layout

```
src/
  public_benchmark.py   data preparation + publisher digest pinning + protocol.json
  models.py             MLP / FourierFeature / DeepONet (trunk_type = mlp | fourier)
  fno.py                FNO1d          (supervised reference, 1D)
  fno1d_multi.py        FNO1dMulti     (multi-channel 1D, for the CFD system)
  fno2d.py              FNO2d          (2D Darcy)
  fno2d_multi.py        FNO2dMulti     (2D reaction-diffusion system)
  derivatives.py        trunk derivatives (vmap + jacfwd/jacrev) for the physics arm
  projection_audit.py   representation / coefficient orthogonal error split
  resume.py             random-state capture/restore, atomic checkpoint save
  health_checks.py      paired-shape contracts for samplers
  utils.py              seeding, relative-L2, environment fingerprint
scripts/
  download_pdebench.py  download_pdebench.py / download_darus.py / download_advection.py
  prepare_*.py          one per operator family; each writes a protocol.json
  run_public_benchmark.py   main supervised training entry point
  run_physics_public.py     physics-informed arm
  run_darcy_benchmark.py    2D Darcy entry point
  run_public_*.py           conditioning / small-fit / stability / POD / modal arms
  diagnose_fluctuation.py   recompute the mean/fluctuation decomposition from checkpoints
  skill_bound.py            trunk-span upper bound on fluctuation skill
  predict_skill_horizon.py  training-split-only predictors of skill
  measure_flops.py          per-step FLOP measurement
  experiment_runner.py      resumable manifest-driven batch runner
tests/                  ~26 files; no test requires CUDA
docs/                   30 experiment reports; start with CURRENT_CONCLUSIONS.md
paper/                  main.md (authoritative), ml4ps2026_4page.md (short draft), archive/
results/                ~130 versioned result families, one directory per experiment
```

---

## 5. The decomposition in one page

For targets `Y` and predictions `P` in `R^{m x N}`, write the per-sample spatial mean
`bar(y)_j` and the fluctuation `Y_c = Y - bar(y) 1^T`.

Because the constant vector and the zero-mean residual are orthogonal,

```
||p_j - y_j||^2 = N (bar(p)_j - bar(y)_j)^2 + ||p_{c,j} - y_{c,j}||^2
```

so `mean share + fluctuation share = E_j relL2_j^2` **exactly** (the test suite asserts the
closure term to be 0). Skill on the fluctuation is measured against a **zero-fluctuation
predictor**, which scores 0:

| Measure | Dominated by |
|---|---|
| `fluctuation_r2` (energy-weighted) | high-energy samples |
| `fluctuation_r2_per_sample` | every sample equally |

The manuscript's headline metric is a per-sample mean, so **the per-sample measure decides**
and the energy-weighted one is reported alongside. The two can disagree in sign: at 20000
updates MLP DeepONet reads `+0.2126` energy-weighted but `-0.2470` per sample.

A second ingredient is an **exact upper bound depending only on the trunk span** (grant the
branch a per-sample oracle over `span([Phi, 1])`). Comparing the bound with achieved skill
separates a trunk limitation from a branch limitation.

---

## 6. Reproducing

Data are **not** in Git (`external_data/` is git-ignored). Every file is byte-pinned, so
results are reproducible from the exact bytes listed in `paper/main.md` §4.

### 6.1 Obtain the data

```powershell
python scripts/download_pdebench.py   # 1D Burgers Nu0.001, 8.2 GB
python scripts/download_advection.py
python scripts/download_darus.py      # remaining families
```

The downloaders resume partial bytes and verify the publisher digest before finishing. If you
obtain a file manually, place it directly in `external_data/pdebench/`; the preparers verify
the digest and refuse to proceed on a mismatch.

### 6.2 Prepare a family

```powershell
python scripts/prepare_public_benchmark.py
```

writes `external_data/pdebench/fixed_t1_v1/` with `pairs.npz` and `protocol.json`
(source digest, split seed, sample ids, target time, spatial stride, normalization).
It refuses to overwrite an existing locked split. The other families have their own preparer
(`prepare_advection_horizons.py`, `prepare_pdebench_1d_horizons.py`,
`prepare_diffsorp_horizons.py`, `prepare_darcy.py`, `prepare_diffreact2d.py`,
`prepare_cfd.py`).

### 6.3 Train

```powershell
python scripts/run_public_benchmark.py `
    --data-dir external_data/pdebench/fixed_t1_v1 `
    --out results/public_fixed_t1_v1 `
    --steps 1500 --seeds 0,1,2,3,4 `
    --architectures deeponet_mlp,deeponet_fourier,fno `
    --device cuda
```

Each run writes `manifest.json`, `events.jsonl`, `progress.json`, a checkpoint and
`result.json`. Re-running the same command **skips completed runs** and resumes interrupted
ones; a run whose identity changed (configuration, source hash, data protocol or environment)
is refused rather than silently overwritten.

### 6.4 Diagnose

```powershell
python scripts/diagnose_fluctuation.py --device cuda
```

recomputes the decomposition from stored checkpoints into `results/fluctuation_v1/`. Every
recomputed `mean_relative_l2` is compared against the value stored in that run's
`result.json`; a run that disagrees beyond tolerance is reported as a failure. All
recomputations in the study reproduce their stored metrics exactly
(`max_abs_metric_difference = 0.00e+00`).

### 6.5 Test

```powershell
python -m unittest discover -s tests -v
```

No test requires CUDA. A handful read `results/` archives or git objects and skip when they
are absent.

---

## 7. Reproducibility guarantees

- **Data**: publisher SHA-256 / MD5 pinned per file; `src/public_benchmark.py` raises on
  mismatch, and consumers re-verify the prepared cache hash at load time.
- **Runs**: each run's identity is `configuration + source hash + data protocol +
  environment`, recorded in `manifest.json` together with the git commit; a changed identity
  is refused. Events are appended to `events.jsonl`.
- **Recomputation**: enforced, not assumed — diagnostics refuse to proceed if the pair-cache
  hash changed and report any metric that fails to reproduce.
- **Provenance of held-out data**: protocols and sign predictions were committed before the
  corresponding runs (`docs/*_V1.md`); the thresholds were never refitted on a held-out
  family.

---

## 8. Reading order

1. `paper/main.md` — the manuscript (§5 results, §6 limitations, §7 withdrawn claims).
2. `docs/CURRENT_CONCLUSIONS.md` — what is supported, what is retracted.
3. `docs/FLUCTUATION_DIAGNOSIS_V1.md` — the decomposition and its protocol.
4. `docs/REMAINING_GAPS_V1.md` — what is still open.
5. `docs/VENUE_STRATEGY_V1.md` — venue reasoning.
6. `docs/RUNBOOK.md` — rules for long, sleep-safe runs.
