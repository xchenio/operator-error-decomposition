# Manuscript revision

## 2026-09-29 — v3: mean/fluctuation rewrite

Current authoritative text: main.md. Short text: ml4ps2026_4page.md.
Pre-rewrite text archived at archive/pre_fluctuation_20260929/main.md;
pre-audit text at archive/pre_audit_20260924/main.md.

| Previous statement | Current treatment |
|---|---|
| "Input spatial-mean baseline" as a weak reference | Reidentified as a **conservation-law predictor**: corr(mean(x), mean(y)) = 1.000000; the fluctuation carries 23.7% of the per-sample squared norm. |
| A dissociation between representability, fitting and generalization | Retained but **localized**: the unmodelled component is the deviation from the conserved mean; measured with an exact orthogonal squared-error split. |
| DeepONet fails to beat a baseline | Replaced by a signed skill measure with a defined zero point (zero-fluctuation reference) and two weightings. |
| R^2 (energy-weighted) alone | Replaced by **both** weightings. The energy-weighted measure reads +0.2126 for MLP at 20k updates where the per-sample measure reads -0.2470; the per-sample one decides, because the headline metric is per-sample. |
| A modal shrinkage rule under development | Line closed: 0 of 12 confirmation comparisons against strong controls passed. Negative result recorded, not argued away. |
| Fourier trunk repairs the public-data failure | Qualified by horizon: +0.4248 versus +0.1695 at t=0.25, but -0.0934 versus -0.0042 at t=1.00. |
| Single 1500-update budget | Step ladder 1500 / 5000 / 20000 added: total relative L2 flat, per-sample fluctuation skill degrades. |

Evidence base: 175 reconstructed runs of the original four families plus 45 new runs
(5k, 20k, two horizons). Every recomputed metric reproduces its stored value exactly
(`max_abs_metric_difference = 0.00e+00`). Manuscript tables are now checked against the
recomputed JSON by `tests/test_fluctuation_diagnosis.py`, so they cannot drift from the
artifacts.

Still open, and stated as open in the manuscript: compute- or FLOP-matched FNO control, a
second public PDE with a reserved holdout, and any physics-informed claim (this study is
supervised throughout).

---

# Manuscript revision, 2026-09-24

Current authoritative text: main.md. Short text: short_paper.md.
ml4ps2026_4page.md and ml4ps2026.tex retain legacy filenames, not a submission claim.
No publication, acceptance, official template verification or PDF compilation is asserted.

| Previous statement | Current treatment |
|---|---|
| test relative L2 minus a truncated captured-energy proxy is pure optimization error | Replaced with same-grid squared Frobenius projection identity and attribution caveats. |
| A low supervised value error must imply lower PDE residual than the trivial predictor | Withdrawn: derivative error is not controlled by value error; sampler contracts are direct checks. |
| Alignment mechanism confirmed by output NTK/sharpness correlations | Historical exploratory observation, not current causal claim. |
| Problem-independent variance floor and pooled historical significance | Removed as active evidence; unknown source fingerprints remain unknown. |
| Fourier trunk repairs the public-data failure | Contradicted by initial public controls; better oracle does not imply better predictions. |
| Stable convergence improves validation | Replaced by the observed dissociation, including the MLP/full-batch exception. |
| New physics-informed method / novel orthogonalization | Current public study is supervised; prior two-step and POD work acknowledged. |

Completed public records: 15 + 40 + 60 + 40 = 155, including 10 repeated controls.
Do not count these as independent data partitions or as 155 novel method evaluations.
Publication gap assessment and proposed next work are in ../docs/PUBLICATION_ASSESSMENT.md.
