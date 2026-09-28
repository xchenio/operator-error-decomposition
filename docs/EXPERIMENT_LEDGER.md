# Experiment ledger

> 2026-09-22 authority update: older rankings and statistical claims below are historical.
> AUDIT_V2.md supersedes their error-decomposition and provenance interpretation.
> New public-data evidence is indexed below and must not be pooled with legacy runs.

| New family | Protocol / evidence | Limits |
|---|---|---|
| `public_objective_modal_v1` | 123 completed neural records +48 ridge grid refits; normalized loss helps two Burgers development splits but hurts linear Advection; strict internal-CV mask comparisons | 25 redundant panel records; Burgers masks equal energy selection, Advection selects all16. Tuned Advection ridge beats residuals. Three new confirmation sets unopened. OBJECTIVE_MODAL_V1.md. |
| `audit_v2` | Full-grid checkpoint projection recomputation and regrouped historical statistics | Unknown historical source hashes remain unknown. |
| `public_fixed_t1_v1` | 15 completed matched supervised runs; 3 architectures, 5 seeds | One PDE/subset/time, equal updates not equal compute; DeepONet below initial-mean baseline. |
| `public_coefficients_v1` | All 40 runs completed; small average whitening gains, all arms below initial-mean baseline; PUBLIC_COEFFICIENTS_V1.md | Extra branch training and relaxed constant term; same exploratory test split, no final-holdout claim. |
| `public_smallfit_v1` | 60 completed runs; n=32 zero-init fits oracle in 10/10 seeds; initialization improves validation in 29/30 pairs | Validation only, pretrained trunk/normalization; not few-shot. Larger subsets not sufficiently fit; no baseline superiority. |
| `public_stability_v1` | 40 completed n=128 zero-init runs; cosine decay stable-tail fit in 20/20, no general validation improvement | Equal updates not equal samples/compute; pretrained trunk, validation only. Smallfit inability to fit n=128 was optimizer/budget dependent. |
| `public_scale1000_v1` | 20 completed controls; cosine improves validation relative L2 in 10/10 pairs | Includes 10 old-control repeats; training still not fully fit; no new test evaluation. |
| `public_pod_ridge_v1` | 18 deterministic training-only residual-POD ridge candidates; selected validation error 0.373656 | Validation-tuned grid boundary, tiny baseline gain; not full POD-DeepONet or two-step reference. |
| `public_pod_readout_v1` | 30 completed same-POD neural runs; nonlinear improves training in 15/15 but worsens validation in 15/15 versus Adam affine control | Exact rank-matched POD/normalization; reused validation only; coefficient error is not uniquely optimization error. |
| `public_ridge_residual_v1` | 40 neural runs +8 ridge refits; sealed selection evaluated on500 new trajectories; primary gain not confirmed, RMSE improves5/5 | Baseline-inclusive early stopping is not an unseen-data guarantee; confirmation IDs are now consumed; no post-confirmation retuning. |

This is the permanent index of experiment families. Raw per-run JSON, logs, checkpoints, and
figures remain versioned under `results/`. A run is evidence only when its source fingerprint,
configuration, and status are recorded.

| Era / result family | Status | What it supports | What must not be reused |
|---|---|---|---|
| `expA`--`expJ`, early DR/Burgers alignment experiments | superseded | Reproduction of the original failure symptoms | Any pre-2026-09-16 representation or accuracy conclusion; these used the per-sample pairing bug. |
| `expK`--`expP`, corrected 1D controls | historical / partly superseded | Causal weighting and L-BFGS negative controls; early alignment/Fourier comparison | Single-seed rankings as statistical claims. |
| `expQ`, NTK checkpoints | active evidence | Paired output-NTK and sharpness mechanism measurements | A universal mechanism claim; the mechanism sample is small. |
| `expS`, `expT`, resolution/width sweeps | historical | Rank and resolution probes | The retracted rank-to-gain linear relation. |
| `expU`, early 2D | superseded | Development trace only | Incomplete seed groups or early 2D conclusions. |
| `expV`, five-seed phase diagram | authoritative | Main 1D and baseline/alignment 2D summaries in `results/final_stats.txt` | Uncorrected p-values as multiplicity-adjusted evidence. |
| `expW`, 2D intervention controls | authoritative | Fourier, random alignment, analytic alignment, orthogonalization, and IC-weight controls | Alignment-necessity claim. |
| `expDO*`, supervised references | active evidence | Budget-aware supervised reference curves | A claim of physics superiority: 50k physics comparisons have limited seeds. |
| `rep1d*`, `rep2d*` | active evidence | Bit-identical repeatability under a fixed code state | Historical code-state comparisons without matching hashes. |

## Statistical interpretation

- The ten-cell phase table has five seeds per cell. Eight uncorrected Welch tests are below 0.05;
  six survive Holm--Bonferroni correction. Both numbers must be reported together.
- The 20k/50k physics rows are exploratory unless and until multi-seed repeats are added.
- `results/phase_analysis.json` and `results/final_stats.txt` are the current summary artifacts;
  individual JSON files are the primary records.

## Change-control rule

Every new experiment directory must contain a `manifest.json` before training, append events to
`events.jsonl` or persist atomic `progress.json` plus checkpointed evaluation history while
running, and receive a terminal `completed`, `interrupted`, or `failed`
status. A result is never silently overwritten; resumed work uses the same manifest and skips
completed configuration keys.
