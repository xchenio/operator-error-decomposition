# Experiment ledger

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
`events.jsonl` while running, and receive a terminal `completed`, `interrupted`, or `failed`
status. A result is never silently overwritten; resumed work uses the same manifest and skips
completed configuration keys.
