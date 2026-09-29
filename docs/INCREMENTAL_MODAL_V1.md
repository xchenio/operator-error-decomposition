# Incremental nonlinear benefit beyond a tuned linear map

Prospective protocol, 2026-09-29. Prior objective-modal results remain unchanged.

- Reuse the three tasks and their still-unopened500-ID confirmation sets. Independent
  Burgers training IDs remain disjoint; Advection remains a linear negative control.
- Freeze each existing POD rank16 basis from representation400. On correction600,
  use3 outer folds. Each outer training400 tunes its own linear coefficient map with
 3 inner folds: ordinary and guarded energy-weighted ridge, alpha0 (truncated SVD
  pseudoinverse) plus10^-8 through10^8. Intercepts are unpenalized and correctly
  centered, normalization and energy floors fit inner training only. Choose mean
  held relative L2; shared alpha across modes. This is extensive finite-grid tuning,
  not proof of an optimal linear model. Report boundary selections.
- Train a zero-output width32 tanh residual over each outer tuned frozen linear map,
  normalized objective, correction penalty1, AdamW decay1e-4, cosine and batch32.
  Two seeds/fold, fixed1500 updates: held folds never choose early stopping. Preserve
  optimizer/RNG every50 updates using the already tested residual core.
- Compute each mode's incremental weighted squared-error gain over that tuned linear
  map on held folds. Hard rule: all3 gains positive and mean>=1%. Numerically negligible
  baseline modes cannot pass. Continuous rule: clip pooled weighted residual-dot-delta
  divided by pooled delta-square to[0,1], averaging seeds before pooling folds. No
  development tuning of either rule. This is nonlinear-branch incremental gain over
  this comparator, not proof of intrinsically nonlinear physics.
- Tune a final linear map on correction600 with3 inner folds; train5 residual seeds
  for the same fixed1500 steps. This round intentionally uses fixed-step fitting rather
  than development early stopping so all posthoc masks share identical networks.
  Compare none, all (diagnostic), stable, leading-energy k and continuous shrinkage on
  the same predictions. k comes from the hard rule. No arm-specific retraining.
  Thus33 neural fits total (18 CV probes+15 final), not75 independent arm fits.
- Development chooses stable versus shrink (stable wins exact ties), and falls back
  to none unless mean relative L2 improves at least1%. All5 seeds retained, no seed
  selection. Lock each candidate, every checkpoint, linear model and mask in Git
  before loading any reserved target. No learning/tuning after opening confirmation.
- Primary per task: selected candidate versus tuned linear/none, at least1% relative
  mean gain AND paired trajectory bootstrap interval wholly below zero. Average5 seed
  errors per trajectory, not errors of an ensemble prediction. Secondary rule-value
  claim additionally requires the same criterion versus energy. Six comparisons total,
  Bonferroni two-sided intervals with tail probability.05/12,5000 bootstrap replicates,
  fixed seed20260929. RMSE is descriptive and cannot rescue primary failure. Intervals
  condition on trained seeds and do not establish cross-PDE universality.
- Confirmation data extraction checkpoints every50 trajectories; opening marker makes
  consumption explicit even after a crash. Resume locked evaluation only. Git tracks
  JSON evidence, not local datasets or full binary checkpoints.

Commands use D:/miniconda3/envs/pytorch/python.exe:

```text
scripts/run_incremental_modal.py plan
# commit protocol/source
scripts/run_incremental_modal.py develop
scripts/run_incremental_modal.py seal
# commit development and selection_lock.json
scripts/run_incremental_modal.py confirm
```

No result or methodological novelty is presumed by this protocol.

## Development results, before confirmation

All33 fits completed (49500 updates). Independent checkpoint replay verifies nested
folds, frozen linear models, coefficient risks, masks and development predictions.

| Rule | Original Burgers | Independent Burgers | Advection |
|---|---:|---:|---:|
| None / tuned linear | 0.373252 | 0.409382 | 0.019756125100 |
| All nonlinear outputs | 0.363277 | 0.403154 | 0.019756122297 |
| Stable mask | 0.362262 | 0.402368 | 0.019756125076 |
| Energy mask | 0.362262 | 0.402368 | 0.019756125159 |
| Continuous shrinkage | 0.362297 | 0.402073 | 0.019756122211 |
| Locked candidate | stable | shrink | none |

Final linear selections: MSE/alpha100, MSE/alpha10000, normalized/alpha1e-6.
Original Burgers outer fold1 selects alpha1e8, the upper boundary: finite-grid tuning
must not be described as a globally optimized linear oracle. Hard masks select zero-based
modes0–3,0–1 and1/7 respectively. On Burgers they still equal energy truncation.
Advection coefficient-level gains are too small to matter for the field metric; the
whole-field1% fallback correctly declines correction despite two positive modal probes.
Shrinkage weights are stored in rules.json; none are tuned using confirmation labels.

Before confirmation:37 tests pass; three reserved sets still unopened. Selection hashes
will be committed before extraction. The table is developmental, not primary success.
