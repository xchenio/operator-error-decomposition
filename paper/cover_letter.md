# Cover letter

**Manuscript:** *An Exact A Posteriori Error Decomposition for Operator Learning:
Conservation, Fluctuation, and the Limit of Branch Capacity*

**Submitted to:** Journal of Computational Physics

**Type:** Research paper. Not a review, not a benchmark/leaderboard submission.

> [confirm before sending] Standard declarations: the work is original, has not been published,
> and is not under consideration elsewhere. An earlier draft existed under a workshop-specific
> filename and an anonymous code bundle was prepared from it, but the manuscript was not
> submitted in that form; please adjust this sentence to match what actually happened.
>
> The relation of this manuscript to the author's 2025 undergraduate thesis is disclosed in
> full below and in the manuscript; nothing in it is reproduced from that thesis.

---

Dear Editors,

Operator learning is usually evaluated by a single scalar: relative L2 on a held-out split.
This manuscript argues that the scalar is not diagnostic, and supplies a decomposition that is.
On periodic Burgers the spatial mean is conserved, so a model can sit near the best reported
error while predicting nothing beyond a conserved number. We split the squared error exactly
into a conserved-mean part and a fluctuation part (**Proposition 1**), give a hard upper bound
on fluctuation skill that depends only on the trunk span (**Proposition 2**), and use both to
localise where the loss occurs. We claim a decomposition with two short proofs, a reproducible
negative result, and a localisation — not a new architecture, not a new optimizer, and not a
theorem for the empirical regularity in section 5.8.

Four things deserve the reviewers' attention.

## 1. The result is negative, and it is localised rather than merely reported

Both DeepONet variants reproduce the conserved mean and have no per-sample fluctuation skill at
forecast time 1.0 on PDEBench 1D Burgers: MLP DeepONet scores −0.0042 ± 0.0072 and Fourier
DeepONet −0.0934 ± 0.0886 against a zero-fluctuation reference, while FNO scores +0.6503 ±
0.0222 on identical data, split and update budget. This is not undertraining: across a 33×
budget range the total relative L2 of MLP DeepONet never improves
(0.39357 / 0.39729 / 0.39311 / 0.42075) while the per-sample fluctuation measure degrades
monotonically. Nor is it a trunk-basis failure: the Fourier trunk's bound is +0.96 to +1.00 at
every horizon, including t = 2.00 where the achieved value is −0.3774. The reachable set does
not shrink with horizon; the map onto it does. The binding constraint is the branch.

This qualifies rather than contradicts Williams et al. (JCP 2026, doi:10.1016/j.jcp.2026.114851),
the closest neighbour to this work and one of the reasons JCP is the right venue: they find that
learned physics-informed bases fail to span the solution space, and we localise the residual
failure in the input-to-coefficient mapping, with the horizon at which trunk features matter
made explicit (the Fourier trunk scores +0.4248 against +0.1695 at t = 0.25 and −0.0934 against
−0.0042 at t = 1.00).

## 2. The predictive record is reported stratified, not as a headline fraction

An a-priori fluctuation-energy threshold, fitted on two scalar families and never refitted,
predicted the sign of fluctuation skill on held-out settings: 52 of 55 preregistered sign
predictions correct. We do not ask the reviewers to take that number at face value, because the
55 are not equally informative. Section 5.16 and `experiments/prediction_ledger.json` stratify
them:

| stratum | n | correct | accuracy | 95% Wilson CI |
|---|---:|---:|---:|---|
| all | 55 | 52 | 94.5% | [85.1, 98.1] |
| saturated (`R^2 >= 0.9`, cannot discriminate) | 25 | 25 | 100% | [86.7, 100] |
| informative | 30 | 27 | 90.0% | [74.4, 96.5] |
| boundary (within 15% of the threshold) | 6 | 5 | 83.3% | [43.6, 97.0] |
| resolvable (`|R^2| > 2 sd`) | 47 | 46 | 97.9% | [88.9, 99.6] |

Twenty-five of the fifty-five are on settings where the architecture is essentially perfect and
the sign is not in doubt. On the thirty that can discriminate, the rule scores 27/30 against
16/30 for an always-positive baseline (CI [36.1, 69.8] versus [74.4, 96.5]) — the only
comparison that matters, and the one a skeptical reader will ask for. The ledger is
machine-readable, each prediction is recomputed from the rule by script so a transcribed sign
cannot drift, and the strata are recomputed by test.

## 3. Three self-falsifications are recorded, and nothing was adjusted in response

- The threshold rule **fails on the three coupled fields of 1D compressible CFD**: 3 of 6, with
  every prediction of positive skill wrong. The thresholds were not refitted.
- The physics-informed arm was preregistered to fail at Burgers `T = 0.5`, a horizon where
  supervised training succeeds. It **did** reach positive skill (+0.0088, +0.0516). The
  prediction is reported as falsified and the loss weights and the fitted coefficient were not
  adjusted.
- The CFD failure initially prompted narrowing the claim to "scalar operators only". A second
  system, 2D reaction-diffusion, scored 4 of 4 and **showed that narrowing was too strong**; the
  claim was corrected again, in the manuscript, rather than quietly reverted.

A fourth line, modal shrinkage, was closed after failing its own predeclared confirmation
(0 of 12 comparisons against strong controls) and is recorded as a negative result rather than
being argued away. Section 7 lists every withdrawn claim from earlier drafts of this project.

## 4. Reproducibility is enforced, not asserted

- Every input file is byte-pinned (SHA-256 / MD5 with publisher revision identifiers); the
  runner refuses to proceed if a digest changes.
- Each run carries an identity hash over its configuration, source files, data protocol and
  environment; replaying with a different identity is refused rather than silently overwritten.
- Every number in sections 5.1-5.5 was recomputed from stored checkpoints. All recomputations
  reproduce their stored metrics exactly (`max_abs_metric_difference = 0.00e+00`).
- Protocols for all six held-out families were committed before their runs; the commit hash of
  each preregistration document is recorded in the prediction ledger.
- The test suite (95 tests) asserts the squared-error closure, the mean-invariance of the skill
  measure, and that the manuscript's tables equal the recomputed artifacts.

**One limitation is being actively addressed rather than defended.** Every number comes from one
data split whose test set was exposed during exploratory development. A replication on a second
split, disjoint from the first and with the thresholds refitted rather than carried over, is
predeclared in `docs/REPLICATION_SPLIT2_V1.md` with its endpoints and tolerances fixed in
advance. The protocol was committed before any run on split 2; the runs are in progress and the
result will be reported whatever it is. Reviewers may read that document as evidence of how the
study is conducted.

## 5. Disclosure: relation to the author's prior thesis

For full transparency before review: part of this line of work began in the author's
undergraduate thesis (*Research and Implementation of DeepONet-Based Methods for Solving Partial
Differential Equations*, Harbin Institute of Technology, May 2025). That thesis studied a
physics-informed DeepONet on a **self-generated** one-dimensional diffusion-reaction problem,
used **total relative L2** as its only reported measure, and concluded that adding a residual
loss **improved** accuracy (4.80e-3 on its test set).

The submitted manuscript shares only the broad topic and standard background with that thesis.
It differs in data source (byte-pinned PDEBench deposits downloaded in 2026 rather than a
self-written finite-difference solver), in the PDEs studied (six families, none of which appears
in the thesis), in the reported measure (an exact mean/fluctuation decomposition rather than a
single scalar), and in conclusion: where the thesis reported improvement, section 5.11 localises
the physics-informed deficit and finds that at most 0.38 of supervised fluctuation skill is
recovered. The concepts that carry this paper — the conserved-mean decomposition, the
trunk-span bound, the skill horizon, the fluctuation-energy threshold — are absent from the
thesis. No figure, table, experiment or number is reused.

The editors may therefore treat this as new work. It is disclosed rather than hidden because the
earlier framing is part of what this paper corrects, and that is worth the reviewer knowing.

## Relation to the venue and suggested reviewers

JCP publishes the two nearest neighbours to this work — Williams et al. on what
physics-informed DeepONets learn, and Lee and Shin (SIAM J. Sci. Comput.) on DeepONet training —
and its readership is exactly "machine learning applied to PDE problems". The contribution is an
a-posteriori error decomposition with two proved propositions and an a-priori computable
statistic, which is classical numerical-analysis territory, rather than a state-of-the-art
accuracy claim.

Conflicts of interest: none suggested — [the authors of refs 2, 3 and 11 are the closest
neighbours and the editors may prefer to avoid them].

## Practical

- Single author, publishing as an independent researcher; no funding was received and no
  employer resources were used. Declarations of competing interest, funding, data availability
  and author contribution are all in the manuscript, before the references.
- The manuscript's acknowledgment names no individual. COPE guidance is to ask before listing
  someone, and an acknowledgment can read as endorsement, so the undergraduate-thesis
  relationship is disclosed in the "Relation to the author's prior thesis" section without
  naming the supervisors.
- Data: the PDEBench deposits are too large to redistribute; the manuscript gives
  publisher-checked digests and the repository provides download scripts, so every result is
  reproducible on the exact bytes used. [confirm before sending] The code is released under MIT at
  https://github.com/xchenio/operator-error-decomposition with a Zenodo DOI, and an arXiv preprint
  (`cs.LG`, cross-listed `math.NA`) is being prepared — confirm the repository is public and the
  DOI is live before sending.

Thank you for considering this manuscript.

Xiaochen Yan
ORCID: 0009-0002-9280-4323
Independent Researcher, Shenzhen, China · xiaochen-yan@qq.com
