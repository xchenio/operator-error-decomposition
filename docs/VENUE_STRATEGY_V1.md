# Venue strategy

Written 2026-10-08 for the current state of the manuscript (`paper/main.md`:
*Conserved Means and Lost Fluctuations*). This file records the reasoning behind venue
choice, the cost and review-time picture, and how to handle affiliation as a single
independent author who is now employed.

**Money and review-time figures are approximate and change.** Two are marked as confirmed on
2026-10-08; the rest must be re-checked on the journal's own page before submitting.

---

## 1. What the paper is, which fixes its tier

| Property | Consequence for venue choice |
|---|---|
| Diagnostic decomposition, not a new architecture | no main-conference novelty claim |
| Core result is negative (DeepONet loses the fluctuation) | only venues that accept negative results |
| Six settings, all from one benchmark suite (PDEBench) | below the bar for NeurIPS/ICLR/ICML main tracks |
| Strong reproducibility assets (byte-pinned data, audit scripts, pre-registration) | an asset at methodology/reproducibility venues, not counted at SOTA venues |
| No theory, no causal mechanism | rules out theory-leaning venues |

The `PUBLICATION_ASSESSMENT.md` judgement stands: **not a main conference**. The realistic
band is methodology/empirical journals and topical workshops.

## 2. Recommended venues

### Tier 1 — best fit, no author-side publication fee

| Venue | Type | Fee | First decision | Why it fits |
|---|---|---|---|---|
| **Journal of Computational Physics (JCP)** | journal, Elsevier | **confirmed 2026-10-08: "No publication fee charged to authors"** for the subscription route; open access costs extra | typically 3-6 months | The closest neighbour to this work, Williams et al. on what PI-DeepONets learn, is published here in 2026. The audience is exactly "machine learning applied to PDE problems". |
| **CMAME** (Comput. Methods Appl. Mech. Eng.) | journal, Elsevier | subscription route free (verify); OA extra | typically 3-6 months | The newly added spectral-bias reference [12] is published here, and CMAME carries operator-learning empirics. |
| **TMLR** | journal, rolling, OpenReview | reported to charge no publication fee — **verify on jmlr.org/tmlr before submitting** | first decision commonly 6-10 weeks; whole process 3-6 months | Explicitly welcomes negative results and careful empirical analysis. Double-blind. Single-author submissions are normal. |
| **JMLR** | journal | free | commonly 6-12 months | Free and respected, but slow and its scope is core ML. |

### Tier 2 — safer fallback

| Venue | Fee | First decision | Note |
|---|---|---|---|
| **Neurocomputing** (Elsevier) | subscription free, OA extra | typically 2-4 months | The "stable zone" identified earlier; easiest acceptance of the tier. |
| **PeerJ Computer Science** | low APC, membership options exist; **verify current price** | often 1-3 months | Explicitly accepts negative results. Cheap, but lower prestige. |
| **JMLMC** | verify | verify | Already listed as suitable in `PUBLICATION_ASSESSMENT.md`. |

### Tier 3 — workshops: fast feedback, but often more expensive overall

| Venue | Cost | Decision time | Caveat |
|---|---|---|---|
| NeurIPS / ICLR AI4Science, ML4PS and similar | submission free; **presenting normally requires conference registration (roughly USD 1,000+)** plus travel | 4-8 weeks | For someone optimising cost this can end up more expensive than a journal. Worth it only if attendance is reimbursed or remote presentation is allowed. |

**Not recommended:** NeurIPS/ICLR/ICML main tracks. The NeurIPS Datasets and Benchmarks
track normally expects the authors to *contribute* a benchmark or dataset; this study
analyses an existing one. Could be reconsidered for a future cycle only if the study grows
into a released benchmark artefact.

### Tier 1b — computational mathematics venues

Assessed 2026-10-08 on request. "Computational mathematics" spans a wide band, and the fit
varies sharply inside it.

**What this paper already has that is genuinely computational mathematics:**

- the mean/fluctuation split is an **exact orthogonal identity**, which can be stated and
  proved as a proposition (the constant vector and the zero-mean residual are orthogonal) —
  it is an a-posteriori error decomposition;
- the trunk-span bound is a **projection argument** giving a hard upper bound;
- the energy-weighted versus per-sample disagreement is a **norm-choice** issue, which is
  classical numerical-analysis territory.

**What it does not have, and what most computational-mathematics journals ask for:**

- no new numerical algorithm;
- no convergence analysis and no theorem.

At present the manuscript contains **no Lemma, Proposition, Theorem or Proof environment at
all**; the split is described as an "identity" and the bound as a "hard upper bound".

| Venue | Fit | Fee | Note |
|---|---|---|---|
| **SIAM J. Sci. Comput. (SISC)** | moderate, needs work | subscription route typically free (verify); OA extra | A real precedent exists: Lee and Shin's DeepONet training paper is published here (10.1137/23M1598751), so SIAM does carry DeepONet work. But SISC normally expects a new method or analysis, so the propositions below are needed first. |
| **SIAM J. Math. Data Sci. (SIMODS)** | moderate, needs work | verify | Right subject area — mathematics of operator learning — but expects mathematical depth. |
| **J. Sci. Comput.** (Springer) | moderate | subscription free / OA extra | Publishes ML-for-PDE work; lower bar than SISC. |
| **Appl. Numer. Math.**, **J. Comput. Appl. Math.**, **Comput. Math. Appl.** | moderate, broader | subscription free / OA extra | Broader applied-mathematics scope, more tolerant of empirical content. |
| **Adv. Comput. Math.** | weak | verify | Theory-leaning; unlikely without new analysis. |
| **SINUM**, **Math. Comp.**, **Numer. Math.** | poor | verify | Pure numerical analysis expecting new algorithms with proofs. Not suitable as the paper stands. |

**Before submitting to any of these, three changes are required:**

1. State §3's split as **Proposition 1** with its two-line proof.
2. State §5.7's trunk-span bound as **Proposition 2** with its projection proof.
3. **De-emphasise the empirical threshold rule** (§5.8). At an ML venue the 52-of-55
   cross-family record is a highlight; at a computational-mathematics venue an unproved
   empirical rule is a liability. Move it to an appendix or label it explicitly as an
   empirical boundary and not a theorem.

Also retitle away from a leaderboard reading ("who is better") toward an error-decomposition
reading, for example *An Exact A Posteriori Error Decomposition for Operator Learning:
Conservation, Fluctuation and the Limit of Branch Capacity*.

### Cheapest realistic path in China

Chinese core journals charge a much smaller page fee (order of a few hundred to a couple of
thousand RMB) and some charge nothing, with review on the order of 1-3 months. Worth
considering only if international visibility is not a priority.

## 3. Recommended sequence

1. **Post an arXiv preprint first.** Free, establishes priority, and gives a citable
   reference while the journal review runs.
   - Categories: `cs.LG`, with `math.NA` as cross-list.
   - **Practical obstacle for an unaffiliated author**: arXiv may require endorsement for a
     first submission in a category. Arrange an endorser, or see below.
2. **Primary submission: JCP.** Best scope match, no fee, good standing in this community.
3. If rejected: **CMAME**, then **TMLR**, then **Neurocomputing**.
4. In parallel, submit the 4-page version (`paper/ml4ps2026_4page.md`) to a workshop only if
   registration is covered or remote presentation is allowed.

## 4. Single author, no supervisor

This is **not a problem at any journal or conference**. No venue requires a supervisor, and
single-author papers are routine. The only real friction points are practical:

- **arXiv endorsement** without an institutional email. Start early; it can take time.
- Some submission systems nudge for an institutional address. A personal address is
  accepted in practice.
- With one author, "author contributions" statements are trivial; some systems still ask.
- Consider **ORCID** (free, immediate). For an independent researcher it is the main way to
  establish identity, and most Elsevier and Springer systems ask for it.
- Deposit code and data under a personal GitHub plus a **Zenodo DOI** (free). It makes the
  work citable independently of the publisher.

## 5. Affiliation while employed

**Recommended: publish as an independent researcher, not under the employer.**

```
Author:      <name>
Affiliation: Independent Researcher, <city>, <country>
Email:       a long-term personal address (not the employer address)
ORCID:       0000-0000-0000-0000
```

Add two short statements to the manuscript:

> **Funding.** This work was carried out independently by the author, outside employment
> duties, without institutional or employer resources. No funding was received.

> **Competing interests.** The author declares no competing interests. The author's employer
> had no role in the design, execution, or writing of this work and provided no funding,
> facilities, or approval.

### Why not list the employer

- Most employment contracts contain intellectual-property clauses covering work related to
  the business. Naming the employer can trigger a disclosure or approval requirement, or an
  ownership dispute.
- A PDE/operator-learning diagnostic study is very likely unrelated to the employer's
  business, so naming them buys nothing and adds risk.

### Why not list the former university

- You are no longer there. Listing an institution without its consent is not appropriate.
- If part of the work genuinely used university resources (for example the data was
  downloaded while enrolled), say so in the acknowledgements — "part of the data was
  acquired while the author was at X" — but do **not** put the university in the
  affiliation line.

### If you want institutional backing anyway

The only clean route is to ask: contact a former supervisor or the department, explain the
paper, and ask whether they are willing to be listed (as affiliation or as a co-author).
Do not add a co-author who has not agreed and has not seen the manuscript; inventing or
implying authorship is a serious problem and is not worth the risk.

## 6. Before submitting, re-check

- [ ] Confirm the current APC and page-charge policy on the journal's own page.
- [ ] Confirm the current editorial scope still covers operator-learning empirics.
- [ ] Confirm no new version of the PDEBench deposit appeared
      (`docs/PDEBENCH_DATA_VERSIONS_V1.md` gives the query).
- [ ] Confirm the manuscript's data-version statement still matches the deposit.
