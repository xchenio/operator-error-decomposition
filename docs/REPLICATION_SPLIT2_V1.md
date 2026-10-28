# Independent replication on a second, disjoint data split

Date: 2026-10-28. This protocol is committed **before** any run on split 2.

## 1. What this closes, and what it does not

The manuscript's most frequently attackable limitation is stated in section 6:

> The test split was exposed during exploratory development and validation guided later
> study choices.

Every number in the paper comes from one split: seed `20260922`, sizes 1000/200/200, reused
across all families so that trajectory ids match. The preregistration discipline of the later
families (each protocol committed before its runs) protects the *rule* from hindsight, but it
does not protect the *split* from it. Thresholds were fitted on the Burgers Nu0.001 and
advection families of that same split.

This protocol repeats the whole procedure on a **second split that shares no trajectory with
the first**, with the thresholds refitted on the new split rather than carried over. It closes
the single-split limitation.

It does **not** close: the absence of an out-of-operator boundary setting (no file in the
deposit provides one; `DARCY_BETA_LADDER_V1.md`), the compressible-CFD failure (no second
compressible parameter set is available at a usable size), or the missing shallow-water
family. Those remain reported as data limitations.

## 2. Split 2

| property | value |
|---|---|
| seed | **20261028** |
| sizes | 1000 train / 200 val / 200 test (unchanged) |
| disjointness | drawn from the complement of **all 1400 ids** used by split 1 (seed 20260922) |
| source files | identical, byte-pinned, already downloaded — **no new download required** |

Disjointness is deliberate. With 1400 of 10000 trajectories drawn per split, two independent
seeds would be expected to share about 196 ids, so a bare new seed is not an independent
sample. Drawing from the 8600 never used by split 1 makes every trajectory in split 2 unseen
by split 1, including its test set.

Sampling: `pool = setdiff(range(10000), split1_ids)`, then
`np.random.default_rng(20261028).permutation(pool)[:1400]`, split in order into
train/val/test. Deterministic given seed and exclusion set; both are recorded in
`protocol.json` (`split_seed`, `excluded_seed`, `disjoint_from_split1`, `sample_ids`).

**Verification before any training:** assert the intersection of split-1 and split-2 id sets is
empty, per split and overall. This is a hard preflight check, not a post-hoc note.

Required code change, to be implemented **after** this protocol is committed: add
`--split-seed` and `--exclude-seed` to `scripts/prepare_pdebench_1d_horizons.py` and
`scripts/prepare_advection_horizons.py`, defaulting to the current behaviour
(`seed=20260922`, no exclusion) so every prepared cache and its committed record are
unchanged. Neither script appears in any frozen `source_hash()` list, so this does not
invalidate a single existing run identity.

## 3. Design: what is refitted and what is held out

The replication repeats the *procedure*, not the number. That is the point.

| role | families | purpose |
|---|---|---|
| **fitting set** | Burgers Nu0.001, 7 horizons; linear advection, 7 horizons | refit the three thresholds from scratch on split 2 |
| **held out** | Burgers Nu0.01, 7 horizons | 21 sign predictions (7 horizons x 3 architectures), thresholds applied unchanged |

The split-1 thresholds (0.2477 MLP, 0.2728 Fourier, 0.1502 FNO) are **not** applied to split 2.
They are recorded here only so that endpoint P1 can compare against them. Reusing them would
test the number, not the rule.

Budget, seeds, grid and architectures are identical to every other family: Adam 1e-3, batch 32,
**1500 updates**, seeds 0-4, MLP DeepONet / Fourier DeepONet / FNO, stride 4 to 256 points.
Normalization statistics are computed from split-2 training arrays only.

Cost: 21 families x 15 runs = 315 runs. At the measured 1500-update cost (MLP 6 s, Fourier 6 s,
FNO 14 s) that is roughly 1.5-2 hours including diagnostic recomputation.

## 4. Predeclared endpoints

Endpoint **R0** is primary because under the JCP framing the manuscript's claim is the
error decomposition and the negative result, not the threshold rule. P1 and P2 are secondary
and are reported even if R0 fails.

### R0 (primary): the dissociation replicates

On split 2, Burgers Nu0.001, `t = 1.0`, test split, per-sample fluctuation R^2:

| condition | pass band |
|---|---|
| MLP DeepONet | within `[-0.10, +0.05]`, i.e. still indistinguishable from the zero-fluctuation reference |
| FNO | `>= +0.50` |
| FNO - MLP gap | `>= +0.50` |

The band for MLP is asymmetric on purpose: the claim is "no better than predicting zero
fluctuation", so a materially *positive* value falsifies it just as a large negative one does,
while a small negative value is consistent with it. Section 5.2 on split 1 gives
-0.0042 +- 0.0072 (MLP) and +0.6503 +- 0.0222 (FNO).

### P1 (secondary): the thresholds are a stable quantity

For each architecture, the split-2-refit threshold must lie within **25% relative** of the
split-1 value, **and** the ordering `FNO < MLP < Fourier` must be preserved:

| architecture | split 1 | acceptance band |
|---|---:|---:|
| MLP DeepONet | 0.2477 | [0.1858, 0.3096] |
| Fourier DeepONet | 0.2728 | [0.2046, 0.3410] |
| FNO | 0.1502 | [0.1127, 0.1878] |

25% is a judgement call, not a derived number: the threshold is a linear interpolation between
two horizons roughly 0.05 apart in fluctuation energy, so sampling noise of a few hundredths is
expected, and a rule with no substance would drift much further than this. The tolerance is
fixed here so it cannot be widened after seeing the values.

### P2 (secondary): held-out sign accuracy replicates

On the 21 held-out Burgers Nu0.01 predictions: **at least 18 of 21** correct, and strictly more
than the always-positive baseline computed on the same 21. (Split 1 gave 21 of 21.)

Strata are reported by `scripts/analyze_prediction_ledger.py` with the same cuts as section
5.16 — saturated (`R^2 >= 0.9`), boundary (within 15% of the threshold) and resolvable
(`|R^2| > 2 sd`) — so the split-2 record is stratified the moment it exists rather than being
reported as a bare fraction. A new ledger file is written for split 2; split-1 entries are not
merged with it.

## 5. Falsification, fixed in advance

1. If R0 fails, **the core negative result does not replicate** on independent data and is
   reported as such. Section 6's limitation is upgraded from "one split" to "one split; a
   disjoint replication did not reproduce the headline dissociation", and the claim is narrowed
   accordingly. It is not re-expressed as a partial success.
2. If P1 or P2 fails, the threshold rule is reported as **not replicated**. The thresholds are
   not refitted, reweighted or re-derived to obtain a pass. A new fit would be a new protocol
   with its own held-out test.
3. **No endpoint is swapped after the fact.** Following `RIDGE_RESIDUAL_V1.md`: a failed
   primary is not replaced by a secondary that happens to pass.
4. Every outcome is written into the manuscript, including unfavourable ones. If the
   replication fails, the paper is *more* honest for it, not less publishable.

## 6. Known limitations, stated in advance

1. **One operator class at the boundary.** As on split 1, the only held-out family that
   straddles a threshold is Burgers at a second viscosity. This replication removes the
   single-split weakness; it does **not** create an out-of-operator boundary test, because no
   file in the deposit provides one. That gap is unchanged by this experiment and must not be
   described as closed.
2. **Same solver, same initial-condition generator, same grid.** Split 2 reshuffles
   trajectories, not the physics. Variation across independent *data-generating processes* is
   still untested.
3. **A disjoint split is not an exchangeable resample.** Conditioning on split 1's complement
   is deliberate and is what makes the test independent, but it means split 2 is not a
   bootstrap replicate of split 1; the two are paired, and agreement between them is weaker
   evidence than agreement between two fully independent datasets would be.
4. **Seed standard deviations still do not measure task variability** — five seeds share one
   split, on split 2 as on split 1.
5. The advection fitting family contributes no threshold crossing (its fluctuation energy is
   constant at 0.6446), exactly as on split 1. It is retained for procedural fidelity, not
   because it constrains the thresholds.

## 7. Commands

Run through `scripts/experiment_runner.py` with `--resume`; the machine may sleep, and the
runner restarts only jobs lacking their completion artifact.

```bash
# 0) preflight, before anything else: split 2 must share no trajectory with split 1
python scripts/verify_split_disjointness.py \
    --seed-a 20260922 --seed-b 20261028 --sizes 1000,200,200 --total 10000

# 1) prepare split-2 caches (one directory per horizon)
python scripts/prepare_pdebench_1d_horizons.py \
    --source external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5 \
    --out external_data/pdebench/rs2/burgers_nu0001 \
    --equation "1D Burgers, Nu=0.001" --split-seed 20261028 --exclude-seed 20260922
python scripts/prepare_advection_horizons.py \
    --out external_data/pdebench/rs2/advection \
    --split-seed 20261028 --exclude-seed 20260922
python scripts/prepare_pdebench_1d_horizons.py \
    --source external_data/pdebench/darus/1D_Burgers_Sols_Nu0.01.hdf5 \
    --out external_data/pdebench/rs2/burgers_nu001 \
    --equation "1D Burgers, Nu=0.01" --split-seed 20261028 --exclude-seed 20260922

# 2) train: 21 families, 3 architectures x 5 seeds, 1500 updates
#    results/rs2_<family>_t<tag>_v1
python scripts/run_public_benchmark.py \
    --data-dir external_data/pdebench/rs2/burgers_nu0001/fixed_t100_v1 \
    --steps 1500 --out results/rs2_burgers_nu0001_t100_v1 --device cuda

# 3) diagnostics and the refit thresholds
python scripts/diagnose_fluctuation.py \
    --data external_data/pdebench/rs2/burgers_nu0001/fixed_t100_v1 \
    --families rs2_burgers_nu0001_t100_v1=endtoend:test,val \
    --out results/rs2_fluctuation/burgers_nu0001/t100 --device cuda
python scripts/predict_skill_horizon.py --out results/rs2_skill_prediction_v1

# 4) stratified held-out record
python scripts/analyze_prediction_ledger.py \
    --ledger experiments/prediction_ledger_rs2.json \
    --out results/rs2_prediction_ledger_v1
```

Result directory prefix `rs2_` throughout, so no existing family can be overwritten and no
existing identity can collide.

## 8. Status

Protocol committed 2026-10-28 with no runs started. Section 9 below is filled in after the
runs, and is the only part of this document that may change.

## 9. Result

_To be written after the runs. Endpoints, bands and tolerances above are fixed._
