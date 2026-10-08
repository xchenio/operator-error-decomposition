# PDEBench data version history

**Scope: this document is a provenance analysis. It is deliberately NOT part of the
manuscript.** The paper states only that the current release is used
(`doi:10.18419/darus-2986` version 8, released 2024-02-13) and that every input file is
byte-pinned. The version-by-version history below exists so that the question "is this data
current, and did it change?" can be answered from the repository rather than from memory.

Queried **2026-10-08** through the Dataverse API at `darus.uni-stuttgart.de` and the
HuggingFace dataset API. Raw per-version file inventories were saved to
`results/darus_version_files.json`.

---

## 1. Version timeline

Dataset id 133015, persistent identifier `doi:10.18419/darus-2986`, title "PDEBench
Datasets". Eight versions, all `RELEASED`; no draft version exists above version 8.

| Version | id | created | released | files | delta vs previous |
|---|---:|---|---|---:|---|
| v1 | 3177 | 2022-06-08 | **2022-07-26** | 373 | initial |
| v2 | 3505 | 2022-10-12 | **2022-10-14** | 373 | +12 / -12 |
| v3 | 3531 | 2022-10-23 | **2023-04-18** | 584 | +278 / -67 |
| v4 | 4107 | 2023-08-29 | **2023-08-30** | 578 | -6 |
| v5 | 4127 | 2023-09-01 | **2023-09-05** | 584 | +6 |
| v6 | 4188 | 2023-10-05 | **2023-10-09** | 586 | +8 / -6 |
| v7 | 4271 | 2023-11-24 | **2023-11-30** | 594 | +8 |
| v8 | 4465 | 2024-02-07 | **2024-02-13** | 336 | +4 / -262 |

The deposit therefore grew from 373 to 594 files between 2022 and late 2023, then **shrank
to 336 in v8**, which removed 262 files.

## 2. How the deposit changes files

Across all eight versions, **no datafile identifier ever changed its MD5 in place** (`~0`
content changes at every step). Revisions are done by removing an identifier and adding a
new one with the same filename and different content. Two consequences:

1. An identical MD5 across versions is strong evidence of identical bytes.
2. A changed identifier with the same filename is a genuine content revision.

The two revision events that touch this project are listed in section 4.

## 3. Our nine input files

| File | Source | Identifier | Present in | Content |
|---|---|---|---|---|
| `1D_Burgers_Sols_Nu0.001.hdf5` | HuggingFace `pdebench/Burgers` | git rev `97bf5bb5315b` | current HEAD | see section 5 |
| `1D_Advection_Sols_beta0.4.hdf5` | HuggingFace `pdebench/Advection` | git rev `33d90a245d98b` | current HEAD | see section 5 |
| `2D_diff-react_NA_NA.h5` | DaRUS | 133017 | **v1 - v8** | unchanged, MD5 stable |
| `1D_diff-sorp_NA_NA.h5` | DaRUS | 133020 | **v1 - v8** | unchanged, MD5 stable |
| `2D_DarcyFlow_beta0.01_Train.hdf5` | DaRUS | 133217 | **v1 - v8** | unchanged, MD5 stable |
| `2D_DarcyFlow_beta1.0_Train.hdf5` | DaRUS | 133219 | **v1 - v8** | unchanged, MD5 stable |
| `2D_DarcyFlow_beta10.0_Train.hdf5` | DaRUS | 133220 | **v1 - v8** | unchanged, MD5 stable |
| `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5` | DaRUS | 164672 | **v2 - v8** | **revised in v2** |
| `1D_Burgers_Sols_Nu0.01.hdf5` | DaRUS | 281363 | **v8 only** | **revised in v8** |

**Five of the seven DaRUS files are byte-identical to the initial v1 release of
2022-07-26.** For those files the description "2022 data" is literally accurate at the byte
level, even though we obtained them from the v8 deposit in 2026.

## 4. The two revision events

| Version | File | old id | old MD5 (prefix) | new id | new MD5 (prefix) |
|---|---|---:|---|---:|---|
| v2 (2022-10-14) | `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5` | 133152 | `d8d71c22c0` | 164672 | `2fdd00138f` |
| v8 (2024-02-13) | `1D_Burgers_Sols_Nu0.01.hdf5` | 133136 | `0a49257a0e` | 281363 | `e6d9a4f62b` |

The v2 event replaced twelve files that day, all CFD variants. The v8 event replaced four
Burgers files (`Nu0.01`, `Nu0.04`, `Nu1.0`, `Nu2.0`) while removing 262 others.

Both revisions predate our downloads (2026-09-22 to 2026-10-08), so we hold the revised
content in both cases. What changed *inside* those files is not determined here; the
deposit publishes no per-version changelog.

## 5. HuggingFace revisions are the current HEAD

| Dataset | `lastModified` from API | current `sha` | revision pinned in `scripts/` | match |
|---|---|---|---|---|
| `pdebench/Burgers` | 2024-02-15 | `97bf5bb5315b9a083b4f9fc03e5c84d35286d52c` | `97bf5bb5315b9a083b4f9fc03e5c84d35286d52c` | **yes** |
| `pdebench/Advection` | 2024-02-14 | `33d90a245d98b8a250884eab613c6f9c26a13b68` | `33d90a245d98b8a250884eab613c6f9c26a13b68` | **yes** |

The two HuggingFace datasets were last modified on 2024-02-14 and 2024-02-15, one to two
days after the DaRUS v8 release of 2024-02-13, consistent with a coordinated re-release of
the same material on both mirrors.

## 6. Answer: is the data current?

**Yes, on every channel checked, as of 2026-10-08.**

- DaRUS `darus-2986` is at version 8, `RELEASED`, 2024-02-13. There is no later version and
  no draft.
- The HuggingFace datasets are at the revisions we pin; their `sha` equals the current HEAD.
- Our files were downloaded 2026-09-22 to 2026-10-08, i.e. after all revisions above.
- Upstream has been static since **2024-02-15**, about 2.7 years.

So the previous concern about "a four-year-old benchmark" was overstated in two ways: the
data is 2.7 years old rather than 4, and it is the newest release rather than a stale
snapshot. The remaining honest caveat is not about staleness but about the field having
moved on: newer surrogate-benchmark frameworks (PDEArena and successors) exist, and nothing
here is claimed to transfer to foundation-model-scale operator training.

## 7. Limitations of this analysis

- The Dataverse API returns MD5 for each file; no per-version changelog or diff of file
  *contents* is published, so the nature of the two revisions in section 4 is not determined.
- The HuggingFace datasets expose only the current commit here; no attempt was made to walk
  their commit history, so we cannot say whether the Burgers Nu0.001 file changed before
  revision `97bf5bb`.
- File counts in section 1 are as returned by the API and include all files in the deposit,
  not only the HDF5/H5 data files.
- This analysis is pointwise in time. Re-run
  `https://darus.uni-stuttgart.de/api/datasets/:persistentId/versions?persistentId=doi:10.18419/darus-2986`
  before any submission to confirm that no version 9 has appeared.
