# Third operator class: what to download and why

## Status (2026-09-29)

The download could **not** be completed from this machine. Measured throughput from the
University of Stuttgart DaRUS mirror was about 34 KB/s on a single connection and about
96 KB/s with four parallel range requests, so `1D_diff-sorp_NA_NA.h5` (4.2 GB) would need
roughly 33 hours. The partial file is kept at

```
external_data/pdebench/darus/1D_diff-sorp_NA_NA.h5.part      (about 16 MB)
external_data/pdebench/darus/1D_diff-sorp_NA_NA.h5.status.json
```

`scripts/download_darus.py` resumes from that `.part` file if the network is better
elsewhere:

```bash
python scripts/download_darus.py --file 1D_diff-sorp_NA_NA.h5
```

## Why a third operator class is the highest-value remaining experiment

The current evidence supports two linked claims:

1. a **skill horizon** exists on nonlinear Burgers and is absent on linear advection;
2. the binding constraint is the **branch**, not the trunk span (section 5.7).

Both rest on two PDEs. A third class turns them from a two-point contrast into a tested
rule. It is also the direct test of the threshold rule in section 5.8, which is currently
fitted in-sample and explicitly not validated.

## Why HuggingFace is not usable here

- `huggingface.co` — connection timeout, then connection reset (`WinError 10054`).
- `hf-mirror.com` — HTTP 401 on both the API and dataset pages.
- `darus.uni-stuttgart.de` — reachable, but only when the `certifi` CA bundle is passed
  explicitly; the default SSL context raises `CERTIFICATE_VERIFY_FAILED`.
  `scripts/download_darus.py` already does this.
- Control check: `pypi.org` and `api.github.com` both answer, so general egress works and
  the failure is specific to the HuggingFace host.

## Recommended datasets, in priority order

Source deposit: `doi:10.18419/darus-2986` (PDEBench, University of Stuttgart).
Page: https://darus.uni-stuttgart.de/ (search `darus-2986`).
Direct access API: `https://darus.uni-stuttgart.de/api/access/datafile/<id>`

| File | id | size | MD5 | what it tests |
|---|---:|---:|---|---|
| `1D_diff-sorp_NA_NA.h5` | 133020 | 4,217,044,280 B (4.2 GB) | `9d466d1213065619d087319e16d9a938` | **first choice.** Parabolic, diffusion-dominated, time dependent, so it has forecast horizons. If a skill horizon appears here too, the driver is fluctuation-energy decay rather than shock formation. |
| `2D_DarcyFlow_beta1.0_Train.hdf5` | 133219 | 1,310,724,488 B (1.3 GB) | `81694ed31306ff2e5f6b76349b0b4389` | Steady **elliptic** operator, smallest download. No forecast horizon, but it tests the fluctuation-energy threshold rule on a completely different operator class. |
| `1D_Burgers_Sols_Nu0.01.hdf5` | 281363 | 8,232,968,312 B (8.2 GB) | `e6d9a4f62baf9a29121a816b919e2770` | Same operator, ten times the viscosity. A viscosity ladder, not a new class; lower priority. |

If you can only fetch one, take `1D_diff-sorp_NA_NA.h5`.

Direct URLs, if you prefer a browser or a download manager:

```
https://darus.uni-stuttgart.de/api/access/datafile/133020   # 1D_diff-sorp_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133219   # 2D_DarcyFlow_beta1.0_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/281363   # 1D_Burgers_Sols_Nu0.01.hdf5
```

Mirror on HuggingFace, for machines where it is reachable (currently not this one):

```
https://huggingface.co/datasets/pdebench/<DatasetName>/resolve/main/<filename>
```

## What to do after the file arrives

1. Verify the MD5 before use. `scripts/download_darus.py` does this automatically and
   refuses to rename `.part` to the final name on mismatch.
2. Inspect the layout first. `prepare()` in `src/public_benchmark.py` requires
   `tensor` with shape `(10000, 201, 1024)` plus `t-coordinate` and `x-coordinate`, and it
   pins the **Burgers** SHA-256, so it will refuse a new file. Two options:
   * 1D files that match that layout: extend `scripts/prepare_advection_horizons.py`, which
     already reuses the same helpers and records the file's own hash instead of a pinned
     digest. Add an entry, do not edit `src/`.
   * 2D or differently shaped files: a new preparation script is needed. Do **not** relax
     the layout checks to make a file fit.
3. Keep the split identical (seed 20260922, sizes 1000/200/200) so trajectories match the
   existing families.
4. Run the same 1500-update budget and 5 seeds, then
   `scripts/diagnose_fluctuation.py --data <new dir> --families <name>=endtoend:test,val`.
5. The predeclared test of section 5.8 is: does the measured sign of per-sample fluctuation
   skill match the prediction from the fluctuation-energy threshold (about 0.248 for MLP
   DeepONet, 0.273 for Fourier, 0.150 for FNO) on a PDE that was not used to fit them?
   Record the answer whatever it is; a falsified threshold is still a result.
