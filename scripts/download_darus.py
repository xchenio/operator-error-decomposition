"""Resumable download of additional PDEBench files from the DaRUS mirror.

Why this exists
---------------
`huggingface.co` is unreachable from this machine (connection timeout / reset), and
`hf-mirror.com` answers 401. The University of Stuttgart DaRUS deposit
`doi:10.18419/darus-2986` serves the same PDEBench files over HTTPS and is reachable when
the `certifi` CA bundle is used explicitly (the default context failed with
`CERTIFICATE_VERIFY_FAILED`).

Publisher integrity is therefore verified with the **MD5 published by DaRUS**, not with the
SHA-256 pinned in `scripts/download_pdebench.py`, which covers the HuggingFace copy only.

Usage
-----
    python scripts/download_darus.py --file 1D_diff-sorp_NA_NA.h5
    python scripts/download_darus.py --file 2D_DarcyFlow_beta1.0_Train.hdf5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATALOG = {
    "1D_diff-sorp_NA_NA.h5": dict(id=133020, size=4217044280,
                                  md5="9d466d1213065619d087319e16d9a938",
                                  note="1D diffusion-sorption; parabolic, diffusion-dominated"),
    "2D_DarcyFlow_beta1.0_Train.hdf5": dict(id=133219, size=1310724488,
                                            md5="81694ed31306ff2e5f6b76349b0b4389",
                                            note="2D Darcy flow; steady elliptic"),
    "1D_Burgers_Sols_Nu0.01.hdf5": dict(id=281363, size=8232968312,
                                        md5="e6d9a4f62baf9a29121a816b919e2770",
                                        note="1D Burgers, higher viscosity"),
}
DATASET = "doi:10.18419/darus-2986"


def _context() -> ssl.SSLContext:
    import certifi
    return ssl.create_default_context(cafile=certifi.where())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, choices=sorted(CATALOG))
    ap.add_argument("--out-dir", default="external_data/pdebench/darus")
    ap.add_argument("--attempts", type=int, default=60)
    args = ap.parse_args()

    meta = CATALOG[args.file]
    url = f"https://darus.uni-stuttgart.de/api/access/datafile/{meta['id']}"
    folder = ROOT / args.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / args.file
    part = folder / (args.file + ".part")
    state = folder / (args.file + ".status.json")
    ctx = _context()

    def status(kind, **kw):
        current = (part.stat().st_size if part.exists()
                   else (target.stat().st_size if target.exists() else 0))
        info = dict(status=kind, dataset=DATASET, url=url, filename=args.file,
                    expected_bytes=meta["size"], expected_md5=meta["md5"],
                    downloaded_bytes=current, time_unix=time.time(), **kw)
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
        tmp.replace(state)
        print(json.dumps(info), flush=True)

    failures = 0
    while not target.exists():
        offset = part.stat().st_size if part.exists() else 0
        if offset >= meta["size"]:
            break
        status("downloading")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                if offset and response.status != 206:
                    raise RuntimeError("server did not honor resume range")
                with part.open("ab") as f:
                    last = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        f.write(chunk)
                        if time.monotonic() - last > 10:
                            f.flush()
                            os.fsync(f.fileno())
                            status("downloading")
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:  # transient: retry, the file is appended in place
            failures += 1
            status("retrying", error=str(e), attempt=failures)
            if failures >= args.attempts:
                status("failed", error=str(e))
                raise
            time.sleep(3)

    candidate = target if target.exists() else part
    if candidate.stat().st_size != meta["size"]:
        status("failed", error="size mismatch")
        raise RuntimeError("size mismatch")

    status("verifying")
    h = hashlib.md5()
    with candidate.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    if h.hexdigest() != meta["md5"]:
        status("failed", error="MD5 mismatch", actual_md5=h.hexdigest())
        raise RuntimeError("MD5 mismatch")
    if candidate == part:
        part.replace(target)
    status("completed", md5=h.hexdigest(), source="DaRUS " + DATASET, note=meta["note"])


if __name__ == "__main__":
    main()
