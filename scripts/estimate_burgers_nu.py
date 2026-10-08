"""Estimate the effective diffusion coefficient of a PDEBench Burgers file from the data.

Why this exists
---------------
The manuscript already records that the upstream generator uses `epsilon/pi` as the diffusion
coefficient and that the filename parameter must not be substituted silently into a PDE
residual. Before any physics-informed loss can be written, the coefficient appearing in the
stored trajectories has to be measured, not assumed.

Method
------
The stored field is periodic and cell-centred, so spatial derivatives are taken spectrally and
the time derivative by a second-order central difference on interior layers. For
`u_t + u u_x - nu u_xx = 0` the coefficient is then the least-squares solution

    nu* = <u_t + u u_x, u_xx> / <u_xx, u_xx>

pooled over layers and samples. The script also reports the residual at `nu*` and at the
filename value, so a reader can see whether the fit is meaningful or degenerate, and checks
periodicity of the stored field.

Usage
-----
    python scripts/estimate_burgers_nu.py
    python scripts/estimate_burgers_nu.py --samples 8 --layers 20,40,60,80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def spectral_derivatives(u: np.ndarray, domain: float = 1.0):
    """u: (..., N) periodic, cell-centred -> du, d2u on the same grid."""
    n = u.shape[-1]
    k = 2.0j * np.pi * np.fft.fftfreq(n, d=domain / n)
    uh = np.fft.fft(u, axis=-1)
    du = np.real(np.fft.ifft(k * uh, axis=-1))
    # k is i*2*pi*f, so k**2 is the correct second-derivative multiplier. Writing
    # -(k**2) here flips the sign and yields a negative fitted diffusion coefficient.
    d2u = np.real(np.fft.ifft((k ** 2) * uh, axis=-1))
    return du, d2u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--layers", default="20,40,60,80,100")
    ap.add_argument("--out", default="results/nu_estimate_v1")
    args = ap.parse_args()

    source = ROOT / args.source
    layers = [int(x) for x in args.layers.split(",") if x]
    rows = []
    with h5py.File(source, "r") as h:
        data = h["tensor"]
        ts = h["t-coordinate"][:]
        dt = float(ts[1] - ts[0])
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
        for s in range(args.samples):
            traj = data[s][:, :].astype(np.float64)          # (201, 1024)
            for li in layers:
                ut = (traj[li + 1] - traj[li - 1]) / (2.0 * dt)
                ux, uxx = spectral_derivatives(traj[li])
                rows.append((ut, traj[li], ux, uxx, float(ts[li])))

    ut = np.concatenate([r[0] for r in rows])
    uu = np.concatenate([r[1] for r in rows])
    ux = np.concatenate([r[2] for r in rows])
    uxx = np.concatenate([r[3] for r in rows])
    rhs = ut + uu * ux
    denom = float(np.dot(uxx, uxx))
    nu_fit = float(np.dot(rhs, uxx) / denom) if denom > 0 else float("nan")

    def residual(nu):
        r = rhs - nu * uxx
        return float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(rhs ** 2)))

    filename_nu = float(attrs.get("Nu", float("nan")))
    report = dict(source=str(source.name), file_attrs=attrs,
                  layers=layers, samples=args.samples, dt=dt,
                  nu_fitted=nu_fit,
                  relative_residual_at_fit=residual(nu_fit),
                  relative_residual_at_filename=residual(filename_nu),
                  relative_residual_at_zero=residual(0.0),
                  filename_nu=filename_nu,
                  periodicity=dict())
    # periodicity check on one trajectory
    with h5py.File(source, "r") as h:
        traj = h["tensor"][0][:, :].astype(np.float64)
        gap = np.abs(traj[:, 0] - traj[:, -1])
        report["periodicity"] = dict(max_abs_endpoint_gap=float(gap.max()),
                                     mean_abs_field=float(np.abs(traj).mean()))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
