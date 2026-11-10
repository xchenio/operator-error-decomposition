"""Draw the blog figure: a conserved mean is a horizontal line.

Uses one real trajectory from the PDEBench 1D Burgers Nu=0.001 deposit, the
same file the manuscript uses. A representative sample is chosen by taking,
from a fixed candidate set, the trajectory whose fluctuation amplitude is
closest to the median, so the picture is neither a flat nor an extreme case.
The figure is illustrative: the reported relative L2 and shares in section 5.2
are test-set statistics, not the values of this single trajectory, and are
therefore not printed on the figure.

Run: python scripts/make_blog_figure.py
"""

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external_data" / "pdebench" / "1D_Burgers_Sols_Nu0.001.hdf5"
OUT = ROOT / "blog" / "level_line.png"

CANDIDATES = np.arange(0, 40)
T_INPUT, T_TARGET = 0, 100  # t = 0 and t = 1.0, as in the manuscript
STRIDE = 4                  # 1024 -> 256 points, as in the manuscript

BLUE = "#1f4e79"
GREY = "#9a9a9a"
ORANGE = "#d95f02"


def main() -> None:
    with h5py.File(DATA, "r") as f:
        x = f["x-coordinate"][:][::STRIDE].astype(np.float64)
        amps = []
        for i in CANDIDATES:
            y = f["tensor"][i, T_TARGET, ::STRIDE].astype(np.float64)
            amps.append(float(np.std(y - y.mean())))
        idx = int(CANDIDATES[int(np.argmin(np.abs(np.array(amps) - np.median(amps))))])
        u0 = f["tensor"][idx, T_INPUT, ::STRIDE].astype(np.float64)
        y = f["tensor"][idx, T_TARGET, ::STRIDE].astype(np.float64)

    mean_in, mean_y = float(u0.mean()), float(y.mean())
    fluct = y - mean_y

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=160)

    ax0.plot(x, u0, color=GREY, lw=1.1, alpha=0.9, label="Input  u(x, 0)")
    ax0.plot(x, y, color=BLUE, lw=2.0, label="Target  u(x, t = 1.0)")
    ax0.axhline(mean_y, color=ORANGE, lw=2.2, ls="--",
                label="Conserved mean = zero-fluctuation prediction")
    ax0.fill_between(x, mean_y, y, color=BLUE, alpha=0.13)
    ax0.set_title("A constant prediction is a horizontal line", fontsize=12)
    ax0.set_xlabel("x  (periodic domain)")
    ax0.set_ylabel("u(x)")
    ax0.text(0.02, 0.04,
             f"mean(input) = {mean_in:.4f}\nmean(target) = {mean_y:.4f}",
             transform=ax0.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9))
    ax0.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax0.grid(alpha=0.25)

    ax1.plot(x, fluct, color=BLUE, lw=1.8, label="Target fluctuation  y - mean(y)")
    ax1.axhline(0.0, color=ORANGE, lw=2.6, ls="--",
                label="Predicted fluctuation = 0")
    ax1.fill_between(x, 0.0, fluct, color=BLUE, alpha=0.16)
    ax1.set_title("Everything outside the line is left unpredicted", fontsize=12)
    ax1.set_xlabel("x  (periodic domain)")
    ax1.set_ylabel("fluctuation")
    ax1.text(0.02, 0.04, "fluctuation R² = 0  by definition",
             transform=ax1.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9))
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax1.grid(alpha=0.25)

    fig.suptitle("Why relative L2 can look acceptable when no fluctuation is predicted",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)
    print(f"trajectory {idx}: mean(input)={mean_in:.6f} mean(target)={mean_y:.6f} "
          f"difference={abs(mean_in - mean_y):.2e}, fluctuation RMS={np.sqrt((fluct**2).mean()):.4f}")


if __name__ == "__main__":
    main()
