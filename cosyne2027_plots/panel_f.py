from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    ROOT_DIR
    / "data"
    / "gridSearch_standard"
    / "60000"
    / "coeff_infty"
    / "max_points_df.csv"
)
OUTPUT_FILE = Path(__file__).resolve().parent / "panel_f.pdf"

POINTS_PER_INCH = 72
FIGURE_SIZE_PT = (211.3, 160)
FIGURE_SIZE_INCHES = tuple(length / POINTS_PER_INCH for length in FIGURE_SIZE_PT)
N_TRIALS = 6
N_VALUES = [1, 2, 3, 4, 5, 10]


def smooth_spline(x, y, y_std, s_factor=1, k=3):
    non_zero_idx = np.argwhere(y > 0).flatten()
    x = x[non_zero_idx]
    y = y[non_zero_idx]
    y_std = y_std[non_zero_idx]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y_std = np.maximum(y_std[idx], 1e-10)

    return UnivariateSpline(x, y, w=1 / y_std, k=k, s=s_factor * len(y))


def configure_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
        }
    )


def main():
    configure_style()
    df = pd.read_csv(DATA_FILE)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, layout="constrained")

    for n_idx, n_value in enumerate(N_VALUES):
        subset = df[df["ndR"] == n_value].copy()
        subset = subset.loc[subset.groupby(["capacity", "nd"])["b"].idxmax()]
        subset = subset.sort_values("nd")

        if len(subset) < 4:
            print(f"Skipping n = {n_value}: too few points")
            continue

        min_nd = subset["nd"].min()
        max_nd = subset["nd"].max()
        nd_plot_list = np.linspace(min_nd, max_nd, max(int(max_nd - min_nd), 200))

        try:
            spline = smooth_spline(
                np.log(subset["nd"].to_numpy()),
                subset["capacity"].to_numpy(),
                subset["sem"].to_numpy(),
            )
            ax.plot(
                nd_plot_list,
                spline(np.log(nd_plot_list)),
                label=f"{n_value}",
                color=f"C{n_idx}",
                linewidth=0.75,
            )
        except Exception as exc:
            print(f"Spline failed for n = {n_value}: {exc}")

        ax.errorbar(
            x=subset["nd"],
            y=subset["capacity"],
            yerr=np.sqrt(N_TRIALS) * subset["sem"],
            fmt="o",
            markersize=1.0,
            capsize=1.5,
            capthick=0.6,
            elinewidth=0.6,
            color=f"C{n_idx}",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_d$")
    ax.set_ylabel("Capacity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)

    legend = ax.legend(
        title=r"$n$",
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
        borderpad=0.2,
        labelspacing=0.25,
    )
    legend.get_title().set_fontsize(8)

    fig.savefig(OUTPUT_FILE)
    plt.close(fig)


if __name__ == "__main__":
    main()
