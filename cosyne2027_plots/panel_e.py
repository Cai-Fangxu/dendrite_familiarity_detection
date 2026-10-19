from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
CAPACITY_DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "capa_vs_coeff"
WEIGHT_CORR_DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "weight_corr_vs_coeff"
OUTPUT_FILE = Path(__file__).resolve().parent / "panel_e.pdf"

POINTS_PER_INCH = 72
FIGURE_SIZE_PT = (300, 160)
FIGURE_SIZE_INCHES = tuple(length / POINTS_PER_INCH for length in FIGURE_SIZE_PT)
COEFFS = np.array([1, 2, 4, 8, 16, 32, 64])
# COEFFS = np.array([1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64])
TICK_COEFFS = np.array([1, 4, 16, 64])

SERIES = (
    ("nonInt", "C0", "non-interacting"),
    ("int_variableN_min1", "C1", r"variable-$u_{\text{target}}$"),
    ("int", "C2", "n-WTA"),
)


def configure_axes(ax, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TICK_COEFFS)
    ax.set_xticklabels(["1" if coeff == 1 else f"1/{coeff}" for coeff in TICK_COEFFS])
    ax.invert_xaxis()
    ax.set_xlabel("Input correlation")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)


def plot_capacity(ax, capacity_trials, color, label):
    mean_capacity = np.mean(capacity_trials, axis=1)
    std_capacity = np.std(capacity_trials, axis=1)

    ax.plot(
        COEFFS,
        mean_capacity,
        marker="o",
        markersize=3,
        linewidth=1.0,
        color=color,
        label=label,
    )
    ax.errorbar(
        COEFFS,
        mean_capacity,
        yerr=std_capacity,
        fmt="none",
        capsize=2,
        capthick=0.6,
        elinewidth=0.6,
        color=color,
    )


def plot_weight_correlation(ax, values, color):
    ax.plot(
        COEFFS,
        np.mean(values, axis=1),
        marker="o",
        markersize=3,
        linewidth=1.0,
        color=color,
    )


def main():
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

    fig, (capacity_ax, weight_corr_ax) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_INCHES,
    )

    for prefix, color, label in SERIES:
        capacity_trials = np.load(CAPACITY_DATA_DIR / f"{prefix}_6_trials_params3.npy")
        weight_correlations = np.load(WEIGHT_CORR_DATA_DIR / f"{prefix}_params3.npy")
        plot_capacity(capacity_ax, capacity_trials, color, label)
        plot_weight_correlation(weight_corr_ax, weight_correlations, color)

    configure_axes(capacity_ax, "Capacity")
    configure_axes(weight_corr_ax, "Average weight correlation")

    handles, labels = capacity_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.1,
        labelspacing=0.2,
    )
    # Reserve fixed regions for the shared legend and per-axis labels.
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.28, top=0.78, wspace=0.48)
    fig.savefig(OUTPUT_FILE)
    plt.close(fig)


if __name__ == "__main__":
    main()
