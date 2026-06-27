from pathlib import Path

import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = Path(__file__).resolve().parent / "capa_vs_nd_4Panels.pdf"

COEFF_LIST = ["infty", "4", "1"]
PANEL_N_VALUES = [1, 3, 5, 10]
MODEL_SPECS = [
    ("gridSearch_NonInt", "C0", "non-interacting"),
    ("gridSearch_int_variable_n_min1", "C1", r"variable-$u_{\text{target}}$"),
    ("gridSearch_standard", "C2", "n-WTA"),
]
COEFF_LINESTYLES = {
    "infty": "-",
    "4": "--",
    "1": ":",
}
COEFF_LABELS = {
    "infty": r"$c=\infty$",
    "4": r"$c=4$",
    "1": r"$c=1$",
}
N_TRIALS = 6


def load_grid_search_data():
    dfs_by_model = {}
    for model_dir, _, _ in MODEL_SPECS:
        dfs_by_model[model_dir] = {
            coeff: pd.read_csv(
                ROOT_DIR
                / "data"
                / model_dir
                / "60000"
                / f"coeff_{coeff}"
                / "max_points_df.csv"
            )
            for coeff in COEFF_LIST
        }
    return dfs_by_model


def best_capacity_by_nd(df, n_value):
    subset = df[df["ndR"] == n_value]
    subset = subset.loc[subset.groupby(["capacity", "nd"])["b"].idxmax()]
    return subset.sort_values("nd")


def plot_panel(ax, dfs_by_model, n_value):
    for coeff in COEFF_LIST:
        for model_dir, color, _ in MODEL_SPECS:
            subset = best_capacity_by_nd(dfs_by_model[model_dir][coeff], n_value)
            if subset.empty:
                continue

            ax.errorbar(
                x=subset["nd"],
                y=subset["capacity"],
                yerr=np.sqrt(N_TRIALS) * subset["sem"],
                fmt="o",
                markersize=1.6,
                capsize=1.2,
                capthick=0.5,
                elinewidth=0.5,
                color=color,
                alpha=0.9,
            )
            ax.plot(
                subset["nd"],
                subset["capacity"],
                color=color,
                linestyle=COEFF_LINESTYLES[coeff],
                linewidth=0.8,
            )

    ax.set_xlim(0.5, 2000)
    ax.set_ylim(0, 1800)
    ax.set_xscale("log")
    ax.set_title(rf"$n={n_value}$", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)


def add_top_legend(legend_ax):
    legend_ax.axis("off")

    model_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="-", markersize=3, linewidth=0.9, label=label)
        for _, color, label in MODEL_SPECS
    ]
    coeff_handles = [
        mlines.Line2D([], [], color="0.25", linestyle=COEFF_LINESTYLES[coeff], linewidth=1.0, label=COEFF_LABELS[coeff])
        for coeff in COEFF_LIST
    ]

    model_legend = legend_ax.legend(
        handles=model_handles,
        loc="center",
        bbox_to_anchor=(0.33, 0.65),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.1,
    )
    legend_ax.add_artist(model_legend)

    legend_ax.legend(
        handles=coeff_handles,
        loc="center",
        bbox_to_anchor=(0.80, 0.65),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.1,
    )


def main():
    plt.rcParams.update({
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
    })

    dfs_by_model = load_grid_search_data()

    fig = plt.figure(figsize=(7.2, 2.25))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.34, 1.9], hspace=0.14, wspace=0.18)

    legend_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, idx]) for idx in range(4)]

    for idx, (ax, n_value) in enumerate(zip(axes, PANEL_N_VALUES)):
        plot_panel(ax, dfs_by_model, n_value)
        if idx != 0:
            ax.tick_params(labelleft=False)

    add_top_legend(legend_ax)
    fig.supxlabel(r"$N_d$", y=0.02)
    fig.supylabel("Capacity", x=0.002)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.98)

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
