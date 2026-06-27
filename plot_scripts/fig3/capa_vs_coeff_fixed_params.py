from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "capa_vs_coeff"
OUTPUT_FILE = Path(__file__).resolve().parent / "capa_vs_coeff_fixed_params3.pdf"

COEFF_LIST = np.array([1, 2, 4, 8, 16, 32, 64])


def plot_capacity(ax, coeffs, capacity_trials, color, label):
    mean_capacity = np.mean(capacity_trials, axis=1)
    std_capacity = np.std(capacity_trials, axis=1)

    ax.plot(
        coeffs,
        mean_capacity,
        marker="o",
        markersize=3,
        linewidth=1.0,
        color=color,
        label=label,
    )
    ax.errorbar(
        coeffs,
        mean_capacity,
        yerr=std_capacity,
        fmt="none",
        capsize=2,
        capthick=0.6,
        elinewidth=0.6,
        color=color,
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

    capa_list_non_int = np.load(DATA_DIR / "nonInt_6_trials_params3.npy")
    capa_list_variable_n_min1_int = np.load(DATA_DIR / "int_variableN_min1_6_trials_params3.npy")
    capa_list_int = np.load(DATA_DIR / "int_6_trials_params3.npy")

    fig, (legend_ax, ax) = plt.subplots(
        2,
        1,
        figsize=(2.3, 1.9),
        gridspec_kw={"height_ratios": [0.28, 1.0], "hspace": 0.02},
    )
    legend_ax.axis("off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_capacity(ax, COEFF_LIST, capa_list_non_int, "C0", "non-interacting")
    plot_capacity(ax, COEFF_LIST, capa_list_variable_n_min1_int, "C1", r"variable-$u_{\text{target}}$")
    plot_capacity(ax, COEFF_LIST, capa_list_int, "C2", "n-WTA")

    ax.set_xscale("log", base=2)
    ax.set_xticks(COEFF_LIST)
    ax.set_xticklabels([str(coeff) for coeff in COEFF_LIST])
    ax.set_xlabel(r"$c$")
    ax.set_ylabel("Capacity")
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.45,
        borderpad=0.1,
        labelspacing=0.2,
    )

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.96)
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
