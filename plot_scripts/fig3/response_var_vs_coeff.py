from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "response_distribution_vs_coeff"
OUTPUT_FILE = Path(__file__).resolve().parent / "response_var_vs_coeff_params3.pdf"

COEFF_LIST = np.array([1, 2, 4, 8, 16, 32, 64])


def configure_axes(ax, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(COEFF_LIST)
    ax.set_xticklabels([str(coeff) for coeff in COEFF_LIST])
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)


def plot_series(ax, values, color, label):
    ax.plot(
        COEFF_LIST,
        values,
        marker="o",
        markersize=3,
        linewidth=1.0,
        color=color,
        label=label,
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

    response_non_int = np.load(DATA_DIR / "nonInt_6_trials_params3.npy")
    response_variable_n_min1_int = np.load(DATA_DIR / "int_variableN_min1_6_trials_params3.npy")
    response_int = np.load(DATA_DIR / "int_6_trials_params3.npy")

    mean_non_int = np.mean(np.var(response_non_int, axis=2), axis=1)
    mean_variable_n_min1_int = np.mean(np.var(response_variable_n_min1_int, axis=2), axis=1)
    mean_int = np.mean(np.var(response_int, axis=2), axis=1)

    fig, ax = plt.subplots(figsize=(2.3, 1.9))

    plot_series(ax, mean_non_int, "C0", "non-interacting")
    plot_series(ax, mean_variable_n_min1_int, "C1", r"variable-$u_{\text{target}}$")
    plot_series(ax, mean_int, "C2", "n-WTA")

    configure_axes(ax, "Response variance at age 250")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 10, 20, 40])
    ax.set_yticklabels(["1", "2", "4", "10", "20", "40"])

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.98)
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
