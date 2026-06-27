from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "wLens_distribution"
OUTPUT_FILE = Path(__file__).resolve().parent / "wLen_distribution_params3.pdf"


def hist_midpoints(data, bins, range_):
    counts, edges = np.histogram(data, bins=bins, range=range_, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, counts


def main():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    wlens_array_non_int = np.load(DATA_DIR / "nonInt_params3.npy")
    wlens_array_int_variable_n_min1 = np.load(DATA_DIR / "int_variableN_min1_params3.npy")
    wlens_array_int = np.load(DATA_DIR / "int_params3.npy")

    xmin = 0.85
    xmax = 1.15
    bins = 100
    range_ = (xmin, xmax)

    fig, ax = plt.subplots(figsize=(2.3, 1.9), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for data, label in [
        (wlens_array_non_int, "non-interacting"),
        (wlens_array_int_variable_n_min1, r"variable-$u_{\text{target}}$"),
        (wlens_array_int, "n-WTA"),
    ]:
        mids, counts = hist_midpoints(data, bins, range_)
        ax.plot(mids, counts, label=label)

    ax.set_xlabel("Dendritic weight length")
    ax.set_ylabel("Probability density")
    ax.set_xlim(xmin, xmax)
    ax.legend(bbox_to_anchor=(0.55, 1), loc="upper left", frameon=False)

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
