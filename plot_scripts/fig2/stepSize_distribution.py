from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "stepSize_distribution"
OUTPUT_FILE = Path(__file__).resolve().parent / "stepSize_distribution_params3.pdf"


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

    step_sizes_non_int = np.load(DATA_DIR / "nonInt_params3.npy")
    step_sizes_int_variable_n_min1 = np.load(DATA_DIR / "int_variableN_min1_params3.npy")
    step_sizes_int = np.load(DATA_DIR / "int_params3.npy")

    fig, ax = plt.subplots(figsize=(2.3, 1.9), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin = 0.0
    xmax = 0.3
    bins = 100
    range_ = (xmin, xmax)

    for data, label in [
        (step_sizes_non_int, "non-interacting"),
        (step_sizes_int_variable_n_min1, r"variable-$u_{\text{target}}$"),
        (step_sizes_int, "n-WTA"),
    ]:
        mids, counts = hist_midpoints(data, bins, range_)
        ax.plot(mids, counts, label=label)

    ax.set_xlabel("Update step size")
    ax.set_ylabel("Probability density")
    ax.set_xlim(xmin, xmax)

    stats_data = [
        ("non-interacting", np.mean(step_sizes_non_int), np.var(step_sizes_non_int)),
        (r"variable-$u_{\text{target}}$", np.mean(step_sizes_int_variable_n_min1), np.var(step_sizes_int_variable_n_min1)),
        ("n-WTA", np.mean(step_sizes_int), np.var(step_sizes_int)),
    ]
    table_data = [[f"{mu:.3f}", f"{var:.2e}"] for _, mu, var in stats_data]

    table = ax.table(
        cellText=table_data,
        colLabels=[r"$\mu$", r"$\sigma^2$"],
        cellLoc="center",
        rowLoc="right",
        loc="upper right",
        bbox=[0.63, 0.32, 0.65, 0.6],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)

    for i, color in enumerate(["C0", "C1", "C2"], start=1):
        for j in range(2):
            table[(i, j)].get_text().set_color(color)

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
