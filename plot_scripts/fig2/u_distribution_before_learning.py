from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfcinv


OUTPUT_FILE = Path(__file__).resolve().parent / "u_distribution_before_learning.pdf"


def hist_midpoints(data, bins, range_):
    counts, edges = np.histogram(data, bins=bins, range=range_, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, counts


def activation_samples(nd=300, n=10, n_samples=500000):
    b = np.sqrt(2) * erfcinv(2 * n / nd)
    xs = np.random.normal(0, 1, (nd, n_samples))

    mask_non_interacting = xs > b
    xs_non_interacting = xs[np.nonzero(mask_non_interacting)]

    xs_sorted = np.sort(xs, axis=0)[-n:, :]
    all_below_threshold = np.sum(mask_non_interacting, axis=0) < 1
    xs_rank1_for_empty_columns = xs_sorted[-1, np.nonzero(all_below_threshold)].flatten()
    xs_variable_target = np.concatenate([xs_non_interacting, xs_rank1_for_empty_columns], axis=0)

    xs_n_wta = xs_sorted.flatten()
    return xs_non_interacting, xs_variable_target, xs_n_wta


def main():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    xs_non_interacting, xs_variable_target, xs_n_wta = activation_samples()
    stats_data = [
        ("non-interacting", np.mean(xs_non_interacting), np.var(xs_non_interacting)),
        (r"variable-$u_{\text{target}}$", np.mean(xs_variable_target), np.var(xs_variable_target)),
        ("n-WTA", np.mean(xs_n_wta), np.var(xs_n_wta)),
    ]

    fig, ax = plt.subplots(figsize=(2.3, 1.9), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin = 1.0
    xmax = 5.0
    bins = 100
    range_ = (xmin, xmax)

    for data, label in [
        (xs_non_interacting, "non-interacting"),
        (xs_variable_target, r"variable-$u_{\text{target}}$"),
        (xs_n_wta, "n-WTA"),
    ]:
        mids, counts = hist_midpoints(data, bins, range_)
        if label == "non-interacting":
            ax.plot(mids, counts, label=label, linewidth=2.7)
        else:
            ax.plot(mids, counts, label=label)

    ax.set_xlabel("Dendritic activation")
    ax.set_ylabel("Probability density")

    table_data = [[f"{mu:.2f}", f"{var:.3f}"] for _, mu, var in stats_data]
    table = ax.table(
        cellText=table_data,
        colLabels=[r"$\mu$", r"$\sigma^2$"],
        cellLoc="center",
        rowLoc="right",
        loc="upper right",
        bbox=[0.5, 0.35, 0.5, 0.6],
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
