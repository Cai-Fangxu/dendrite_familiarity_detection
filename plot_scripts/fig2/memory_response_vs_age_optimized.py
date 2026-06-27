from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "fixed_params" / "memory_response_vs_age"
OUTPUT_FILE = Path(__file__).resolve().parent / "memory_response_vs_age_coeff_infty_optimized.pdf"


def load_response(prefix):
    return {
        "upper99": np.load(prefix / "coeff_infty_upper99.npy"),
        "transient_upper99": np.load(prefix / "coeff_infty_transient_upper99.npy"),
        "transient_lower01": np.load(prefix / "coeff_infty_transient_lower01.npy"),
        "transient_mean": np.load(prefix / "coeff_infty_transient_mean.npy"),
        "capa": np.load(prefix / "coeff_infty_capa.npy"),
    }


def main():
    responses = [
        (load_response(DATA_DIR / "nonInt_params3"), "C0", "non-interacting", 1),
        (load_response(DATA_DIR / "int_variableN_min1_params5"), "C1", r"variable-$u_{\text{target}}$", 2),
        (load_response(DATA_DIR / "int_params4"), "C2", "n-WTA", 3),
    ]

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = 2000
    plot_point_spacing = 10
    xs = np.arange(0, xmax, plot_point_spacing)

    for response, color, label, zorder in responses:
        ax.fill_between(
            xs,
            response["transient_upper99"][::plot_point_spacing],
            response["transient_lower01"][::plot_point_spacing],
            color=color,
            alpha=0.3,
            label=label,
            zorder=zorder,
        )

    for response, color, _, _ in responses:
        ax.plot(xs, response["transient_mean"][::plot_point_spacing], color=color)

    y_min, y_max = ax.get_ylim()

    for response, color, _, _ in responses:
        ax.hlines(
            y=response["upper99"][0],
            xmin=0,
            xmax=xmax,
            colors=color,
            linestyles="--",
            linewidth=1.0,
        )
        ax.vlines(
            x=response["capa"][0],
            ymin=y_min,
            ymax=response["upper99"][0],
            colors=color,
            linestyles="--",
            linewidth=1.0,
        )
        ax.plot(response["capa"], response["upper99"], color=color, marker="o", markersize=5)

    ax.set_xlim(0, xmax)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Pattern age")
    ax.set_ylabel("Memory response")
    # ax.legend(fontsize=8, frameon=False)

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
