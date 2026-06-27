from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
PANEL_SIZE = (2.3, 1.9)
N_TRIALS = 6
N_VALUES = [1, 2, 3, 4, 5, 10, 20, 30, 60, 100, 300]
VARIABILITY_STRINGS = ["0.1", "0.3", "0.5", "0.7", "0.9", "0.99"]


def configure_style():
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


def load_variable_ns_scan(coeff="infty"):
    dfs = []

    baseline_path = ROOT_DIR / "data" / "gridSearch_standard" / "60000" / f"coeff_{coeff}" / "max_points_df.csv"
    baseline_df = pd.read_csv(baseline_path)
    baseline_df["noise"] = "0"
    dfs.append(baseline_df)

    variable_ns_base = ROOT_DIR / "data" / "gridSearch_variableNs"
    for variability_str in VARIABILITY_STRINGS[1:]:
        file_path = variable_ns_base / variability_str / "max_points_df.csv"
        if not file_path.exists():
            continue
        tmp_df = pd.read_csv(file_path)
        tmp_df["noise"] = variability_str
        dfs.append(tmp_df)

    return pd.concat(dfs, ignore_index=True)


def plot_variable_ns_panel(output_file, panel_title, x_label, nd=100, show_ylabel=False, show_legend=False):
    configure_style()
    df = load_variable_ns_scan()
    subset = df[df["nd"] == nd]

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    fig.subplots_adjust(
        left=0.22 if show_ylabel else 0.11,
        right=0.98,
        bottom=0.22,
        top=0.86,
    )

    for n_idx, n_value in enumerate(N_VALUES):
        subset_n = subset[subset["ndR"] == n_value].copy()
        if subset_n.empty:
            continue

        subset_n = subset_n.loc[subset_n.groupby(["capacity", "noise"])["b"].idxmax()]
        subset_n["noise_value"] = subset_n["noise"].to_numpy().astype(float)
        subset_n = subset_n.sort_values("noise_value")

        ax.plot(
            subset_n["noise_value"],
            subset_n["capacity"],
            label=f"{n_value}",
            color=f"C{n_idx}",
            linewidth=0.75,
        )
        ax.errorbar(
            x=subset_n["noise_value"],
            y=subset_n["capacity"],
            yerr=np.sqrt(N_TRIALS) * subset_n["sem"],
            fmt="o",
            markersize=1.0,
            capsize=1.5,
            capthick=0.6,
            elinewidth=0.6,
            color=f"C{n_idx}",
        )

    ax.set_title(panel_title, pad=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Capacity" if show_ylabel else "")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=3)
    ax.tick_params(axis="both", which="minor", length=2)

    if show_legend:
        leg = ax.legend(
            title=r"$n$",
            ncol=2,
            frameon=False,
            handlelength=1.2,
            columnspacing=0.8,
            borderpad=0.2,
            labelspacing=0.25,
        )
        leg.get_title().set_fontsize(8)

    plt.savefig(Path(__file__).resolve().parent / output_file, bbox_inches="tight")
    plt.close(fig)
