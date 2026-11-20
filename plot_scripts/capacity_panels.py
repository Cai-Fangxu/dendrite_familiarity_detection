"""Shared layout and summary handling for capacity-versus-dendrite figures."""

import argparse
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


N_TRIALS = 6  # Legacy summaries omit the sample count.
X_LIMITS = (0.5, 2000)
Y_LIMITS = (0, 1800)


def make_parser(description, panel_values, nrows, ncols, output):
    """Build the common CLI; the number of panels is len(n_values)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-values", type=float, nargs="+", default=panel_values,
                        help="One n value per panel, in display order")
    parser.add_argument("--nrows", type=int, default=nrows,
                        help="Grid rows; inferred when omitted")
    parser.add_argument("--ncols", type=int, default=ncols,
                        help="Grid columns; defaults to at most four")
    parser.add_argument("--output", type=Path, default=output,
                        help="Output figure path (default: PDF beside this script)")
    return parser


def panel_shape(n_panels, nrows=None, ncols=None):
    """Infer unspecified dimensions and reject grids that cannot fit the panels."""
    if n_panels < 1:
        raise ValueError("At least one panel is required")
    if any(value is not None and value < 1 for value in (nrows, ncols)):
        raise ValueError("nrows and ncols must be positive")
    if nrows is None and ncols is None:
        ncols = min(4, n_panels)
    if nrows is None:
        nrows = math.ceil(n_panels / ncols)
    if ncols is None:
        ncols = math.ceil(n_panels / nrows)
    if nrows * ncols < n_panels:
        raise ValueError(f"{n_panels} panels do not fit in a {nrows} by {ncols} grid")
    return nrows, ncols


def load_summary(path, panel_column):
    """Read a maximum-point table and validate values needed for plotting."""
    df = pd.read_csv(path)
    required = {"nd", "b", "capacity", "sem", panel_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: empty summary")
    numeric = sorted(required | ({"n_samples"} & set(df.columns)))
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError(f"{path}: plot values must be finite")
    if (df["nd"] <= 0).any() or (df[["capacity", "sem"]] < 0).any().any():
        raise ValueError(f"{path}: invalid dendrite count, capacity, or SEM")
    if "n_samples" in df and (df["n_samples"] < 2).any():
        raise ValueError(f"{path}: n_samples must be at least two")
    return df


def best_capacity_by_nd(df, n_value):
    """Choose one maximum per Nd, breaking capacity ties by largest threshold b."""
    subset = df[np.isclose(df["panel_n"], n_value, rtol=1e-12, atol=1e-12)]
    # Different synapse budgets must not silently become one curve.
    if "ns" in subset and (subset.groupby("nd")["ns"].nunique() > 1).any():
        raise ValueError(f"Multiple ns values for the same Nd at n={n_value:g}")
    return (subset.sort_values(["capacity", "b"], ascending=False, kind="stable")
            .drop_duplicates("nd").sort_values("nd"))


def make_figure(curves, n_values, nrows=None, ncols=None, panel_symbol="n",
                panel_y_limits=None):
    """Plot (label, color, linestyle, dataframe) curves with SD error bars.

    Tables supply SEM; sqrt(n_samples) * SEM recovers the trial standard
    deviation, matching fig3/capa_vs_nd_4Panels.py. Legacy tables assume six.
    Optional panel_y_limits maps n values to (lower, upper) bounds; unspecified
    panels use Y_LIMITS. Different ranges get independent axes and tick labels.
    """
    if any(not np.isfinite(n) or n <= 0 for n in n_values):
        raise ValueError("Panel n values must be finite and positive")
    if len(set(n_values)) != len(n_values):
        raise ValueError("Panel n values must be distinct")
    nrows, ncols = panel_shape(len(n_values), nrows, ncols)
    limits_by_n = panel_y_limits if panel_y_limits is not None else {}
    y_limits = [tuple(limits_by_n.get(n, Y_LIMITS)) for n in n_values]
    for n, limits in zip(n_values, y_limits):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"Y limits for n={n:g} must be two finite, increasing values")
    share_y = all(limits == y_limits[0] for limits in y_limits)
    subsets = []
    for n in n_values:
        selected = [best_capacity_by_nd(df, n) for _, _, _, df in curves]
        if all(df.empty for df in selected):
            available = sorted({float(n) for *_, df in curves for n in df["panel_n"]})
            raise ValueError(f"No data for n={n:g}; available values: {available}")
        for (label, *_), df in zip(curves, selected):
            if df.empty:
                warnings.warn(f"No {label} data for n={n:g}", stacklevel=2)
        subsets.append(selected)

    style = {
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
    legend_cols = min(len(curves), max(1, ncols))
    legend_height = 0.22 * math.ceil(len(curves) / legend_cols)
    with plt.rc_context(style):
        fig = plt.figure(figsize=(max(2.5, 1.8 * ncols), 0.6 + legend_height + 1.5 * nrows),
                         layout="constrained")
        grid = fig.add_gridspec(nrows + 1, ncols,
                               height_ratios=[legend_height] + [1.5] * nrows,
                               hspace=0.08, wspace=0.08)
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.axis("off")
        handles = [Line2D([], [], color=color, linestyle=style, linewidth=0.8,
                          marker="o", markersize=2, label=label)
                   for label, color, style, _ in curves]
        legend_ax.legend(handles=handles, loc="center", ncol=legend_cols,
                         frameon=False, handlelength=2, columnspacing=1.0)
        axes = []
        for index, (n, selected) in enumerate(zip(n_values, subsets)):
            row, col = divmod(index, ncols)
            ax = fig.add_subplot(grid[row + 1, col],
                                 sharex=axes[0] if axes else None,
                                 sharey=axes[0] if axes and share_y else None)
            axes.append(ax)
            for (_, color, linestyle, _), df in zip(curves, selected):
                if df.empty:
                    continue
                counts = df["n_samples"] if "n_samples" in df else N_TRIALS
                ax.errorbar(df["nd"], df["capacity"],
                            yerr=np.sqrt(counts) * df["sem"], fmt="o",
                            markersize=1.6, capsize=1.2, capthick=0.5,
                            elinewidth=0.5, color=color, alpha=0.9)
                ax.plot(df["nd"], df["capacity"], color=color,
                        linestyle=linestyle, linewidth=0.8)
            ax.set(xscale="log", xlim=X_LIMITS, ylim=y_limits[index])
            ax.set_title(rf"${panel_symbol}={n:g}$", pad=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", which="major", length=3)
            ax.tick_params(axis="both", which="minor", length=2)
            ax.tick_params(labelleft=(col == 0 or not share_y))
        fig.supxlabel(r"$N_d$")
        fig.supylabel("Capacity")
    return fig


def save_figure(fig, output):
    """Save the figure and release its Matplotlib resources."""
    try:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved {output}")
