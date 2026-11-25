"""Plot all enabled learning rules together from saved Figure 1 experiments.

    python plot_scripts/theory_fig1/plot_coefficients.py /path/to/fig1_run
    python plot_scripts/theory_fig1/plot_coefficients.py /path/to/fig1_run --rules nwta variable_target
    python plot_scripts/theory_fig1/plot_coefficients.py /path/to/fig1_run --exclude-rules softmax_nonint
    python plot_scripts/theory_fig1/plot_coefficients.py /path/to/fig1_run --max-points 100 --point-size 2 --no-error-bars

Accepts multiple run directories, e.g. for 10- versus 20-step forks. Each
architecture/fork-length combination gets its own figures so finite-duration
measurements are not pooled. Error bars are approximate conditional 95% intervals,
not figure-wide intervals treating old patterns/dendrites as independent seeds.
By default all saved rules are shown. Rule selection applies to every panel,
diagnostic and legend, without changing rule colors or saved simulation data.
Shared-input results also include shared_CDE_by_stage, with one row per stage
and common axis limits within each coefficient column.
Decay variants use lighter shades of their spherical families and their own
leading-order Gaussian predictions. In A/B their state-dependent predictions
are x markers; spherical predictions remain lines. Subsampling is display-only.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


COLORS = {"threshold": "#0072B2", "nwta": "#D55E00", "variable_target": "#009E73",
          "softmax_int": "#CC79A7", "softmax_nonint": "#E69F00",
          "threshold_decay": "#56B4E9", "nwta_decay": "#F3A366", "variable_target_decay": "#73C9A8"}
LABELS = {"threshold": "Threshold", "nwta": "n-WTA", "variable_target": "Variable target",
          "softmax_int": "Softmax interacting", "softmax_nonint": "Softmax non-interacting",
          "threshold_decay": "Threshold (decay)", "nwta_decay": "n-WTA (decay)",
          "variable_target_decay": "Variable target (decay)"}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*")
INT_FIELDS = ("seed", "t", "nd", "ns", "forks", "fork_steps", "age", "acquisition", "r", "s")
FLOAT_FIELDS = ("n", "b", "kappa", "eta", "beta_select", "beta_decay", "overlap", "predicted", "observed", "empirical_se", "theory_se")


@dataclass(frozen=True)
class PlotOptions:
    """Display-only controls; saved estimates and their uncertainties are unchanged."""

    max_points: int | None = None
    sample_seed: int = 0
    point_size: float = 3.0
    error_bars: bool = True


def subsample_points(points, options):
    """Uniform deterministic subset of one rule/stage/panel, independent of values.

    Ranking hashes of point identities keeps selection stable across input order,
    rule filters, and combined versus stage-row figures. Seeds/ages are sampled
    together; this changes only display density, not the stored estimators.
    """
    if options.max_points is None or len(points) <= options.max_points:
        return points
    def priority(row):
        # Drift and diagonal diffusion retain the same probes when their ages
        # match (notably A/B); pairwise diffusion has a different identity.
        component = "cross" if row["kind"] == "cross" else "dendrite"
        identity = [options.sample_seed, component] + [row[key] for key in
            ("rule", "condition", "seed", "t", "nd", "fork_steps", "age", "r", "s", "age_averaged")]
        return hashlib.sha256(json.dumps(identity).encode()).digest()
    return sorted(points, key=priority)[:options.max_points]


def load_runs(paths, rules=None):
    """Read completed snapshots for selected rules and validate their datasets."""
    selected = None if rules is None else set(rules)
    points, snapshots, seen, parameters = [], [], set(), {}
    for root in paths:
        files = sorted(root.glob("*/points.csv"))
        for path in files:
            if not (path.parent / "complete.json").exists():
                continue
            metadata = json.loads((path.parent / "metadata.json").read_text())
            if selected is not None and metadata["rule"]["name"] not in selected:
                continue
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for key in INT_FIELDS:
                    row[key] = int(row[key])
                for key in FLOAT_FIELDS:
                    row[key] = float(row[key]) if row.get(key) not in (None, "") else None
                row["age_averaged"] = row["age_averaged"] == "True"
                for key in ("overlap", "predicted", "observed", "empirical_se", "theory_se"):
                    if not np.isfinite(row[key]):
                        raise ValueError(f"Nonfinite {key} in {path}")
                group = (row["nd"], row["fork_steps"])
                identity = tuple(row[key] for key in ("rule", "condition", "seed", "t", "kind", "age", "r", "s"))
                if (group, identity) in seen:
                    raise ValueError(f"Duplicate data for {group}, {identity}; do not load a run twice")
                seen.add((group, identity))
                setting = tuple(row[key] for key in ("n", "b", "kappa", "eta", "beta_select", "beta_decay"))
                parameter_key = (row["condition"], row["nd"], row["rule"])
                if parameter_key in parameters and parameters[parameter_key] != setting:
                    raise ValueError(f"Conflicting learning parameters for {parameter_key}; plot separately")
                parameters[parameter_key] = setting
            points.extend(rows)
            snapshots.append(metadata)
    if not points:
        if selected is not None:
            raise ValueError("No completed snapshots found for the selected rules: " + ", ".join(sorted(selected)))
        raise ValueError("No completed snapshots with points.csv found in the supplied run directories")
    return points, snapshots


def decorate(ax):
    """Keep comparable panels legible without imposing fitted axis limits."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)


def add_legends(fig, points, stages):
    """Reserve color for rule and shape for training stage throughout."""
    from matplotlib.lines import Line2D

    present = {row["rule"] for row in points}
    handles = [Line2D([], [], color=color, marker="o", linestyle="", label=LABELS[rule])
               for rule, color in COLORS.items() if rule in present]
    handles.extend(Line2D([], [], color="0.3", marker=marker, linestyle="", label=f"t={t}")
                   for t, marker in stages.items())
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8)


def draw_points(ax, points, stages, x="predicted", residual=False, options=PlotOptions(), theory_only=False):
    """Show conditional uncertainty, using shared-fork age averages when supplied."""
    for rule, color in COLORS.items():
        for t, marker in stages.items():
            subset = [row for row in points if row["rule"] == rule and row["t"] == t]
            subset = subsample_points(subset, options)
            for age in sorted({row["age"] for row in subset}):
                rows = [row for row in subset if row["age"] == age]
                if not rows:
                    continue
                maximum_age = max(max((r["age"] for r in subset), default=0), 1)
                alpha = 0.45 if age < 0 else 0.45 - 0.25 * age / maximum_age
                values = np.array([row["predicted"] if theory_only else
                                   row["observed"] - (row["predicted"] if residual else 0) for row in rows])
                errors = np.array([row["theory_se"] if theory_only else row["empirical_se"] for row in rows])
                if residual:
                    errors = np.hypot(errors, [row["theory_se"] for row in rows])
                xerr = 1.96 * np.array([row["theory_se"] for row in rows]) if x == "predicted" and options.error_bars else None
                ax.errorbar([row[x] for row in rows], values, yerr=1.96*errors if options.error_bars else None, xerr=xerr,
                            fmt="x" if theory_only else marker, color=color, alpha=alpha, markersize=options.point_size,
                            elinewidth=0.45, markeredgewidth=0.25, capsize=0, linestyle="")


def save_figure(fig, directory, name):
    """Save editable-vector PDF and a convenient PNG preview."""
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")


def shared_coefficient_panel(ax, points, stages, kind, title, limits=None, options=PlotOptions()):
    """Draw one C/D/E panel and return its common prediction/measurement limits."""
    rows = [r for r in points if r["kind"] == kind and r["age_averaged"] == (kind != "drift")]
    if not rows:
        ax.set_visible(False)
        return None
    draw_points(ax, rows, stages, options=options)
    if kind == "cross":
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)
    if limits is None:
        limits = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
                  max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(limits, limits, "k--", lw=1)
    ax.set(xlim=limits, ylim=limits, title=title,
           xlabel="Local theory prediction", ylabel="Measured coefficient")
    decorate(ax)
    return limits


def shared_stage_plots(points, directory, stages, nd, steps, limits, options=PlotOptions()):
    """Separate training stages into rows, retaining all selected rules per row."""
    import matplotlib.pyplot as plt

    times = sorted({r["t"] for r in points})
    height = 3.4 * len(times) + 1.4
    fig, axes = plt.subplots(len(times), 3, figsize=(15, height), squeeze=False,
                             sharex="col", sharey="col")
    titles = ("C · Shared drift", "D · Shared diagonal diffusion", "E · Shared cross-diffusion")
    for i, t in enumerate(times):
        rows = [r for r in points if r["t"] == t]
        for j, kind in enumerate(("drift", "diagonal", "cross")):
            ax = axes[i, j]
            shared_coefficient_panel(ax, rows, {t: stages[t]}, kind,
                                     titles[j] if i == 0 else "", limits=limits[j], options=options)
            if i < len(times) - 1:
                ax.set_xlabel("")
        axes[i, 0].set_ylabel(f"Stage t={t}\nMeasured coefficient")
    fig.suptitle(f"Nd={nd}, {steps}-step forks · D/E average ages with shared-fork uncertainty", fontsize=11)
    # Stage identities are row labels; the legend only needs rule colors here.
    add_legends(fig, points, {})
    fig.tight_layout(rect=(0, 0.8 / height, 1, 1 - 0.6 / height))
    save_figure(fig, directory, "shared_CDE_by_stage")
    plt.close(fig)


def coefficient_plots(points, directory, stages, nd, steps, options=PlotOptions()):
    """Render A/B and C/D/E with local Gaussian predictions and no fitted slopes."""
    import matplotlib.pyplot as plt

    independent = [r for r in points if r["condition"] == "independent" and not r["age_averaged"]]
    if independent:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, kind, title in zip(axes, ("drift", "diagonal"), ("A · Independent drift", "B · Independent diffusion")):
            rows = [r for r in independent if r["kind"] == kind]
            draw_points(ax, rows, stages, x="overlap", options=options)
            decay_rows = [r for r in rows if r["rule"].endswith("_decay")]
            draw_points(ax, decay_rows, stages, x="overlap", options=options, theory_only=True)
            low, high = min(r["overlap"] for r in rows), max(r["overlap"] for r in rows)
            grid = np.linspace(low-0.05*(high-low+1), high+0.05*(high-low+1), 100)
            for rule, color in COLORS.items():
                if rule.endswith("_decay"):
                    continue  # Decay predictions depend on actual norms and are shown as x markers.
                diffusion = [r["predicted"] for r in independent if r["rule"] == rule and r["kind"] == "diagonal"]
                if diffusion:
                    # Averaging numerical integrations is not a fit to measured coefficients.
                    d = np.mean(diffusion)
                    ax.plot(grid, -0.5*d*grid if kind == "drift" else np.full_like(grid, d), color=color, lw=1.5)
            ax.set(title=title, xlabel="Old-pattern overlap m", ylabel="Measured drift" if kind == "drift" else "Measured diffusion")
            decorate(ax)
        reference = "spherical theory: lines; decay theory: x markers" if any(r["rule"].endswith("_decay") for r in independent) else "lines: local R4 predictions"
        fig.suptitle(f"Nd={nd}, {steps}-step forks · {reference}", fontsize=11)
        add_legends(fig, independent, {t: s for t, s in stages.items() if any(r["t"] == t for r in independent)})
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        save_figure(fig, directory, "independent_AB")
        plt.close(fig)
    shared = [r for r in points if r["condition"] == "shared"]
    if shared:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        limits = []
        for ax, kind, title in zip(axes, ("drift", "diagonal", "cross"),
                                   ("C · Shared drift", "D · Shared diagonal diffusion", "E · Shared cross-diffusion")):
            limits.append(shared_coefficient_panel(ax, shared, stages, kind, title, options=options))
        fig.suptitle(f"Nd={nd}, {steps}-step forks · D/E average ages with shared-fork uncertainty", fontsize=11)
        add_legends(fig, shared, {t: s for t, s in stages.items() if any(r["t"] == t for r in shared)})
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        save_figure(fig, directory, "shared_CDE")
        plt.close(fig)
        shared_stage_plots(shared, directory, stages, nd, steps, limits, options=options)


def diagnostic_plots(points, snapshots, directory, stages, options=PlotOptions()):
    """Show age-resolved residuals and the actual Gram states tested."""
    import matplotlib.pyplot as plt

    for condition in ("independent", "shared"):
        rows = [r for r in points if r["condition"] == condition and not r["age_averaged"]]
        if not rows:
            continue
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for i, kind in enumerate(("drift", "diagonal", "cross")):
            subset = [r for r in rows if r["kind"] == kind]
            for j, x in enumerate(("age", "overlap", "t")):
                ax = axes[i, j]
                if not subset:
                    ax.set_visible(False)
                    continue
                draw_points(ax, subset, stages, x=x, residual=True, options=options)
                ax.axhline(0, color="black", linestyle="--", lw=0.8)
                ax.set(xlabel={"age": "Memory age", "overlap": "Old-pattern overlap", "t": "Training stage"}[x],
                       ylabel=f"{kind.capitalize()} residual")
                decorate(ax)
        intervals = "conditional 95% intervals" if options.error_bars else "error bars hidden"
        fig.suptitle(f"{condition.capitalize()}: measured minus local prediction · {intervals}")
        add_legends(fig, rows, {t: s for t, s in stages.items() if any(r["t"] == t for r in rows)})
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        save_figure(fig, directory, f"{condition}_residuals")
        plt.close(fig)
    shared = [s for s in snapshots if s["condition"] == "shared"]
    if shared:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, key, label in zip(axes, ("R", "off_diagonal_rms", "rank"), ("R", "Off-diagonal RMS Q", "Rank Q")):
            for rule, color in COLORS.items():
                for seed in sorted({s["seed"] for s in shared}):
                    rows = sorted([s for s in shared if s["rule"]["name"] == rule and s["seed"] == seed], key=lambda s: s["stage"])
                    ax.plot([s["stage"] for s in rows], [s[key] for s in rows], color=color, alpha=0.5, lw=0.8)
                    for s in rows:
                        ax.plot(s["stage"], s[key], marker=stages[s["stage"]], color=color, markersize=options.point_size)
            ax.set(xlabel="Training stage", ylabel=label)
            decorate(ax)
        add_legends(fig, points, stages)
        fig.tight_layout(rect=(0, 0.22, 1, 1))
        save_figure(fig, directory, "gram_diagnostics")
        plt.close(fig)


def main(argv=None):
    """Make plots without rerunning any simulation or Gaussian integration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="Default: first run directory / figures")
    parser.add_argument("--max-points", type=int, help="Maximum displayed points per rule/stage/panel; default: all")
    parser.add_argument("--sample-seed", type=int, default=0, help="Reproducible display subsampling seed")
    parser.add_argument("--point-size", type=float, default=3.0, help="Marker size in points (default: 3)")
    parser.add_argument("--no-error-bars", action="store_true", help="Hide horizontal and vertical point error bars")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--rules", nargs="+", choices=list(COLORS),
                           help="Show only these rules (default: all saved rules)")
    selection.add_argument("--exclude-rules", nargs="+", choices=list(COLORS),
                           help="Show all saved rules except these")
    args = parser.parse_args(argv)
    try:
        if args.max_points is not None and args.max_points < 1:
            raise ValueError("--max-points must be positive")
        if not np.isfinite(args.point_size) or args.point_size <= 0:
            raise ValueError("--point-size must be finite and positive")
        if args.sample_seed < 0:
            raise ValueError("--sample-seed must be nonnegative")
        options = PlotOptions(args.max_points, args.sample_seed, args.point_size, not args.no_error_bars)
        selected = args.rules
        if args.exclude_rules is not None:
            selected = [rule for rule in COLORS if rule not in args.exclude_rules]
        if selected == []:
            raise ValueError("Select at least one rule to plot")
        import matplotlib
        matplotlib.use("Agg")
        points, snapshots = load_runs(args.runs, rules=selected)
        root = args.output or args.runs[0] / "figures"
        for nd, steps in sorted({(r["nd"], r["fork_steps"]) for r in points}):
            rows = [r for r in points if (r["nd"], r["fork_steps"]) == (nd, steps)]
            metadata = [s for s in snapshots if (s["rule"]["nd"], s["fork_steps"]) == (nd, steps)]
            times = sorted({r["t"] for r in rows})
            if len(times) > len(MARKERS):
                raise ValueError(f"At most {len(MARKERS)} distinct stages can be shown together")
            stages = dict(zip(times, MARKERS))
            directory = root / f"nd{nd}_steps{steps}"
            directory.mkdir(parents=True, exist_ok=True)
            coefficient_plots(rows, directory, stages, nd, steps, options=options)
            diagnostic_plots(rows, metadata, directory, stages, options=options)
            (directory / "plot_options.json").write_text(json.dumps(dict(
                max_points=options.max_points, sample_seed=options.sample_seed,
                point_size=options.point_size, error_bars=options.error_bars,
                rules=selected, runs=[str(path.resolve()) for path in args.runs]), indent=2) + "\n")
            print(f"Saved figures to {directory}")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
