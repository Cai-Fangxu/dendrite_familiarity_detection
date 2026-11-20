"""Aggregate Softmax capacities and extend a linear b/eta grid.

Input: nd{nd}_ns{ns}/beta_select{value}.csv with b, eta, -1, capacities.
Exports a ready-to-run seven-column table, without a decay calculation.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_data_processing import plan_grid_search as base
from common_data_processing import softmax_calibration_registry as registry


class SoftmaxGrid(base.Grid):
    """Apply the existing boundary policy on a strictly positive eta lattice."""

    def valid_bk_index(self, index):
        return self.bk_value(index) > 0


def rename_fields(value):
    """Translate the shared planner's internal bk terminology at the file boundary."""
    if isinstance(value, dict):
        return {("eta" if key == "bk" else "eta_spacing" if key == "bk_spacing"
                 else "eta_size" if key == "bk_size" else key): rename_fields(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [rename_fields(item) for item in value]
    if isinstance(value, str):
        return value.replace("bk", "eta")
    return value


def load_eta_scales(grids, report_paths):
    """Recover target n_eff by (nd, beta_select) from saved calibration reports.

    Use the requested target_n, not the Monte Carlo effective_n estimate.
    Multiple reports may cover different batches; conflicting mappings are rejected.
    """
    records = registry.merge_calibrations(*(registry.read_calibrations(path) for path in report_paths))
    scales = {(int(result["nd"]), float(result["beta_select"])): float(result["target_n"])
              for result in records}
    missing = {(key.nd, key.beta_select) for key in grids} - scales.keys()
    if missing:
        raise ValueError(f"Calibration reports/registry lack target_n for {sorted(missing)}")
    return {key: scales[key.nd, key.beta_select] for key in grids}


def grid_settings(key, settings, eta_scales=None):
    """Convert reference spacing to actual eta spacing for this grid only."""
    if eta_scales is None:
        return settings
    n_eff = eta_scales[key]
    if not math.isfinite(n_eff) or not 1 <= n_eff <= key.nd:
        raise ValueError(f"{key}: n_eff must be finite and in [1, nd]")
    return replace(settings, bk_spacing=settings.bk_spacing * n_eff)


def analyze(grids, settings, eta_scales=None):
    """Plan each fixed beta_select separately, including the lower eta constraint."""
    reports = []
    for key, points in sorted(grids.items()):
        actual_settings = grid_settings(key, settings, eta_scales)
        report = rename_fields(base.plan_grid(points, actual_settings, grid_class=SoftmaxGrid))
        if eta_scales is not None:
            report.update(n_eff=eta_scales[key], eta_spacing=actual_settings.bk_spacing)
        report.update(asdict(key))
        if report["status"] in ("skipped_all_zero", "eta_blocked", "b_center_limit"):
            warnings.warn(f"{key}: {report['status']}", stacklevel=2)
        reports.append(report)
    return reports


def write_outputs(output_dir, grids, reports, settings, data_dir, seed, eta_scales=None):
    """Write explicit Softmax summaries, runnable parameter rows, and audit metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, best_rows = [], []
    for key, points in sorted(grids.items()):
        def row(point):
            return dict(**asdict(key), **rename_fields(asdict(point)))
        all_rows.extend(row(point) for point in points)
        ranked = SoftmaxGrid(points, grid_settings(key, settings, eta_scales)).ranked_points()
        best_rows.extend(row(point) for point in ranked if point.capacity == ranked[0].capacity)
    fields = ["nd", "ns", "beta_select", "b", "eta", "beta", "capacity", "sem", "n_samples"]
    for name, rows in (("tmp_df.csv", all_rows), ("max_points_df.csv", best_rows)):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "params_list_softmax.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for report in reports:
            for proposal in report["proposals"]:
                writer.writerow([
                    report["nd"], report["ns"], proposal["eta"], proposal["b_center"],
                    report["beta_select"], -1, seed,
                ])
    settings_report = rename_fields(asdict(settings))
    if eta_scales is not None:
        settings_report["eta_reference_spacing"] = settings_report.pop("eta_spacing")
    audit = dict(data_dir=str(Path(data_dir).resolve()),
                 settings=settings_report, seed=seed, grids=reports,
                 eta_scaling="target_n" if eta_scales is not None else "actual")
    (output_dir / "grid_search_report.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    """Run the Softmax planner; dry runs never write files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--b-spacing", type=float, default=0.15)
    spacing = parser.add_mutually_exclusive_group(required=True)
    spacing.add_argument("--eta-spacing", type=float, help="Spacing in actual eta units")
    spacing.add_argument("--eta-reference-spacing", type=float,
                         help="Reference spacing, multiplied by each grid's target n_eff")
    parser.add_argument("--calibration-report", type=Path, action="append",
                        help="Additional batch report or registry; linked and sibling registries are included")
    parser.add_argument("--registry", type=Path,
                        help="Override automatic discovery of softmax_calibrations.json above data_dir")
    parser.add_argument("--b-size", type=int, default=9)
    parser.add_argument("--eta-size", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0, help="Seed for proposed rows; match the original sweep for paired comparisons")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.calibration_report or args.registry) and args.eta_reference_spacing is None:
        parser.error("--calibration-report and --registry require --eta-reference-spacing")
    try:
        if not 0 <= args.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        eta_spacing = args.eta_spacing if args.eta_reference_spacing is None else args.eta_reference_spacing
        settings = base.Settings(args.top_k, args.b_spacing, eta_spacing,
                                 args.b_size, args.eta_size, args.n_trials)
        grids = base.load_grids(args.data_dir, settings, softmax=True)
        eta_scales = None
        if args.eta_reference_spacing is not None:
            sources = registry.discover_sources(args.data_dir, args.calibration_report or (), args.registry)
            if not sources:
                raise ValueError("No calibration registry found; supply --registry or --calibration-report")
            eta_scales = load_eta_scales(grids, sources)
        reports = analyze(grids, settings, eta_scales=eta_scales)
        for report in reports:
            print(f"nd={report['nd']} ns={report['ns']} beta_select={report['beta_select']}: {report['status']}")
            for proposal in report["proposals"]:
                print(f"  {proposal['reason']}: b_center={proposal['b_center']:.12g}, eta={proposal['eta']:.12g}")
        if not args.dry_run:
            output_dir = args.output_dir or args.data_dir / "next_grid"
            write_outputs(output_dir, grids, reports, settings, args.data_dir, args.seed, eta_scales=eta_scales)
            print(f"Wrote summaries, params_list_softmax.csv, and grid_search_report.json to {output_dir}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
