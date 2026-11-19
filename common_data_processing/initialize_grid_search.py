"""Start a new b/bk grid around maxima from an existing experiment.

Accepts an experiment directory containing nd{nd}_ns{ns}/n{n}.csv files, or
a summary CSV with nd, ndR (or n), b, bk, capacity, and optionally ns.
Writes headerless b_center,bk,n,nd,ns rows for Mathematica, not runner inputs.
"""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import warnings

# Support both direct execution from any directory and package imports.
if __package__:
    from .plan_grid_search import GridKey, Settings, load_grids
else:
    from plan_grid_search import GridKey, Settings, load_grids


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tmp_data" / "tmp.csv"


def _integer(value, name):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def load_summary(path, total_synapses=None):
    """Read modern summaries or infer missing ns from an explicit total."""
    if total_synapses is not None:
        total_synapses = _integer(total_synapses, "total_synapses")
    grids = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"nd", "b", "bk", "capacity"} <= fields or not fields & {"n", "ndR"}:
            raise ValueError("Summary requires nd, n (or ndR), b, bk, capacity")
        if "ns" not in fields and total_synapses is None:
            raise ValueError("Summary has no ns column; supply --total-synapses, e.g. 60000")
        for line_number, row in enumerate(reader, 2):
            try:
                nd = _integer(row["nd"], "nd")
                n = _integer(row["n"] if "n" in fields else row["ndR"], "n")
                if "n" in fields and "ndR" in fields and _integer(row["ndR"], "ndR") != n:
                    raise ValueError("n and ndR disagree")
                if "ns" in fields:
                    ns = _integer(row["ns"], "ns")
                else:
                    if total_synapses % nd:
                        raise ValueError("total_synapses must be divisible by nd to infer ns")
                    ns = total_synapses // nd
                if n > nd:
                    raise ValueError("n cannot exceed nd")
                if total_synapses is not None and nd * ns != total_synapses:
                    raise ValueError("nd * ns does not match total_synapses")
                b, bk, capacity = (float(row[name]) for name in ("b", "bk", "capacity"))
                if not all(math.isfinite(value) for value in (b, bk, capacity)) or capacity < 0:
                    raise ValueError("b, bk, capacity must be finite; capacity must be nonnegative")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            # Only mean capacity is needed; do not invent repeat counts or SEMs.
            point = {"b": b, "bk": bk, "capacity": capacity}
            grids.setdefault(GridKey(nd, ns, n), []).append(point)
    if not grids:
        raise ValueError(f"{path}: no summary rows")
    return grids


def load_source(source, settings, total_synapses=None):
    """Load raw trials or already aggregated capacities without modifying them."""
    source = Path(source)
    if source.is_dir():
        grids = load_grids(source, settings)
        if total_synapses is not None:
            total = _integer(total_synapses, "total_synapses")
            if any(key.nd * key.ns != total for key in grids):
                raise ValueError("A source geometry does not match total_synapses")
        return {
            key: [{"b": p.b, "bk": p.bk, "capacity": p.capacity} for p in points]
            for key, points in grids.items()
        }
    return load_summary(source, total_synapses)


def initial_grid(grids, settings):
    """Choose one best source point per grid, then center a new bk neighborhood.

    Ties prefer the largest b, including plateaus at a fixed bk. If capacity
    and b both tie across bk values, the smaller bk wins deterministically.
    Centers are clamped and rounded onto the new, zero-anchored lattice,
    matching the notebook's initialization branch.
    """
    reports = []
    for key, points in sorted(grids.items()):
        if not points:
            raise ValueError(f"{key}: empty grid")
        best = min(points, key=lambda p: (-p["capacity"], -p["b"], p["bk"]))
        report = {**asdict(key), "source_best": best, "proposals": []}
        if best["capacity"] == 0:
            warnings.warn(f"{key}: all capacities are zero; skipping initial grid", stacklevel=2)
            report["status"] = "skipped_all_zero"
        else:
            center_index = round(max(0.0, best["b"]) / settings.b_spacing)
            bk_index = round(best["bk"] / settings.bk_spacing)
            b_center = round(center_index * settings.b_spacing, 12)
            report.update(
                status="initialized",
                b_center=b_center,
                bk_center=round(bk_index * settings.bk_spacing, 12),
                b_min=round((center_index - settings.b_size // 2) * settings.b_spacing, 12),
                b_max=round((center_index + settings.b_size // 2) * settings.b_spacing, 12),
            )
            for offset in range(-(settings.bk_size // 2), settings.bk_size // 2 + 1):
                report["proposals"].append({
                    "b_center": b_center,
                    "bk": round((bk_index + offset) * settings.bk_spacing, 12),
                })
        reports.append(report)
    return reports


def write_initial_grid(output, reports, settings, source):
    """Replace the candidate CSV and write a neighboring provenance report."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for report in reports:
            for proposal in report["proposals"]:
                writer.writerow([
                    format(proposal["b_center"], ".12g"),
                    format(proposal["bk"], ".12g"),
                    report["n"], report["nd"], report["ns"],
                ])
    report_path = output.with_suffix(".report.json")
    with report_path.open("w") as handle:
        json.dump({
            "source": str(Path(source).resolve()),
            "settings": asdict(settings),
            "grids": reports,
        }, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    """Generate initial grid centers; --dry-run performs no writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Raw experiment directory or summary CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Candidate CSV to replace (default: repository tmp_data/tmp.csv)")
    parser.add_argument("--total-synapses", type=int, help="Required for summaries without ns")
    parser.add_argument("--b-spacing", type=float, default=0.15)
    parser.add_argument("--bk-spacing", type=float, default=0.25)
    parser.add_argument("--b-size", type=int, default=9, help="Scan width used to report the b range")
    parser.add_argument("--bk-size", type=int, default=5, help="Odd number of bk rows around the optimum")
    parser.add_argument("--n-trials", type=int, default=6, help="Capacity columns in each raw row")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = Settings(
            b_spacing=args.b_spacing, bk_spacing=args.bk_spacing,
            b_size=args.b_size, bk_size=args.bk_size, n_trials=args.n_trials,
        )
        if args.output.suffix.lower() != ".csv":
            raise ValueError("--output must have a .csv extension")
        output = args.output.resolve()
        source = args.source.resolve()
        if source.is_file() and output == source:
            raise ValueError("Output must not overwrite the source summary")
        if source.is_dir() and output in {p.resolve() for p in source.glob("nd*_ns*/n*.csv")}:
            raise ValueError("Output must not overwrite a source result file")
        grids = load_source(args.source, settings, args.total_synapses)
        reports = initial_grid(grids, settings)
        for report in reports:
            label = f"nd={report['nd']} ns={report['ns']} n={report['n']}"
            if report["status"] == "skipped_all_zero":
                print(f"{label}: skipped_all_zero")
                continue
            best = report["source_best"]
            print(
                f"{label}: best b={best['b']:.6g}, bk={best['bk']:.6g}, "
                f"capacity={best['capacity']:.6g}; "
                f"b_center={report['b_center']:.6g}, "
                f"b=[{report['b_min']:.6g}, {report['b_max']:.6g}], "
                f"bk={[p['bk'] for p in report['proposals']]}"
            )
        count = sum(len(report["proposals"]) for report in reports)
        print(f"{count} candidate rows from {len(reports)} source grids")
        if not args.dry_run:
            write_initial_grid(args.output, reports, settings, args.source)
            print(f"Wrote {args.output} and {args.output.with_suffix('.report.json')}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
