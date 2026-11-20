"""Export maximum mean-capacity points from a legacy grid-search experiment folder.

Read nd{nd}_ns{ns}/n{n}.csv results and write max_points_df.csv
with the original bk and beta values. No grid spacing is needed.
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_data_processing.plan_grid_search import Settings, load_grids


FIELDS = ["nd", "ndR", "b", "bk", "beta", "capacity", "sem", "ns", "n_samples"]


def maximum_rows(grids):
    """Match the planner's plateau policy without requiring a regular lattice.

    At the maximum mean capacity, keep the largest b for each exact bk.
    Retain ties across bk values, including grids whose capacities are all zero.
    """
    rows = []
    for key, points in sorted(grids.items()):
        maximum = max(point.capacity for point in points)
        representatives = {}
        for point in points:
            if point.capacity != maximum:
                continue
            previous = representatives.get(point.bk)
            if previous is None or point.b > previous.b:
                representatives[point.bk] = point
        for point in sorted(representatives.values(), key=lambda p: (-p.b, p.bk)):
            rows.append(dict(
                nd=key.nd, ns=key.ns, ndR=key.n,
                b=point.b, bk=point.bk, beta=point.beta,
                capacity=point.capacity, sem=point.sem, n_samples=point.n_samples,
            ))
    return rows


def export_max_points(data_dir, output, n_trials=6):
    """Validate and aggregate raw results, then write a headered maximum table."""
    data_dir, output = Path(data_dir), Path(output)
    if not data_dir.is_dir():
        raise ValueError(f"Data folder does not exist or is not a directory: {data_dir}")
    sources = list(data_dir.glob("nd*_ns*/n*.csv"))
    if any(output.resolve() == source.resolve()
           or (output.exists() and output.samefile(source)) for source in sources):
        raise ValueError("Output must not overwrite a raw result file")
    # A new summary matching the raw-results glob would break subsequent reads.
    try:
        relative = output.resolve().relative_to(data_dir.resolve())
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) == 2 and relative.match("nd*_ns*/n*.csv"):
        raise ValueError("Output must not use a raw result filename under a geometry folder")

    grids = load_grids(data_dir, Settings(n_trials=n_trials))
    rows = maximum_rows(grids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    """Export legacy grid-search maxima to a user-selected CSV path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path,
                        help="Experiment folder containing nd*_ns* subfolders")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV path, for example tmp_data/max_points_df.csv")
    parser.add_argument("--n-trials", type=int, default=6,
                        help="Capacity columns per raw result row (default: 6)")
    args = parser.parse_args(argv)
    try:
        rows = export_max_points(args.data_dir, args.output, args.n_trials)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {len(rows)} maximum points to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
