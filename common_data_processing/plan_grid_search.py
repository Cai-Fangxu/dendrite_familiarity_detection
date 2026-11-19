"""Aggregate capacity sweeps and propose the next b/bk grid extensions.

Run from any working directory, for example:
    python common_data_processing/plan_grid_search.py tmp_data/int_variable_n_min1

Only NumPy and the Python standard library are required. This script does not
calculate beta or run simulations. See common_data_processing/README.md.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
import warnings

import numpy as np


@dataclass(frozen=True, order=True)
class GridKey:
    """Identify one grid independently of other geometries and winner counts."""

    nd: int
    ns: int
    n: int


@dataclass(frozen=True)
class Settings:
    """Scan dimensions must match those used by the simulation runner."""

    top_k: int = 4
    b_spacing: float = 0.15
    bk_spacing: float = 0.25
    b_size: int = 9
    bk_size: int = 5
    n_trials: int = 6

    def __post_init__(self):
        for name in ("top_k", "n_trials"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("b_size", "bk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer")
        for name in ("b_spacing", "bk_spacing"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Point:
    """Capacity mean and sample SEM for one b/bk/beta parameter point."""

    b: float
    bk: float
    beta: float
    capacity: float
    sem: float
    n_samples: int


def load_grids(data_dir, settings):
    """Read direct nd*_ns*/n*.csv children and pool repeated parameter points.

    A file must have exactly three parameter columns followed by n_trials
    capacity columns. Malformed input raises an error instead of silently
    producing an incomplete plan.
    """
    grids = {}
    for path in sorted(Path(data_dir).glob("nd*_ns*/n*.csv")):
        geometry = re.fullmatch(r"nd(\d+)_ns(\d+)", path.parent.name)
        winner = re.fullmatch(r"n(\d+)\.csv", path.name)
        if geometry is None or winner is None:
            continue
        key = GridKey(int(geometry[1]), int(geometry[2]), int(winner[1]))
        if min(key.nd, key.ns, key.n) < 1 or key.n > key.nd:
            raise ValueError(f"{path}: expected nd >= n >= 1 and ns >= 1")
        if path.stat().st_size == 0:
            raise ValueError(f"{path}: empty result file")
        try:
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError as exc:
            raise ValueError(f"{path}: cannot read numeric CSV: {exc}") from exc
        if rows.shape[1] != 3 + settings.n_trials:
            raise ValueError(
                f"{path}: expected {3 + settings.n_trials} columns "
                f"(b,bk,beta and {settings.n_trials} capacities), got {rows.shape[1]}"
            )
        if not np.isfinite(rows).all() or (rows[:, 3:] < 0).any():
            raise ValueError(f"{path}: values must be finite; capacities must be nonnegative")
        if key in grids:
            raise ValueError(f"Multiple files resolve to the same grid: {key}")

        samples = {}
        beta_by_coordinate = {}
        for row in rows:
            b, bk, beta = map(float, row[:3])
            coordinate = (b, bk)
            if coordinate in beta_by_coordinate and beta_by_coordinate[coordinate] != beta:
                raise ValueError(
                    f"{path}: different beta values at b={b}, bk={bk}; "
                    "separate these experiments before planning a two-dimensional grid"
                )
            beta_by_coordinate[coordinate] = beta
            samples.setdefault((b, bk, beta), []).extend(row[3:].tolist())

        points = []
        for (b, bk, beta), capacities in sorted(samples.items()):
            mean = float(np.mean(capacities))
            sem = float(np.std(capacities, ddof=1) / np.sqrt(len(capacities))) if len(capacities) > 1 else 0.0
            points.append(Point(b, bk, beta, mean, sem, len(capacities)))
        if len(points) < len(rows):
            warnings.warn(
                f"{path}: pooled repeated parameter rows; confirm their trials are intended repeats",
                stacklevel=2,
            )
        grids[key] = points
    if not grids:
        raise ValueError(f"No nd{{nd}}_ns{{ns}}/n{{n}}.csv result files found under {data_dir}")
    return grids


class Grid:
    """Use integer lattice coordinates to avoid floating-point edge comparisons."""

    def __init__(self, points, settings):
        if not points:
            raise ValueError("Cannot plan an empty grid")
        self.points = points
        self.settings = settings
        self.b_origin = min(point.b for point in points)
        self.bk_origin = min(point.bk for point in points)
        self.coordinates = set()
        for point in points:
            self.coordinates.add((self.b_index(point.b), self.bk_index(point.bk)))
        if len(self.coordinates) != len(points):
            raise ValueError("Multiple parameter points resolve to the same b/bk grid location")
        self.b_max_index = max(b for b, _ in self.coordinates)

    @staticmethod
    def _index(value, origin, spacing):
        position = (value - origin) / spacing
        index = round(position)
        if not math.isclose(position, index, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"Value {value} does not align with origin {origin}, spacing {spacing}")
        return index

    def b_index(self, value):
        return self._index(value, self.b_origin, self.settings.b_spacing)

    def bk_index(self, value):
        return self._index(value, self.bk_origin, self.settings.bk_spacing)

    def b_value(self, index):
        return round(self.b_origin + index * self.settings.b_spacing, 12)

    def bk_value(self, index):
        return round(self.bk_origin + index * self.settings.bk_spacing, 12)

    def ranked_points(self):
        """Represent exact capacity ties at each bk by their largest b."""
        representatives = {}
        for point in self.points:
            key = (self.bk_index(point.bk), point.capacity)
            previous = representatives.get(key)
            if previous is None or point.b > previous.b:
                representatives[key] = point
        return sorted(
            representatives.values(),
            key=lambda point: (-point.capacity, -point.b, point.bk),
        )

    def strip(self, point):
        """Choose a b strip containing the point, preferring the right edge.

        This matches the notebook's left/right strips for one or two blocks.
        An interior point in a wider grid uses a centered strip instead of
        being checked against a distant left edge that does not contain it.
        """
        width = self.settings.b_size - 1
        position = self.b_index(point.b)
        if position >= self.b_max_index - width:
            start = max(0, self.b_max_index - width)
        elif position <= width:
            start = 0
        else:
            start = position - width // 2
        return start, start + width

    def proposal(self, center_index, bk_index, reason, trigger):
        """Build a complete scan, or return None if every point already exists."""
        minimum_center = math.ceil(-self.b_origin / self.settings.b_spacing - 1e-5)
        center_index = max(center_index, minimum_center)
        half_width = self.settings.b_size // 2
        indices = range(center_index - half_width, center_index + half_width + 1)
        n_existing = sum((b, bk_index) in self.coordinates for b in indices)
        if n_existing == self.settings.b_size:
            return None
        return {
            "b_center": self.b_value(center_index),
            "bk": self.bk_value(bk_index),
            "b_min": self.b_value(center_index - half_width),
            "b_max": self.b_value(center_index + half_width),
            "n_new": self.settings.b_size - n_existing,
            "n_existing": n_existing,
            "reason": reason,
            "trigger": asdict(trigger),
        }


def extend_bk(grid, top_points):
    """Check bk boundaries independently in each strip of the complete grid."""
    proposals = []
    boundary_hit = False
    edge_counts = {}
    for point in top_points:
        start, end = grid.strip(point)
        rows = [bk for b, bk in grid.coordinates if start <= b <= end]
        lower, upper = min(rows), max(rows)
        point_bk = grid.bk_index(point.bk)
        for direction, edge, label in ((1, upper, "bk_upper"), (-1, lower, "bk_lower")):
            if point_bk != edge:
                continue
            boundary_hit = True
            counter_key = (start, end, direction)
            edge_counts[counter_key] = edge_counts.get(counter_key, 0) + 1
            next_bk = edge + direction * edge_counts[counter_key]
            proposal = grid.proposal(start + grid.settings.b_size // 2, next_bk, label, point)
            if proposal is not None:
                proposals.append(proposal)
    return boundary_hit, proposals


def extend_b(grid, top_points):
    """Extend the first actionable b boundary with a neighborhood of bk rows."""
    limited = False
    half_b = grid.settings.b_size // 2
    half_bk = grid.settings.bk_size // 2
    for point in top_points:
        position = grid.b_index(point.b)
        if position == grid.b_max_index:
            center = position + half_b + 1
            reason = "b_upper"
        elif position == 0:
            center = position - half_b - 1
            reason = "b_lower"
        else:
            continue
        proposals = []
        for offset in range(-half_bk, half_bk + 1):
            proposal = grid.proposal(center, grid.bk_index(point.bk) + offset, reason, point)
            if proposal is not None:
                proposals.append(proposal)
        if proposals:
            return "extend_b", proposals
        # Clamping at the lowest legal center can leave an already sampled block.
        limited = True
    return ("b_center_limit" if limited else "complete"), []


def plan_grid(points, settings):
    """Apply the all-zero check, plateau policy, bk gate, and then b check."""
    grid = Grid(points, settings)
    top_points = grid.ranked_points()[:settings.top_k]
    report = {
        "n_points": len(points),
        "top_points": [asdict(point) for point in top_points],
        "proposals": [],
    }
    if all(point.capacity == 0 for point in points):
        report["status"] = "skipped_all_zero"
        return report
    failed_bk, proposals = extend_bk(grid, top_points)
    if failed_bk:
        status = "extend_bk" if proposals else "bk_blocked"
    else:
        status, proposals = extend_b(grid, top_points)

    unique = {}
    for proposal in proposals:
        unique.setdefault((proposal["b_center"], proposal["bk"]), proposal)
    report["status"] = status
    report["proposals"] = sorted(unique.values(), key=lambda p: (p["b_center"], p["bk"]))
    return report


def analyze(grids, settings):
    """Plan each grid independently and attach geometry to the audit report."""
    reports = []
    for key, points in sorted(grids.items()):
        try:
            report = plan_grid(points, settings)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        report.update(asdict(key))
        if report["status"] in ("skipped_all_zero", "bk_blocked", "b_center_limit"):
            warnings.warn(f"{key}: {report['status']}", stacklevel=2)
        reports.append(report)
    return reports


def write_outputs(output_dir, grids, reports, settings, data_dir):
    """Write summaries, a headerless Mathematica input table, and audit JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["nd", "ndR", "b", "bk", "beta", "capacity", "sem", "ns", "n_samples"]
    all_rows = []
    best_rows = []
    for key, points in sorted(grids.items()):
        def row(point):
            return dict(
                nd=key.nd, ndR=key.n, ns=key.ns,
                **asdict(point),
            )
        all_rows.extend(row(point) for point in points)
        ranked = Grid(points, settings).ranked_points()
        best_rows.extend(row(point) for point in ranked if point.capacity == ranked[0].capacity)
    for name, rows in (("tmp_df.csv", all_rows), ("max_points_df.csv", best_rows)):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "tmp.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for report in reports:
            for proposal in report["proposals"]:
                writer.writerow([
                    format(proposal["b_center"], ".12g"),
                    format(proposal["bk"], ".12g"),
                    report["n"], report["nd"], report["ns"],
                ])
    audit = {
        "data_dir": str(Path(data_dir).resolve()),
        "settings": asdict(settings),
        "grids": reports,
    }
    with (output_dir / "grid_search_report.json").open("w") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    """Run the command-line planner; --dry-run performs no filesystem writes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="Experiment root containing nd{nd}_ns{ns}/n{n}.csv")
    parser.add_argument("--output-dir", type=Path, help="Default: DATA_DIR/next_grid; existing output files are replaced")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--b-spacing", type=float, default=0.15)
    parser.add_argument("--bk-spacing", type=float, default=0.25)
    parser.add_argument("--b-size", type=int, default=9, help="Odd number of b values per simulation row")
    parser.add_argument("--bk-size", type=int, default=5, help="Odd number of bk rows for a b extension")
    parser.add_argument("--n-trials", type=int, default=6, help="Capacity columns per raw row")
    parser.add_argument("--dry-run", action="store_true", help="Print proposals without writing files")
    args = parser.parse_args(argv)
    try:
        settings = Settings(
            args.top_k, args.b_spacing, args.bk_spacing,
            args.b_size, args.bk_size, args.n_trials,
        )
        grids = load_grids(args.data_dir, settings)
        reports = analyze(grids, settings)
        output_dir = args.output_dir or args.data_dir / "next_grid"
        for report in reports:
            print(
                f"nd={report['nd']} ns={report['ns']} n={report['n']}: "
                f"{report['status']} ({len(report['proposals'])} scans)"
            )
            for proposal in report["proposals"]:
                print(
                    f"  {proposal['reason']}: b_center={proposal['b_center']:.6g}, "
                    f"bk={proposal['bk']:.6g}, "
                    f"b=[{proposal['b_min']:.6g}, {proposal['b_max']:.6g}], "
                    f"{proposal['n_new']} new/{proposal['n_existing']} existing points"
                )
        if not args.dry_run:
            write_outputs(output_dir, grids, reports, settings, args.data_dir)
            print(f"Wrote tmp_df.csv, max_points_df.csv, tmp.csv, grid_search_report.json to {output_dir}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
