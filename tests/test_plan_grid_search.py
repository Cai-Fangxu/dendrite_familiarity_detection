"""Behavioral tests for grid-search planning; no simulations are launched."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from common_data_processing.plan_grid_search import (
    Grid, GridKey, Point, Settings, analyze, load_grids, main, plan_grid, write_outputs,
)


def make_grid(peaks=None, b_origin=0.6, b_count=9, bk_count=5):
    """Make a sampled rectangle with distinct low background capacities."""
    peaks = peaks or {}
    return [
        Point(
            round(b_origin + i * 0.15, 12),
            round(2.0 + j * 0.25, 12),
            5.0,
            peaks.get((i, j), 1.0 + i * 0.001 + j * 0.00001),
            0.1,
            6,
        )
        for i in range(b_count)
        for j in range(bk_count)
    ]


class PlanningTests(unittest.TestCase):
    def test_bk_failure_prevents_b_extension(self):
        result = plan_grid(make_grid({(8, 4): 20.0}), Settings(top_k=1))
        self.assertEqual(result["status"], "extend_bk")
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        self.assertEqual((proposal["b_center"], proposal["bk"]), (1.2, 3.25))
        self.assertEqual(proposal["reason"], "bk_upper")

    def test_configurable_top_k_controls_number_of_bk_rows(self):
        points = make_grid({(8, 4): 20.0, (7, 4): 19.0})
        one = plan_grid(points, Settings(top_k=1))
        two = plan_grid(points, Settings(top_k=2))
        self.assertEqual([p["bk"] for p in one["proposals"]], [3.25])
        self.assertEqual([p["bk"] for p in two["proposals"]], [3.25, 3.5])

    def test_b_extension_is_adjacent_block_with_five_bk_rows(self):
        result = plan_grid(make_grid({(8, 2): 20.0}), Settings(top_k=1))
        self.assertEqual(result["status"], "extend_b")
        self.assertEqual([p["bk"] for p in result["proposals"]], [2.0, 2.25, 2.5, 2.75, 3.0])
        for proposal in result["proposals"]:
            self.assertEqual(proposal["b_center"], 2.55)
            self.assertEqual((proposal["b_min"], proposal["b_max"]), (1.95, 3.15))
            self.assertEqual(proposal["n_new"], 9)

    def test_plateau_uses_rightmost_representative_before_top_k(self):
        points = make_grid({(0, 2): 20.0, (4, 2): 20.0, (6, 1): 19.0})
        result = plan_grid(points, Settings(top_k=2))
        self.assertEqual(result["status"], "complete")
        self.assertEqual([p["b"] for p in result["top_points"]], [1.2, 1.5])
        self.assertEqual(len(result["top_points"]), 2)

    def test_full_width_plateau_extends_right(self):
        points = make_grid({(i, 2): 20.0 for i in range(9)})
        result = plan_grid(points, Settings(top_k=1))
        self.assertEqual(result["status"], "extend_b")
        self.assertTrue(all(p["reason"] == "b_upper" for p in result["proposals"]))

    def test_lower_extension_clamps_center_and_reports_overlap(self):
        result = plan_grid(make_grid({(0, 2): 20.0}), Settings(top_k=1))
        for proposal in result["proposals"]:
            self.assertEqual(proposal["b_center"], 0.0)
            self.assertEqual(proposal["reason"], "b_lower")
            self.assertEqual((proposal["n_new"], proposal["n_existing"]), (8, 1))

    def test_lower_limit_does_not_repropose_existing_blocks(self):
        points = make_grid({(0, 2): 20.0}, b_origin=-0.6)
        result = plan_grid(points, Settings(top_k=1))
        self.assertEqual(result["status"], "b_center_limit")
        self.assertEqual(result["proposals"], [])

    def test_offset_lattice_stays_aligned_when_center_is_clamped(self):
        result = plan_grid(make_grid({(0, 2): 20.0}, b_origin=0.65), Settings(top_k=1))
        self.assertTrue(all(p["b_center"] == 0.05 for p in result["proposals"]))

    def test_left_and_right_strips_use_independent_data(self):
        points = []
        for i in range(18):
            for j in (range(5) if i < 9 else range(2, 7)):
                capacity = 30.0 if (i, j) == (0, 4) else 20.0 if (i, j) == (17, 2) else 1.0
                points.append(Point(0.6 + i * 0.15, 2.0 + j * 0.25, 5, capacity, 0, 6))
        result = plan_grid(points, Settings(top_k=2))
        self.assertEqual(result["status"], "extend_bk")
        self.assertEqual(
            [(p["b_center"], p["bk"]) for p in result["proposals"]],
            [(1.2, 3.25), (2.55, 2.25)],
        )

    def test_middle_strip_in_wide_grid_contains_trigger(self):
        grid = Grid(make_grid({(13, 4): 20.0}, b_count=27), Settings(top_k=1))
        self.assertEqual(grid.strip(grid.ranked_points()[0]), (9, 17))
        result = plan_grid(grid.points, grid.settings)
        self.assertEqual(result["proposals"][0]["b_center"], 2.55)

    def test_all_zero_grid_warns_and_skips_without_excluding_n_one(self):
        points = make_grid({(i, j): 0.0 for i in range(9) for j in range(5)})
        with self.assertWarnsRegex(UserWarning, "skipped_all_zero"):
            reports = analyze({GridKey(10, 100, 1): points}, Settings())
        self.assertEqual(reports[0]["proposals"], [])
        active = analyze({GridKey(10, 100, 1): make_grid({(8, 4): 20.0})}, Settings(top_k=1))
        self.assertTrue(active[0]["proposals"])

    def test_gate_is_applied_per_grid(self):
        reports = analyze({
            GridKey(10, 100, 1): make_grid({(8, 4): 20.0}),
            GridKey(10, 100, 2): make_grid({(8, 2): 20.0}),
        }, Settings(top_k=1))
        self.assertEqual([r["status"] for r in reports], ["extend_bk", "extend_b"])

    def test_grid_alignment_errors_are_not_silently_ignored(self):
        points = make_grid()
        points[0] = Point(0.61, 2.0, 5, 10, 0, 6)
        with self.assertRaisesRegex(ValueError, "does not align"):
            plan_grid(points, Settings())

    def test_scan_width_can_match_thirteen_point_runner(self):
        result = plan_grid(make_grid({(12, 2): 20.0}, b_count=13), Settings(top_k=1, b_size=13))
        self.assertEqual(result["proposals"][0]["b_center"], 3.45)
        self.assertEqual(result["proposals"][0]["n_new"], 13)

    def test_invalid_settings(self):
        for kwargs in ({"top_k": 0}, {"b_size": 8}, {"bk_size": 0}, {"b_spacing": 0}, {"bk_spacing": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)


class FileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "nd10_ns100" / "n1.csv"
        self.raw.parent.mkdir()

    def write_raw(self, rows):
        np.savetxt(self.raw, rows, delimiter=",")

    def test_single_row_and_sem(self):
        self.write_raw([[0.6, 2, 5, 1, 2, 3, 4, 5, 6]])
        point = load_grids(self.root, Settings())[GridKey(10, 100, 1)][0]
        self.assertEqual(point.capacity, 3.5)
        self.assertAlmostEqual(point.sem, np.std([1, 2, 3, 4, 5, 6], ddof=1) / np.sqrt(6))

    def test_duplicate_parameter_rows_pool_trials(self):
        self.write_raw([[0.6, 2, 5, *([1] * 6)], [0.6, 2, 5, *([3] * 6)]])
        with self.assertWarnsRegex(UserWarning, "pooled"):
            points = load_grids(self.root, Settings())[GridKey(10, 100, 1)]
        self.assertEqual(len(points), 1)
        self.assertEqual((points[0].capacity, points[0].n_samples), (2, 12))

    def test_different_betas_at_same_coordinate_are_rejected(self):
        self.write_raw([[0.6, 2, 5, *([1] * 6)], [0.6, 2, 6, *([3] * 6)]])
        with self.assertRaisesRegex(ValueError, "different beta"):
            load_grids(self.root, Settings())

    def test_wrong_schema_and_nonfinite_data_are_rejected(self):
        for row in ([0] * 10, [0.6, 2, 5, float("nan"), 1, 1, 1, 1, 1]):
            self.write_raw([row])
            with self.assertRaises(ValueError):
                load_grids(self.root, Settings())

    def test_cli_writes_five_column_candidates_and_audit(self):
        points = make_grid({(8, 4): 20.0})
        self.write_raw([[p.b, p.bk, p.beta, *([p.capacity] * 6)] for p in points])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--top-k", "1", "--dry-run"]), 0)
        output = self.root / "next_grid"
        self.assertFalse(output.exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--top-k", "1"]), 0)
        with (output / "tmp.csv").open() as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows, [["1.2", "3.25", "1", "10", "100"]])
        report = json.loads((output / "grid_search_report.json").read_text())
        self.assertEqual(report["settings"]["top_k"], 1)
        self.assertEqual(report["grids"][0]["status"], "extend_bk")
        with (output / "max_points_df.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["ndR"], "1")

    def test_empty_plan_clears_stale_candidate_file(self):
        output = self.root / "output"
        output.mkdir()
        (output / "tmp.csv").write_text("stale candidates")
        settings = Settings(top_k=1)
        grids = {GridKey(10, 100, 1): make_grid({(4, 2): 20.0})}
        reports = analyze(grids, settings)
        write_outputs(output, grids, reports, settings, self.root)
        self.assertEqual((output / "tmp.csv").read_text(), "")


if __name__ == "__main__":
    unittest.main()
