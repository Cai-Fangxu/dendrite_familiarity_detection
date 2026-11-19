"""Tests for initializing a new search from an existing experiment."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from common_data_processing.initialize_grid_search import (
    initial_grid, load_source, load_summary, main,
)
from common_data_processing.plan_grid_search import GridKey, Settings


def point(b, bk, capacity):
    return {"b": b, "bk": bk, "capacity": capacity}


class InitializationTests(unittest.TestCase):
    def test_best_point_generates_five_bk_rows_and_keeps_n_one(self):
        key = GridKey(10, 6000, 1)
        reports = initial_grid({key: [point(1.2, 3, 100), point(1.5, 4, 80)]}, Settings())
        report = reports[0]
        self.assertEqual(report["b_center"], 1.2)
        self.assertEqual((report["b_min"], report["b_max"]), (0.6, 1.8))
        self.assertEqual([p["bk"] for p in report["proposals"]], [2.5, 2.75, 3, 3.25, 3.5])
        self.assertEqual(report["n"], 1)

    def test_plateaus_and_cross_bk_ties_are_order_independent(self):
        points = [point(0.6, 3, 100), point(1.2, 3, 100), point(1.2, 3.5, 100)]
        key = GridKey(10, 6000, 2)
        a = initial_grid({key: points}, Settings())
        b = initial_grid({key: points[::-1]}, Settings())
        self.assertEqual(a, b)
        self.assertEqual(a[0]["source_best"], point(1.2, 3, 100))

    def test_new_centers_are_clamped_and_snapped_like_notebook(self):
        grids = {
            GridKey(10, 6000, 1): [point(-0.4, 3.1, 100)],
            GridKey(10, 6000, 2): [point(1.31, 3.4, 100)],
        }
        reports = initial_grid(grids, Settings(bk_size=3))
        self.assertEqual(reports[0]["b_center"], 0)
        self.assertEqual(reports[0]["bk_center"], 3)
        self.assertEqual(reports[1]["b_center"], 1.35)
        self.assertEqual(reports[1]["bk_center"], 3.5)
        self.assertEqual(len(reports[1]["proposals"]), 3)

    def test_all_zero_group_warns_and_skips(self):
        with self.assertWarnsRegex(UserWarning, "all capacities are zero"):
            reports = initial_grid({GridKey(10, 6000, 1): [point(1, 3, 0)]}, Settings())
        self.assertEqual(reports[0]["status"], "skipped_all_zero")
        self.assertEqual(reports[0]["proposals"], [])

    def test_geometry_groups_remain_separate(self):
        grids = {
            GridKey(10, 100, 1): [point(1.2, 3, 100)],
            GridKey(10, 200, 1): [point(1.5, 4, 200)],
        }
        reports = initial_grid(grids, Settings(bk_size=1))
        self.assertEqual([(r["ns"], r["b_center"]) for r in reports], [(100, 1.2), (200, 1.5)])


class SourceAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.summary = self.root / "max_points_df.csv"

    def test_legacy_summary_requires_explicit_total(self):
        self.summary.write_text("nd,ndR,b,bk,capacity,sem\n10,1,1.2,3,100,2\n")
        with self.assertRaisesRegex(ValueError, "total-synapses"):
            load_summary(self.summary)
        grids = load_summary(self.summary, 60000)
        self.assertEqual(list(grids), [GridKey(10, 6000, 1)])
        with self.assertRaisesRegex(ValueError, "divisible"):
            load_summary(self.summary, 60001)

    def test_summary_ns_and_n_columns(self):
        self.summary.write_text("nd,ns,n,b,bk,capacity\n10,100,2,1.2,3,100\n")
        self.assertIn(GridKey(10, 100, 2), load_summary(self.summary))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_summary(self.summary, 60000)

    def test_invalid_summary_values_and_schema(self):
        for contents in [
            "nd,ns,n,b,bk,capacity\n10,100,2,1.2,3,nan\n",
            "nd,ns,n,b,bk,capacity\n10,100,2,1.2,3,-1\n",
            "nd,ns,n,b,bk,capacity\n10,100,11,1.2,3,100\n",
            "nd,ns,n,ndR,b,bk,capacity\n10,100,2,1,1.2,3,100\n",
            "nd,ns,n,b,bk,capacity\n",
            "nd,ns,n,b,bk\n10,100,2,1.2,3\n",
        ]:
            self.summary.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                load_summary(self.summary)

    def test_raw_source_uses_mean_capacity(self):
        folder = self.root / "nd10_ns100"
        folder.mkdir()
        (folder / "n1.csv").write_text(
            "1.2,3,5,0,0,0,0,0,600\n"
            "1.5,3.25,5,101,101,101,101,101,101\n"
        )
        grids = load_source(self.root, Settings())
        report = initial_grid(grids, Settings())[0]
        self.assertEqual(report["source_best"], point(1.5, 3.25, 101))

    def test_cli_dry_run_and_mathematica_output(self):
        self.summary.write_text("nd,ns,ndR,b,bk,capacity\n10,100,1,1.2,3,100\n")
        output = self.root / "new_experiment" / "tmp.csv"
        args = [str(self.summary), "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args + ["--dry-run"]), 0)
        self.assertFalse(output.parent.exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        with output.open() as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0], ["1.2", "2.5", "1", "10", "100"])
        self.assertEqual(rows[-1], ["1.2", "3.5", "1", "10", "100"])
        report = json.loads(output.with_suffix(".report.json").read_text())
        self.assertEqual(report["grids"][0]["source_best"]["capacity"], 100)

    def test_empty_initialization_clears_old_candidates(self):
        self.summary.write_text("nd,ns,n,b,bk,capacity\n10,100,1,1.2,3,0\n")
        output = self.root / "tmp.csv"
        output.write_text("stale candidates")
        with self.assertWarns(UserWarning), redirect_stdout(io.StringIO()):
            main([str(self.summary), "--output", str(output)])
        self.assertEqual(output.read_text(), "")

    def test_source_cannot_be_overwritten(self):
        original = "nd,ns,n,b,bk,capacity\n10,100,1,1.2,3,100\n"
        self.summary.write_text(original)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([str(self.summary), "--output", str(self.summary)])
        self.assertEqual(self.summary.read_text(), original)


if __name__ == "__main__":
    unittest.main()
