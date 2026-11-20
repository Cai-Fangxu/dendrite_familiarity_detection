"""Tests for standalone legacy grid-search maximum-point export."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common_data_processing import export_grid_search_max_points as export
from common_data_processing.initialize_grid_search import load_summary
from common_data_processing.plan_grid_search import Grid, Point, Settings, GridKey


class MaximumRowsTests(unittest.TestCase):
    def test_ties_match_planner(self):
        key = GridKey(100, 600, 2)
        points = [Point(b, bk, 6.344382, capacity, 0, 6) for b, bk, capacity in
                  [(0.3, 1, 10), (0.6, 1, 10), (0.45, 1.25, 10), (0.75, 1.5, 9)]]
        rows = export.maximum_rows({key: points})
        ranked = Grid(points, Settings()).ranked_points()
        expected = [(p.b, p.bk) for p in ranked if p.capacity == ranked[0].capacity]
        self.assertEqual([(r["b"], r["bk"]) for r in rows], expected)
        self.assertEqual(export.maximum_rows({key: points[::-1]}), rows)

    def test_irregular_grid_zero_capacity_and_separate_geometries(self):
        a, b = GridKey(8, 6, 1), GridKey(8, 12, 1)
        rows = export.maximum_rows({
            b: [Point(1.113, 0.333, 6.344382, 8, 0, 6)],
            a: [Point(1.131, 0.111, 6.344382, 0, 0, 6),
                Point(2.719, 0.111, 6.344382, 0, 0, 6),
                Point(0.431, 0.517, 6.344382, 0, 0, 6)],
        })
        self.assertEqual([(r["ns"], r["b"], r["bk"]) for r in rows],
                         [(6, 2.719, 0.111), (6, 0.431, 0.517), (12, 1.113, 0.333)])


class ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        geometry = self.data / "nd8_ns6"
        geometry.mkdir(parents=True)
        self.source = geometry / "n2.csv"
        self.output = self.root / "summaries" / "max_points_df.csv"

    def test_pooling_mean_sem_and_initializer_compatibility(self):
        self.source.write_text("1.131,2.517,6.344382,0,4\n1.131,2.517,6.344382,8,12\n2.718,3.119,6.344382,0,11\n")
        original = self.source.read_bytes()
        (self.source.parent / "old_n2.csv").write_text("ignored")
        (self.data / "next_grid").mkdir()
        (self.data / "next_grid" / "max_points_df.csv").write_text("ignored")
        with self.assertWarnsRegex(UserWarning, "pooled repeated"), redirect_stdout(io.StringIO()):
            self.assertEqual(export.main([str(self.data), "--output", str(self.output),
                                          "--n-trials", "2"]), 0)
        with self.output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, export.FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["capacity"]), 6)
        self.assertEqual(int(rows[0]["n_samples"]), 4)
        self.assertAlmostEqual(float(rows[0]["sem"]), np.std([0, 4, 8, 12], ddof=1) / 2)
        self.assertEqual(float(rows[0]["bk"]), 2.517)
        self.assertEqual(float(rows[0]["beta"]), 6.344382)
        self.assertEqual(next(iter(load_summary(self.output))).n, 2)
        self.assertEqual(self.source.read_bytes(), original)

    def test_default_trial_count_and_output_within_data(self):
        self.source.write_text("1,2,6.344382,1,2,3,4,5,6\n")
        output = self.data / "next_grid" / "max_points_df.csv"
        rows = export.export_max_points(self.data, output)
        self.assertEqual(rows[0]["capacity"], 3.5)
        self.assertEqual(rows[0]["n_samples"], 6)
        self.assertTrue(output.exists())

    def test_invalid_data_does_not_overwrite_output(self):
        self.output.parent.mkdir()
        self.output.write_text("existing summary")
        for raw in ("", "1,2,6.344382,3,4\n1,2,5,3,4\n", "1,2,6.344382,nan,4\n",
                    "1,2,6.344382,-3,4\n", "1,2,6.344382,3\n"):
            self.source.write_text(raw)
            with self.subTest(raw=raw), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                export.main([str(self.data), "--output", str(self.output), "--n-trials", "2"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(self.output.read_text(), "existing summary")

    def test_protect_raw_paths_and_missing_input(self):
        self.source.write_text("1,2,6.344382,3,4\n")
        for output in (self.source, self.source.parent / "n3.csv"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "raw result"):
                export.export_max_points(self.data, output, 2)
        for source in (self.root / "missing", self.source, self.root / "empty"):
            (self.root / "empty").mkdir(exist_ok=True)
            with self.subTest(source=source), self.assertRaises(ValueError):
                export.export_max_points(source, self.output, 2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            export.export_max_points(self.data, self.output, 0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
