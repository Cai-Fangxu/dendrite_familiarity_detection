"""Tests for Softmax initialization with reference-eta spacing and saved calibrations."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common_data_processing import initialize_softmax_grid_search as init
from common_data_processing import softmax_calibration_registry as registry
from common_data_processing.plan_grid_search import Settings, SoftmaxGridKey


def point(b, eta, capacity):
    return dict(b=b, eta=eta, capacity=capacity)


class InitialGridTests(unittest.TestCase):
    def test_scaled_spacing_fractional_target_and_original_beta(self):
        key = SoftmaxGridKey(100, 600, 4.123456789012345)
        result = init.initial_grid({key: [point(1.2, 10, 100)]}, Settings(), {key: 2.5})[0]
        self.assertEqual(result["beta_select"], key.beta_select)
        self.assertEqual(result["n_eff"], 2.5)
        self.assertEqual(result["b_center"], 1.2)
        self.assertEqual((result["b_min"], result["b_max"]), (0.6, 1.8))
        self.assertEqual([p["eta_reference"] for p in result["proposals"]], [3.5, 3.75, 4, 4.25, 4.5])
        self.assertEqual([p["eta"] for p in result["proposals"]], [8.75, 9.375, 10, 10.625, 11.25])

    def test_ties_and_geometry_groups(self):
        left, right = SoftmaxGridKey(8, 10, 0), SoftmaxGridKey(8, 20, 0)
        points = [point(0.6, 4, 10), point(1.2, 4, 10), point(1.2, 6, 10)]
        a = init.initial_grid({left: points, right: [point(1.5, 8, 20)]},
                              Settings(bk_size=1), {left: 8, right: 8})
        b = init.initial_grid({right: [point(1.5, 8, 20)], left: points[::-1]},
                              Settings(bk_size=1), {left: 8, right: 8})
        self.assertEqual(a, b)
        self.assertEqual(a[0]["source_best"], point(1.2, 4, 10))
        self.assertEqual([r["ns"] for r in a], [10, 20])

    def test_snapping_and_positive_eta_boundary(self):
        key = SoftmaxGridKey(4, 6, 1.2)
        report = init.initial_grid({key: [point(-0.3, 0.01, 10)]}, Settings(), {key: 2})[0]
        self.assertEqual(report["b_center"], 0)
        self.assertEqual(report["eta_reference_center"], 0.25)
        self.assertEqual(report["omitted_nonpositive_eta"], 2)
        self.assertEqual([p["eta"] for p in report["proposals"]], [0.5, 1, 1.5])
        report = init.initial_grid({key: [point(1.31, 1.17, 10)]},
                                   Settings(bk_size=1), {key: 2})[0]
        self.assertEqual(report["b_center"], 1.35)
        self.assertEqual(report["eta_center"], 1)

    def test_all_zero_skips_and_keeps_n_one_supported(self):
        key = SoftmaxGridKey(4, 6, 100)
        with self.assertWarnsRegex(UserWarning, "all capacities are zero"):
            report = init.initial_grid({key: [point(1, 3, 0)]}, Settings(), {key: 1})[0]
        self.assertEqual(report["status"], "skipped_all_zero")
        self.assertEqual(report["proposals"], [])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.summary = self.root / "max_points_df.csv"
        self.beta = 1.2345678901234567
        self.summary.write_text(f"nd,ns,beta_select,b,eta,beta,capacity\n8,6,{self.beta},1.2,5,-1,100\n")
        registry.update_registry(self.root / registry.REGISTRY_NAME,
                                 [dict(nd=8, beta_select=self.beta, target_n=2.5)])
        self.output = self.root / "new" / "tmp.csv"
        self.params = self.root / "new" / "params_list_softmax.csv"

    def test_cli_dry_run_and_both_output_schemas(self):
        original = self.summary.read_bytes()
        args = [str(self.summary), "--output", str(self.output),
                "--params-output", str(self.params), "--seed", "123"]
        with redirect_stdout(io.StringIO()):
            init.main(args + ["--dry-run"])
        self.assertFalse(self.output.parent.exists())
        with redirect_stdout(io.StringIO()):
            init.main(args)
        reference = np.loadtxt(self.output, delimiter=",", ndmin=2)
        params = np.loadtxt(self.params, delimiter=",", ndmin=2)
        self.assertEqual(reference.shape, (5, 5))
        self.assertEqual(params.shape, (5, 7))
        np.testing.assert_array_equal(reference[:, 1], [1.5, 1.75, 2, 2.25, 2.5])
        np.testing.assert_array_equal(reference[:, 2], 2.5)
        np.testing.assert_array_equal(params[:, 2], reference[:, 1] * reference[:, 2])
        np.testing.assert_array_equal(params[:, 4], self.beta)
        np.testing.assert_array_equal(params[:, 5], -1)
        np.testing.assert_array_equal(params[:, 6], 123)
        audit = json.loads(self.output.with_suffix(".report.json").read_text())
        self.assertEqual(audit["settings"]["eta_reference_spacing"], 0.25)
        self.assertEqual(audit["grids"][0]["source_best"]["eta"], 5)
        self.assertEqual(self.summary.read_bytes(), original)

    def test_raw_source_uses_mean_capacity(self):
        geometry = self.root / "results" / "nd8_ns6"
        geometry.mkdir(parents=True)
        (geometry / f"beta_select{self.beta}.csv").write_text(
            "1.2,5,-1,0,0,0,0,0,600\n1.5,6,-1,101,101,101,101,101,101\n"
        )
        grids = init.load_source(geometry.parent, Settings())
        key = next(iter(grids))
        report = init.initial_grid(grids, Settings(), {key: 2.5})[0]
        self.assertEqual(report["source_best"], point(1.5, 6, 101))
        with redirect_stdout(io.StringIO()):
            init.main([str(geometry.parent), "--dry-run"])

    def test_missing_ns_and_total_validation(self):
        self.summary.write_text(f"nd,beta_select,b,eta,capacity\n8,{self.beta},1.2,5,10\n")
        with self.assertRaisesRegex(ValueError, "total-synapses"):
            init.load_summary(self.summary)
        self.assertEqual(next(iter(init.load_summary(self.summary, 48))).ns, 6)
        with self.assertRaisesRegex(ValueError, "divisible"):
            init.load_summary(self.summary, 49)

    def test_invalid_summaries(self):
        for row in (f"8,6,{self.beta},1.2,0,-1,10", f"8,6,{self.beta},1.2,5,0,10",
                    f"8,6,-1,1.2,5,-1,10", f"8,6,{self.beta},nan,5,-1,10",
                    f"8,6,{self.beta},1.2,5,-1,-1", f"8,6.5,{self.beta},1.2,5,-1,10"):
            self.summary.write_text("nd,ns,beta_select,b,eta,beta,capacity\n" + row + "\n")
            with self.subTest(row=row), self.assertRaises(ValueError):
                init.load_summary(self.summary)

    def test_missing_mapping_and_source_protection(self):
        self.summary.write_text("nd,ns,beta_select,b,eta,capacity\n8,6,9,1.2,5,10\n")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            init.main([str(self.summary), "--output", str(self.output)])
        self.assertFalse(self.output.exists())
        for output, params in ((self.summary, None), (self.output, self.summary),
                               (self.output, self.output)):
            with self.subTest(output=output, params=params), self.assertRaises(ValueError):
                init.validate_output_paths(self.summary, output, params, [])
        with self.assertRaises(ValueError):
            init.validate_output_paths(self.summary, self.output, None,
                                       [self.output.with_suffix(".report.json")])

    def test_empty_plan_clears_both_stale_tables(self):
        self.summary.write_text(f"nd,ns,beta_select,b,eta,capacity\n8,6,{self.beta},1.2,5,0\n")
        self.output.parent.mkdir()
        self.output.write_text("stale")
        self.params.write_text("stale")
        with self.assertWarns(UserWarning), redirect_stdout(io.StringIO()):
            init.main([str(self.summary), "--output", str(self.output), "--params-output", str(self.params)])
        self.assertEqual(self.output.read_text(), "")
        self.assertEqual(self.params.read_text(), "")


if __name__ == "__main__":
    unittest.main()
