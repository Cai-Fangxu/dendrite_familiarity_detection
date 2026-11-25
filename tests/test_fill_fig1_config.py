"""Capacity selection, legacy parameter translation, and config preservation."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from theory_check.fill_fig1_config import fill_config, main


class FillConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"condition": "independent", "nd": 100, "n": 10,
                       "stages": [10, 100], "sampling": {"forks": 1000},
                       "rules": [{"name": "nwta", "enabled": True, "b": 99, "kappa": 99}]}

    def summary(self, rows, folder="search"):
        path = self.root / folder / "max_points_df.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def fill(self, config=None, sources=None, **kwargs):
        if sources is None:
            sources = {"independent": {name: self.root / "search" for name in
                                      ("nwta", "nwta_decay", "softmax_int")}}
        return fill_config(self.config if config is None else config, sources, **kwargs)

    def test_highest_capacity_across_nested_files_and_exact_selectors(self):
        self.summary([
            dict(nd=100, ndR=10, ns=600, b=1, bk=3, beta=7, capacity=200),
            dict(nd=100, ndR=5, ns=600, b=9, bk=10, beta=7, capacity=999),
            dict(nd=75, ndR=10, ns=800, b=9, bk=10, beta=7, capacity=999),
            dict(nd=100, ndR=10, ns=800, b=9, bk=10, beta=7, capacity=999),
            dict(nd=100, ndR=10, ns=600, b=9, bk=10, beta=7, capacity="nan"),
        ])
        self.summary([dict(nd=100, ndR=10, ns=600, b=2, bk=5, beta=8, capacity=300)], "search/next_grid")
        before = copy.deepcopy(self.config)
        filled, messages = self.fill()
        self.assertEqual(filled["rules"][0], dict(name="nwta", enabled=True, b=2, kappa=3))
        self.assertEqual(self.config, before)
        self.assertEqual(filled["stages"], before["stages"])
        self.assertEqual(filled["sampling"], before["sampling"])
        self.assertIn("next_grid/max_points_df.csv:2", messages[0])

    def test_per_rule_dimensions_condition_and_decay_beta(self):
        path = self.summary([dict(nd=75, ndR=5, b=1, bk=3, beta=-0.5, capacity=123)], "nd75/n5")
        self.config["condition"] = "shared"
        self.config["rules"] = [dict(name="nwta_decay", nd=75, n=5)]
        filled, _ = self.fill(sources={"shared": {"nwta_decay": str(self.root / "nd{nd}/n{n}")}})
        self.assertTrue(path.exists())
        self.assertEqual(filled["rules"][0], dict(name="nwta_decay", nd=75, n=5, enabled=True,
                                                 b=1, kappa=2, beta_decay=-0.5))

    def test_missing_sources_and_unmatched_dimensions_clear_stale_parameters(self):
        self.summary([dict(nd=100, ndR=5, b=1, bk=3, beta=7, capacity=200)])
        for source in (None, self.root / "absent", self.root / "search"):
            filled, _ = self.fill(sources={"independent": {"nwta": source}})
            self.assertEqual(filled["rules"][0], dict(name="nwta", enabled=False, b=None, kappa=None))

    def test_disabled_rules_are_preserved_without_validation_or_lookup(self):
        rule = dict(name="custom", enabled=False, nd=None, n="unused", b=42, extra=[1, 2])
        self.config["rules"] = [rule]
        filled, _ = self.fill()
        self.assertEqual(filled, self.config)
        self.assertIsNot(filled["rules"][0], rule)

    def test_incomplete_best_row_disables_without_using_inferior_row(self):
        self.summary([dict(nd=100, ndR=10, b=1, bk=3, beta=7, capacity=200),
                      dict(nd=100, ndR=10, b=2, bk="", beta=8, capacity=300)])
        filled, _ = self.fill()
        self.assertEqual(filled["rules"][0], dict(name="nwta", enabled=False, b=2, kappa=None))

    def test_softmax_selection_uses_calibrated_n_not_largest_overall_capacity(self):
        self.summary([dict(nd=100, ns=600, beta_select=1.8, eta=25, capacity=200),
                      dict(nd=100, ns=600, beta_select=2.4, eta=30, capacity=900)])
        registry = self.root / "calibrations.json"
        registry.write_text(json.dumps({"calibrations": [
            dict(nd=100, beta_select=1.8, target_n=10),
            dict(nd=100, beta_select=2.4, target_n=5)]}))
        self.config["rules"] = [dict(name="softmax_int", eta=99, beta_select=99)]
        filled, _ = self.fill(calibration_file=registry)
        self.assertEqual(filled["rules"][0], dict(name="softmax_int", enabled=True, eta=25, beta_select=1.8))
        filled, _ = self.fill(calibration_file=self.root / "missing.json")
        self.assertEqual(filled["rules"][0], dict(name="softmax_int", enabled=False, eta=None, beta_select=None))

    def test_softmax_explicit_target_n_needs_no_registry_and_ties_are_stable(self):
        self.summary([dict(nd=100, target_n=10, beta_select=1.8, eta=25, capacity=200),
                      dict(nd=100, target_n=10, beta_select=1.8, eta=30, capacity=200)])
        self.config["rules"] = [dict(name="softmax_int")]
        filled, _ = self.fill(calibration_file=self.root / "missing.json")
        self.assertEqual(filled["rules"][0]["eta"], 25)

    def test_cli_default_output_dry_run_and_in_place(self):
        source = self.summary([dict(nd=100, ndR=10, b=1, bk=3, beta=7, capacity=200)])
        config_path = self.root / "experiment.json"
        original = json.dumps(self.config)
        config_path.write_text(original)
        output = self.root / "experiment.filled.json"
        with patch("theory_check.fill_fig1_config.SOURCE_DIRS", {"independent": {"nwta": source}}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(config_path), "--dry-run"]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(main(["--config", str(config_path)]), 0)
                self.assertEqual(config_path.read_text(), original)
                self.assertTrue(json.loads(output.read_text())["rules"][0]["enabled"])
                self.assertEqual(main(["--config", str(config_path), "--in-place"]), 0)
        self.assertEqual(json.loads(config_path.read_text()), json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
