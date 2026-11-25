"""Numerical checks for spherical updates and frozen-state coefficient estimates."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from theory_check.run_fig1 import resolve_config, run_experiment, point_rows
from theory_check.spherical import Rule, amplitudes, fork_batch, indexed_key, overlaps, pattern, rotate, train_segment
from theory_check.statistics import Moments, batch_se, choose_pairs, gram_factor, theory_moments
from plot_scripts.theory_fig1.plot_coefficients import load_runs


class TheoryFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_signed_variable_targets_and_fixed_target_gates(self):
        rule = Rule("variable_target", 4, 8, n=2, b=0.5, kappa=0.5)
        u = np.array([[3., 2., -1., -2.], [-1., -2., -3., -4.]])
        expected = np.array([[-2., -1., 0., 0.], [2.5, 0., 0., 0.]])
        np.testing.assert_allclose(amplitudes(u, rule), expected)
        np.testing.assert_allclose(amplitudes(jnp.asarray(u), rule, jnp), expected)
        for name in ("nwta", "threshold"):
            np.testing.assert_array_equal(amplitudes(u[:1], replace(rule, name=name)), np.zeros((1, 4)))
        self.assertAlmostEqual(Rule("threshold", 100, 600, n=10).eligibility, 1.2815515655446004)

    def test_all_amplitudes_agree_across_backends(self):
        u = np.random.default_rng(1).normal(size=(8, 4))
        for name in ("threshold", "nwta", "variable_target", "softmax_int", "softmax_nonint"):
            rule = Rule(name, 4, 8, n=2, b=0.5, kappa=0.2, eta=0.7, beta_select=0.9)
            np.testing.assert_allclose(amplitudes(u, rule), amplitudes(jnp.asarray(u), rule, jnp), atol=1e-14)

    def test_nwta_compiled_selection_handles_ties_and_all_winners(self):
        u = np.array([[1., 2., 2., -1.], [1., 1., 1., 1.], [-2., -1., 0., 2.]])
        for dtype in (jnp.float32, jnp.float64):
            for n in (1, 2, 4):
                rule = Rule("nwta", 4, 8, n=n, b=1, kappa=2)
                compiled = jax.jit(lambda values: amplitudes(values, rule, jnp))
                actual = np.asarray(compiled(jnp.asarray(u, dtype=dtype)))
                np.testing.assert_array_equal(actual, amplitudes(u, rule))
                self.assertTrue(np.all(np.count_nonzero(actual, axis=-1) == n))

    def test_nwta_x64_compiled_training_and_forks_match_reference(self):
        rule = Rule("nwta", 100, 600, n=10, b=1.85, kappa=1.9)
        rng = np.random.default_rng(42)
        w = rng.normal(size=(rule.nd, rule.ns))
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        key = jax.random.PRNGKey(43)
        for condition in ("shared", "independent"):
            actual = train_segment(jnp.asarray(w), key, 0, rule=rule, condition=condition, steps=3)
            expected = w.copy()
            for step in range(3):
                x = np.asarray(pattern(indexed_key(key, step), rule, condition, jnp.float64))
                u = np.sum(expected * x, axis=-1)
                theta = amplitudes(u, rule) / np.sqrt(rule.ns)
                tangent = x - u[:, None] * expected
                tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
                expected = expected * np.cos(theta[:, None]) + tangent * np.sin(theta[:, None])
                expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
            np.testing.assert_allclose(actual, expected, atol=1e-13)
            old = rng.normal(size=(2, rule.ns) if condition == "shared" else (2, rule.nd, rule.ns))
            delta = fork_batch(jnp.asarray(w), jnp.asarray(old), key, jnp.arange(2, dtype=jnp.uint32),
                               rule=rule, condition=condition, steps=3)
            for index in range(2):
                end = train_segment(jnp.asarray(w), indexed_key(key, index), 0,
                                    rule=rule, condition=condition, steps=3)
                np.testing.assert_allclose(delta[index], overlaps(np.asarray(end)-w, old, condition), atol=1e-13)

    def test_rotation_signed_direction_unit_norm_and_parallel_input(self):
        rule = Rule("variable_target", 2, 3, n=2, b=0, kappa=0)
        w = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        x = jnp.array([[1., 2., 0.], [0., -1., 2.]])
        result = np.asarray(rotate(w, x, rule))
        angles = np.array([-1., 1.]) / np.sqrt(3)
        expected = np.array([[np.cos(angles[0]), np.sin(angles[0]), 0],
                             [0, np.cos(angles[1]), np.sin(angles[1])]])
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-14)
        np.testing.assert_array_equal(rotate(w, 3*w, rule), w)
        np.testing.assert_array_equal(rotate(w, jnp.zeros_like(w), rule), w)

    def test_forks_reset_and_chunking_preserves_trajectories(self):
        rule = Rule("softmax_int", 3, 5, n=1, eta=0.8, beta_select=0.5)
        rng = np.random.default_rng(3)
        w = rng.normal(size=(3, 5))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        key = jax.random.PRNGKey(5)
        for condition in ("shared", "independent"):
            old = rng.normal(size=(2, 5) if condition == "shared" else (2, 3, 5))
            initial = w.copy()
            actual = np.asarray(fork_batch(jnp.asarray(w), jnp.asarray(old), key, jnp.arange(4),
                                          rule=rule, condition=condition, steps=3))
            expected = []
            for index in range(4):
                end = jnp.asarray(w)
                for step in range(3):
                    x = pattern(indexed_key(indexed_key(key, index), step), rule, condition, end.dtype)
                    end = rotate(end, x, rule)
                expected.append(overlaps(np.asarray(end)-w, old, condition))
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            chunks = [fork_batch(jnp.asarray(w), jnp.asarray(old), key, jnp.arange(i, i+2),
                                 rule=rule, condition=condition, steps=3) for i in (0, 2)]
            # XLA may reorder the final contraction for a different batch shape.
            np.testing.assert_allclose(actual, np.concatenate(chunks), atol=1e-14)
            np.testing.assert_array_equal(initial, w)
            one = train_segment(jnp.asarray(w), key, 0, rule=rule, condition=condition, steps=5)
            split = train_segment(jnp.asarray(w), key, 0, rule=rule, condition=condition, steps=2)
            split = train_segment(split, key, 2, rule=rule, condition=condition, steps=3)
            np.testing.assert_allclose(one, split, atol=1e-14)

    def test_streaming_covariance_matches_centered_numpy(self):
        values = np.random.default_rng(9).normal(size=(103, 2, 4)) + 1e4
        moments = Moments()
        for batch in np.array_split(values, 7):
            moments.add(batch)
        np.testing.assert_allclose(moments.mean, values.mean(axis=0), atol=1e-10)
        for p in range(2):
            np.testing.assert_allclose(moments.covariance[p], np.cov(values[:, p], rowvar=False), atol=1e-10)

    def test_rank_deficient_shared_theory_and_constant_amplitude(self):
        q = np.ones((3, 3))
        factor, inverse, rank = gram_factor(q)
        self.assertEqual(rank, 1)
        np.testing.assert_allclose(factor @ factor.T, q, atol=1e-14)
        np.testing.assert_allclose(inverse, np.linalg.pinv(q), atol=1e-14)
        rule = Rule("softmax_int", 3, 5, n=1, eta=0.9, beta_select=0)
        m = np.array([[0.2, 0.2, 0.2]])
        result = theory_moments(rule, "shared", q, m, np.random.default_rng(5), 1000, 5, 100)
        np.testing.assert_allclose(result["C"], 0.09, atol=1e-14)
        # Identical dendrites cancel the F contribution even for finite MC samples.
        np.testing.assert_allclose(result["A"], -0.045*m, atol=1e-14)
        independent = theory_moments(rule, "independent", np.eye(3), m,
                                     np.random.default_rng(5), 1000, 5, 100)
        np.testing.assert_allclose(independent["C"], np.eye(3)*0.09, atol=1e-14)
        np.testing.assert_allclose(independent["A"], -0.045*m, atol=1e-14)

    def test_multistep_drift_matches_known_spherical_conditional_mean(self):
        # Constant angular amplitude has an exact mean m*cos(theta)^steps.
        rule = Rule("softmax_int", 2, 8, n=1, eta=1.2, beta_select=0)
        w = np.eye(2, 8)
        old = w[None].copy()
        steps = 4
        delta = np.asarray(fork_batch(jnp.asarray(w), jnp.asarray(old), jax.random.PRNGKey(73),
                                     jnp.arange(6000), rule=rule, condition="independent", steps=steps))
        expected = np.cos((rule.eta/rule.nd)/np.sqrt(rule.ns))**steps - 1
        se = delta.std(axis=0, ddof=1)/np.sqrt(len(delta))
        self.assertTrue(np.all(np.abs(delta.mean(axis=0)-expected) < 5*se))

    def test_rescaling_and_shared_age_uncertainty(self):
        rule = Rule("softmax_int", 2, 8, n=1, eta=1, beta_select=0)
        batches = np.random.default_rng(8).normal(size=(4, 20, 1, 2))
        batches = np.repeat(batches, 2, axis=2)  # Perfectly dependent ages.
        total = Moments()
        groups = []
        for batch in batches:
            group = Moments()
            group.add(batch)
            total.merge(group.count, group.mean, group.m2)
            groups.append(group)
        empirical = (total, np.array([g.mean for g in groups]),
                     np.array([g.covariance for g in groups]), np.full(4, 20))
        theory = theory_moments(rule, "independent", np.eye(2), np.ones((2, 2)),
                               np.random.default_rng(1), 80, 4, 20)
        rows = point_rows(rule, "independent", 0, 5, [0, 1], np.ones((2, 2)),
                          theory, empirical, np.array([[0, 1]]), 4)
        diagonal = [r for r in rows if r["kind"] == "diagonal" and r["r"] == 0]
        self.assertAlmostEqual(diagonal[0]["observed"], 2*total.covariance[0, 0, 0])
        self.assertAlmostEqual(diagonal[-1]["empirical_se"], diagonal[0]["empirical_se"])
        drift = next(r for r in rows if r["kind"] == "drift")
        self.assertAlmostEqual(drift["observed"], 2*total.mean[0, 0])

    def test_config_requires_parameters_and_valid_counts(self):
        with self.assertRaisesRegex(ValueError, "threshold.b"):
            resolve_config({"condition": "independent", "rules": [{"name": "threshold"}]})
        config, rules = resolve_config({"condition": "independent", "rules": [{"name": "threshold", "b": 1, "kappa": 0.5}]})
        self.assertEqual((rules[0].nd, rules[0].ns, rules[0].n), (100, 600, 10))
        self.assertEqual(config["sampling"]["fork_steps"], 10)
        with self.assertRaises(ValueError):
            resolve_config({"condition": "independent", "rules": [{"name": "nwta", "n": 1.5, "b": 1, "kappa": 0.5}]})
        with self.assertRaises(ValueError):
            resolve_config({"condition": "independent", "rules": [{"name": "threshold", "b": 1, "kappa": 0.5}], "sampling": {"fork_steps": 0}})

    def test_config_selects_exactly_one_condition_and_configurable_seeds(self):
        raw = {"rules": [{"name": "threshold", "b": 1, "kappa": 0.5}]}
        for value in (None, ["independent"], ["independent", "shared"], "partial"):
            with self.subTest(condition=value), self.assertRaisesRegex(ValueError, "exactly one condition"):
                resolve_config(dict(raw, condition=value))
        with self.assertRaisesRegex(ValueError, "Replace conditions"):
            resolve_config(dict(raw, conditions=["independent", "shared"]))
        with self.assertRaisesRegex(ValueError, "stages must be a nonempty list"):
            resolve_config(dict(raw, condition="shared", stages={"shared": [6000]}))
        independent, _ = resolve_config(dict(raw, condition="independent"))
        self.assertEqual(independent["condition"], "independent")
        self.assertEqual(independent["stages"], [6000])
        self.assertEqual(independent["seeds"], [0, 1, 2])
        shared, _ = resolve_config(dict(raw, condition="shared", seeds=[0]))
        self.assertEqual(shared["condition"], "shared")
        self.assertEqual(shared["stages"], [600, 3000, 12000])
        self.assertEqual(shared["seeds"], [0])

    def test_pair_selection_includes_signed_extrema(self):
        c = np.array([[1, -0.5, 0.1], [-0.5, 1, 0.8], [0.1, 0.8, 1]])
        np.testing.assert_array_equal(choose_pairs(c, 2), [[0, 1], [1, 2]])
        self.assertEqual(choose_pairs(np.eye(1), 30).shape, (0, 2))

    def test_small_end_to_end_preserves_state_and_streams(self):
        raw = {"nd": 100, "n": 10, "rules": [{"name": "softmax_int", "eta": 0.8, "beta_select": 0.2}],
               "seeds": [1], "ages": [1, 2, 3, 10], "stages": [1, 2, 3],
               "sampling": {"forks": 8, "fork_steps": 2, "fork_batch_size": 4,
                            "uncertainty_groups": 2, "theory_samples": 16, "theory_batch_size": 8, "max_pairs": 2}}
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for condition, eta in (("independent", 0.8), ("shared", 1.2)):
                config, rules = resolve_config(dict(raw, condition=condition,
                    rules=[{"name": "softmax_int", "eta": eta, "beta_select": 0.2}]))
                output = Path(tmp) / condition
                output.mkdir()
                run_experiment(config, rules, output)
                outputs.append(output)
                condition_points, condition_snapshots = load_runs([output])
                self.assertEqual({p["condition"] for p in condition_points}, {condition})
                self.assertEqual(len(condition_snapshots), 2)
                self.assertEqual({p["eta"] for p in condition_points}, {eta})
                self.assertFalse(any(output.glob("*_t1")))
                for t in (2, 3):
                    actual_ages = {p["age"] for p in condition_points
                                   if p["t"] == t and not p["age_averaged"]}
                    self.assertEqual(actual_ages, {age for age in raw["ages"] if age < t})
            # Different condition-specific parameters can be plotted together.
            points, snapshots = load_runs(outputs)
            self.assertEqual(len(snapshots), 4)
            self.assertTrue(points)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_runs([outputs[0], outputs[0]])
            for directory in (d for output in outputs for d in output.iterdir()):
                with np.load(directory / "snapshot.npz") as data:
                    condition = "independent" if "independent" in directory.name else "shared"
                    np.testing.assert_allclose(np.linalg.norm(data["weights"], axis=-1), 1, atol=1e-14)
                    np.testing.assert_allclose(overlaps(data["weights"], data["old_patterns"], condition), data["overlaps"])
                    t = int(directory.name.split("_t")[-1])
                    np.testing.assert_array_equal(data["ages"], [age for age in raw["ages"] if age < t])
                    np.testing.assert_array_equal(data["acquisition_times"], t - data["ages"])
                self.assertTrue((directory / "complete.json").exists())

    def test_all_unavailable_ages_skip_computation_and_snapshots(self):
        config, rules = resolve_config({"condition": "shared", "stages": [1, 2],
            "ages": [2, 5], "seeds": [0],
            "rules": [{"name": "softmax_int", "eta": 0.8, "beta_select": 0.2}]})
        with tempfile.TemporaryDirectory() as tmp, \
                patch("theory_check.run_fig1.train_segment") as train, \
                patch("theory_check.run_fig1.theory_moments") as theory, \
                patch("theory_check.run_fig1.measure_forks") as forks:
            run_experiment(config, rules, Path(tmp))
            train.assert_not_called()
            theory.assert_not_called()
            forks.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
