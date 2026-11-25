"""Checks for legacy decay dynamics and their distinct Gaussian predictions."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from lib.neurons import (HyperDynConfig1, State1, Neuron_Decay_NonInt,
                         Neuron_Decay_Int, Neuron_Decay_Int_Variable_n_min1)
from theory_check.spherical import (Rule, initialize_weights, legacy_neuron,
                                    target_terms, update_weights)
from theory_check.statistics import theory_moments
from theory_check.run_fig1 import resolve_config, run_experiment
from plot_scripts.theory_fig1.plot_coefficients import load_runs


class DecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_exact_library_update_target_hitting_and_nonunit_norms(self):
        classes = {"threshold_decay": Neuron_Decay_NonInt, "nwta_decay": Neuron_Decay_Int,
                   "variable_target_decay": Neuron_Decay_Int_Variable_n_min1}
        rng = np.random.default_rng(3)
        w = np.eye(4, 8) * np.array([0.5, 1.8, 0.9, 1.2])[:, None]
        for name, cls in classes.items():
            rule = Rule(name, 4, 8, n=2, b=1, kappa=0.5, beta_decay=0.7)
            neuron = legacy_neuron(rule)
            self.assertIsInstance(neuron, cls)
            for activations in ([3, 2, -1, 0.4], [-1, -2, -3, -4]):
                x = rng.normal(size=w.shape)
                x[np.arange(4), np.arange(4)] = np.asarray(activations) / np.linalg.norm(w, axis=-1)
                u = np.sum(w*x, axis=-1)
                mask, target = target_terms(u, rule)
                expected = w + mask[:, None] * (
                    x / np.sum(x*x, axis=-1, keepdims=True) * (target-(1-rule.beta_decay/rule.ns)*u)[:, None]
                    - rule.beta_decay/rule.ns*w)
                actual = jax.jit(lambda weights, inputs: update_weights(weights, inputs, rule))(jnp.asarray(w), jnp.asarray(x))
                np.testing.assert_allclose(actual, expected, atol=1e-13)
                direct = neuron.update(HyperDynConfig1(rule.b, rule.b+rule.kappa, rule.beta_decay),
                                       State1(w=jnp.asarray(w)), jnp.asarray(x)).w
                np.testing.assert_allclose(actual, direct, atol=1e-13)
                np.testing.assert_allclose(np.sum(np.asarray(actual)*x, axis=-1)[mask],
                                           np.broadcast_to(target, u.shape)[mask], atol=1e-13)
                np.testing.assert_allclose(np.asarray(actual)[~mask], w[~mask], atol=1e-13)
                self.assertFalse(np.allclose(np.linalg.norm(actual, axis=-1), 1))

    def test_initialization_matches_legacy_gaussian_norms(self):
        rule = Rule("nwta_decay", 4, 8, n=2, b=1, kappa=0.5, beta_decay=0.7)
        key = jax.random.PRNGKey(5)
        expected = legacy_neuron(rule).init_state(key, None).w
        np.testing.assert_allclose(initialize_weights(key, rule, jnp.float64), expected, atol=1e-14)
        self.assertFalse(np.allclose(np.linalg.norm(expected, axis=-1), 1))
        spherical = initialize_weights(key, replace(rule, name="nwta", beta_decay=None), jnp.float64)
        np.testing.assert_allclose(np.linalg.norm(spherical, axis=-1), 1, atol=1e-14)

    def test_decay_masks_include_zero_amplitude_variable_targets(self):
        rule = Rule("variable_target_decay", 4, 8, n=4, b=1, kappa=0.5, beta_decay=0.7)
        mask, target = target_terms(np.array([1.5, 0.5, 0.5, 0.5]), rule)
        np.testing.assert_array_equal(mask, np.ones(4, dtype=bool))
        self.assertEqual(target[0], 1.5)

    def test_linear_decay_reference_uses_actual_norms_and_no_spherical_term(self):
        # Selecting all branches with target zero gives a=-u and H*m=-(1+beta)*m.
        rule = Rule("variable_target_decay", 3, 1000, n=3, b=0, kappa=0, beta_decay=0.7)
        w = np.array([[1., 0.], [0., 2.], [0.5, 0.8]])
        for condition in ("independent", "shared"):
            q = np.diag(np.sum(w*w, axis=1)) if condition == "independent" else w @ w.T
            m = np.array([[0.4, -0.2]]) @ w.T
            result = theory_moments(rule, condition, q, m, np.random.default_rng(4), 20000, 20, 1000)
            np.testing.assert_allclose(result["selection_probability"], 1)
            self.assertTrue(np.all(np.abs(result["C"]-q) <= 6*result["C_se"] + 1e-12))
            error = np.abs(result["A"] + 1.7*m)
            from theory_check.statistics import batch_se
            self.assertTrue(np.all(error <= 6*batch_se(result["batch_A"], result["counts"]) + 1e-12))

    def test_beta_is_explicit_and_not_accepted_for_spherical_rules(self):
        raw = {"condition": "shared", "rules": [{"name": "nwta_decay", "b": 1, "kappa": 0.5}]}
        with self.assertRaisesRegex(ValueError, "nwta_decay.beta_decay"):
            resolve_config(raw)
        raw["rules"][0]["beta_decay"] = 0.7
        _, rules = resolve_config(raw)
        self.assertTrue(rules[0].is_decay)
        raw["rules"][0]["name"] = "nwta"
        with self.assertRaisesRegex(ValueError, "unused learning parameters"):
            resolve_config(raw)

    def test_saved_decay_snapshots_and_legacy_spherical_csv_compatibility(self):
        raw = {"nd": 100, "n": 10, "seeds": [0], "ages": [0, 1], "stages": [2],
               "sampling": {"forks": 4, "fork_steps": 2, "fork_batch_size": 2,
                            "uncertainty_groups": 2, "theory_samples": 16, "theory_batch_size": 8, "max_pairs": 2},
               "rules": [{"name": name, "b": 1.5, "kappa": 0.5, "beta_decay": 0.7}
                         for name in ("threshold_decay", "nwta_decay", "variable_target_decay")]}
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for condition in ("independent", "shared"):
                config, rules = resolve_config(dict(raw, condition=condition))
                root = Path(tmp) / condition
                root.mkdir()
                roots.append(root)
                run_experiment(config, rules, root)
                for path in root.glob("*/snapshot.npz"):
                    with np.load(path) as data:
                        np.testing.assert_allclose(data["Q"].diagonal(), data["weight_norms"]**2)
                        self.assertFalse(np.allclose(data["weight_norms"], 1))
            points, snapshots = load_runs(roots)
            self.assertEqual(len(snapshots), 6)
            self.assertEqual({p["beta_decay"] for p in points}, {0.7})
            self.assertEqual({s["coefficient_reference"] for s in snapshots}, {"local_leading_decay_at_actual_norms"})


if __name__ == "__main__":
    unittest.main()
