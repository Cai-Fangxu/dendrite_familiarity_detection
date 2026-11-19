"""Small CPU tests for fixed random slopes; no production sweeps are run."""

import tempfile
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from lib import driver, input_gen, neurons
from b_bk_grid_search import run_variableSlope as runner


class VariableSlopeTests(unittest.TestCase):
    def setUp(self):
        self.hyper = neurons.HyperConfig1(nd=4, ns=6, n=2, b=0.1, bk=1.5, beta=0.4)
        self.neuron = neurons.Neuron_Decay_Int(self.hyper)
        self.generator = input_gen.Normal(4, 6)
        self.bs = jnp.array([-0.2, 0.1, 0.6])
        self.key = jax.random.PRNGKey(21)
        self.phase_args = dict(
            n_init_steps=8, n_steps_phase1=12, n_steps_phase2=8,
            n_steps_phase3=8, test_every_phase3=2, bs=self.bs,
        )

    def dynamic(self, slopes):
        return neurons.HyperDynConfig1_VariableSlope(
            b=0.1, bk=1.5, beta=0.4, slopes=jnp.asarray(slopes, dtype=jnp.float32),
        )

    def make_driver(self, cls, hyperD):
        return cls(
            self.neuron, self.generator, self.key, hyperD,
            resample_every=4, n_tested_patterns=4,
        )

    def test_unit_slopes_match_original_driver_through_all_phases(self):
        original = self.make_driver(driver.Driver1_bScan, neurons.HyperDynConfig1(0.1, 1.5, 0.4))
        weighted = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([1] * 4))
        original.run(**self.phase_args)
        weighted.run(**self.phase_args)
        for name in ("transient_record", "steady_record", "upper99_s"):
            np.testing.assert_allclose(getattr(weighted, name), getattr(original, name), rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(weighted.state.w, original.state.w)
        self.assertEqual(weighted.transient_record.shape, (8, 3, 4))
        self.assertEqual(weighted.steady_record.shape, (4, 3, 4))

    def test_weighted_recordings_match_explicit_sequential_reference(self):
        hyperD = self.dynamic([0.0, 0.5, 1.5, 2.0])
        weighted = self.make_driver(driver.Driver1_bScan_VariableSlope, hyperD)
        initial = self.neuron.init_state(self.key, hyperD)
        xs = self.generator.generate(jax.random.PRNGKey(1), 4)
        probes = self.generator.generate(jax.random.PRNGKey(2), 4)
        state, expected = initial, []
        for x in xs:
            state = self.neuron.update(hyperD, state, x)
            overlaps = np.sum(np.asarray(probes) * np.asarray(state.w), axis=-1)
            expected.append(np.sum(
                np.maximum(overlaps[None] - np.asarray(self.bs)[:, None, None], 0)
                * np.asarray(hyperD.slopes)[None, None], axis=-1,
            ))
        record = jax.jit(weighted._update_record1)
        final, actual = record(hyperD, initial, self.bs, xs, probes)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(final.w, state.w, rtol=1e-6, atol=1e-6)

        sparse = jax.jit(weighted._update_record3_sparse, static_argnums=(5,))
        sparse_final, sparse_actual = sparse(hyperD, initial, self.bs, xs, probes, 2)
        np.testing.assert_allclose(sparse_actual[::2], np.asarray(expected)[::2], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(sparse_actual[1::2], -100.0)
        np.testing.assert_array_equal(sparse_final.w, final.w)

    def test_reinitialization_changes_slopes_without_changing_learning(self):
        weighted = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([1] * 4))
        weighted.run(**self.phase_args)
        previous_w = np.asarray(weighted.state.w)
        transient = weighted.transient_record.copy()
        steady = weighted.steady_record.copy()
        weighted.re_init(self.key, self.dynamic([2] * 4), initial_state=None)
        self.assertIsNone(weighted.state)
        weighted.run(**self.phase_args)
        np.testing.assert_array_equal(weighted.state.w, previous_w)
        np.testing.assert_allclose(weighted.transient_record, 2 * transient, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(weighted.steady_record, 2 * steady, rtol=1e-6, atol=1e-6)

    def test_slopes_are_dynamic_under_jit(self):
        weighted = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([1] * 4))
        state = self.neuron.init_state(self.key, weighted.hyperD)
        xs = self.generator.generate(self.key, 4)
        traces = []

        @jax.jit
        def record(hyperD):
            traces.append("trace")
            return weighted._update_record1(hyperD, state, self.bs, xs, xs)[1]

        first = record(self.dynamic([1] * 4))
        second = record(self.dynamic([2] * 4))
        np.testing.assert_allclose(second, 2 * first, rtol=1e-6, atol=1e-6)
        self.assertEqual(len(traces), 1)

    def test_nonuniform_slopes_preserve_weights_but_change_responses(self):
        left = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([1] * 4))
        right = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([0.5, 0.75, 1.25, 1.5]))
        left.run(**self.phase_args)
        right.run(**self.phase_args)
        np.testing.assert_array_equal(left.state.w, right.state.w)
        self.assertFalse(np.allclose(left.transient_record, right.transient_record))

    def test_wrong_slope_shape_is_rejected(self):
        weighted = self.make_driver(driver.Driver1_bScan_VariableSlope, self.dynamic([1, 2]))
        with self.assertRaisesRegex(ValueError, "shape"):
            weighted.run(**self.phase_args)


class RunnerTests(unittest.TestCase):
    def test_slopes_are_order_independent_and_trials_resample(self):
        first = runner.sample_slopes(20, 0, 123, 0.5, 1.5)
        runner.sample_slopes(10, 3, 123, 0.5, 1.5)
        again = runner.sample_slopes(20, 0, 123, 0.5, 1.5)
        other = runner.sample_slopes(20, 1, 123, 0.5, 1.5)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all((first >= 0.5) & (first <= 1.5)))
        self.assertNotAlmostEqual(float(first.mean()), 1.0, places=6)

    def test_constant_and_invalid_bounds(self):
        np.testing.assert_array_equal(runner.sample_slopes(4, 0, 1, 1, 1), np.ones(4))
        for bounds in [(-1, 1), (2, 1), (0, np.inf), (np.nan, 1)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                runner.sample_slopes(4, 0, 1, *bounds)

    def test_runner_restarts_trials_and_writes_existing_csv_format(self):
        # Isolate orchestration from the separately tested JAX driver and the
        # optional Numba dependency; production capacity code is not replaced.
        from contextlib import redirect_stdout
        import io
        import types
        from unittest.mock import patch

        starts = []

        class RecordingDriver:
            def __init__(self, **kwargs):
                pass

            def re_init(self, rng, hyperD, initial_state):
                starts.append((np.asarray(rng), np.asarray(hyperD.slopes), initial_state))

            def run(self, **kwargs):
                self.transient_record = np.zeros((8, 3, 4))
                self.steady_record = np.zeros((4, 3, 4))
                self.upper99_s = np.zeros(3)

        fake_utils = types.ModuleType("lib.utils")
        fake_utils.capacity_numba = lambda transient, steady, upper: np.array([11.0, 12.0, 13.0])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            params = root / "params.csv"
            params.write_text("4,6,1.5,0.1,2,0.4,123\n4,6,1.75,0.1,2,0.5,123\n")
            settings = dict(
                params_file=params, output_folder=root / "results",
                n_repeat=2, bs_size=3, resample_every=4, n_tested_patterns=4,
                n_init_steps=8, n_steps_phase1=12, n_steps_phase2=8,
                n_steps_phase3=8, test_every_phase3=2,
            )
            with patch.multiple(runner, **settings), patch.object(
                runner.driver, "Driver1_bScan_VariableSlope", RecordingDriver,
            ), patch.dict("sys.modules", {"lib.utils": fake_utils}), patch(
                "lib.utils", fake_utils, create=True,
            ), redirect_stdout(io.StringIO()):
                runner.main()

            output = np.loadtxt(root / "results" / "nd4_ns6" / "n2.csv", delimiter=",")
            self.assertEqual(output.shape, (6, 5))
            np.testing.assert_allclose(output[:3, 0], [-0.05, 0.1, 0.25])
            np.testing.assert_allclose(output[:3, 1], 1.5)
            np.testing.assert_allclose(output[3:, 1], 1.75)
            np.testing.assert_allclose(output[:, 3:], np.array([[11, 11], [12, 12], [13, 13]] * 2))

        self.assertEqual(len(starts), 4)
        self.assertTrue(all(state is None for _, _, state in starts))
        self.assertFalse(np.array_equal(starts[0][0], starts[1][0]))
        self.assertFalse(np.array_equal(starts[0][1], starts[1][1]))
        np.testing.assert_array_equal(starts[0][1], starts[2][1])
        np.testing.assert_array_equal(starts[1][1], starts[3][1])

    def test_single_row_parameters_and_invalid_schemas(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.csv"
            path.write_text("4,6,1.5,0.1,2,0.4,123\n")
            self.assertEqual(runner.load_params(path).shape, (1, 7))
            for content in [
                ",".join(["1"] * 15),
                "4,6,1.5,0.1,2.5,0.4,123",
                "4,6,1.5,0.1,5,0.4,123",
                "4,6,1.5,0.1,2,nan,123",
            ]:
                path.write_text(content + "\n")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    runner.load_params(path)


if __name__ == "__main__":
    unittest.main()
