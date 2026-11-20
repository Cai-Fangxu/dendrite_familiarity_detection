"""Focused CPU checks for Softmax dynamics, restarts, calibration, and grid planning."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types

import jax
import jax.numpy as jnp
import numpy as np

from lib import neurons, driver, input_gen
from b_bk_grid_search import run_softmax as runner
from common_data_processing import calibrate_softmax as calibration
from common_data_processing import plan_grid_search as base
from common_data_processing import plan_softmax_grid_search as planner
from common_data_processing import softmax_calibration_registry as registry


class SoftmaxTests(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(13)
        self.hyper = neurons.HyperConfigSoftmax(4, 6)
        self.dynamic = neurons.HyperDynConfigSoftmax(
            *[jnp.asarray(x, dtype=jnp.float32) for x in (0.1, 1.2, 0.7)]
        )

    def test_update_matches_numpy_reference_and_forward(self):
        x = np.random.default_rng(2).normal(size=(4, 6)).astype(np.float32)
        for cls in (neurons.Neuron_Softmax_Int, neurons.Neuron_Softmax_NonInt):
            neuron = cls(self.hyper)
            state = neuron.init_state(self.key, self.dynamic)
            w = np.asarray(state.w, dtype=np.float64)
            u = np.sum(w * x, axis=-1)
            if cls is neurons.Neuron_Softmax_Int:
                shares = np.exp(0.7 * (u - u.max()))
                shares /= shares.sum()
                angles = 1.2 / np.sqrt(6) * shares
            else:
                angles = 1.2 / (4 * np.sqrt(6)) * np.exp(0.7 * u - 0.7**2 / 2)
            tangent = x - u[:, None] * w
            tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
            expected = w * np.cos(angles[:, None]) + tangent * np.sin(angles[:, None])
            expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
            actual = neuron.update(self.dynamic, state, jnp.asarray(x))
            np.testing.assert_allclose(actual.w, expected, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(np.linalg.norm(actual.w, axis=-1), 1, atol=2e-7)
            expected_y = np.maximum(np.sum(np.asarray(actual.w) * x, axis=-1) - 0.1, 0).sum()
            np.testing.assert_allclose(neuron.forward(self.dynamic, actual, jnp.asarray(x)[None]), [expected_y], rtol=2e-6)

    def test_uniform_allocation_and_readout_threshold_do_not_change_learning(self):
        left = neurons.Neuron_Softmax_Int(self.hyper)
        right = neurons.Neuron_Softmax_NonInt(self.hyper)
        dynamic = self.dynamic.replace(beta_select=jnp.float32(0))
        state = left.init_state(self.key, dynamic)
        x = input_gen.Normal(4, 6).generate(self.key, 1)[0]
        np.testing.assert_allclose(left.update(dynamic, state, x).w, right.update(dynamic, state, x).w, atol=1e-7)
        np.testing.assert_array_equal(
            left.update(dynamic, state, x).w,
            left.update(dynamic.replace(b=jnp.float32(10)), state, x).w,
        )

    def test_degenerate_inputs_and_long_term_unit_norm(self):
        for cls in (neurons.Neuron_Softmax_Int, neurons.Neuron_Softmax_NonInt):
            neuron = cls(self.hyper)
            state = neuron.init_state(self.key, self.dynamic)
            for x in (jnp.zeros_like(state.w), state.w, -2 * state.w):
                np.testing.assert_array_equal(neuron.update(self.dynamic, state, x).w, state.w)
            xs = input_gen.Normal(4, 6).generate(self.key, 256)
            final, _ = jax.lax.scan(
                lambda s, x: (neuron.update(self.dynamic, s, x), None), state, xs,
            )
            np.testing.assert_allclose(jnp.linalg.norm(final.w, axis=-1), 1, atol=2e-7)
            single = cls(neurons.HyperConfigSoftmax(4, 1))
            single_state = single.init_state(self.key, self.dynamic)
            np.testing.assert_array_equal(
                single.update(self.dynamic, single_state, jnp.ones((4, 1))).w, single_state.w,
            )

    def test_parallel_driver_matches_threshold_scan(self):
        neuron = neurons.Neuron_Softmax_Int(self.hyper)
        generator = input_gen.Normal(4, 6)
        bs = jnp.array([-0.2, 0.1, 0.6])
        one = driver.Driver1_bScan(neuron, generator, self.key, self.dynamic, resample_every=4, n_tested_patterns=4)
        dynamic = neurons.HyperDynConfigSoftmax(bs, jnp.full(3, 1.2), jnp.full(3, 0.7))
        parallel = driver.Driver1_parallel_states(neuron, generator, self.key, dynamic, resample_every=4, n_tested_patterns=4)
        phases = dict(n_init_steps=8, n_steps_phase1=12, n_steps_phase2=8, n_steps_phase3=8, test_every_phase3=2)
        one.run(**phases, bs=bs)
        parallel.run(**phases, bs=None)
        for name in ("transient_record", "steady_record", "upper99_s"):
            np.testing.assert_allclose(getattr(one, name), getattr(parallel, name), rtol=3e-6, atol=3e-6)

    def test_restarts_change_values_without_retracing_any_phase(self):
        for correlated in (False, True):
            traces = []
            parent = driver.DriverSoftmax_bScan_Correlated if correlated else driver.Driver1_bScan

            class CountingDriver(parent):
                def _update_no_record(self, *args):
                    traces.append("no_record")
                    return super()._update_no_record(*args)

                def _update_record1(self, *args):
                    traces.append("record1")
                    return super()._update_record1(*args)

                def _update_record3_sparse(self, *args):
                    traces.append("record3")
                    return super()._update_record3_sparse(*args)

            for cls in (neurons.Neuron_Softmax_Int, neurons.Neuron_Softmax_NonInt):
                neuron = cls(self.hyper)
                generator = (input_gen.Normal_Corr_across_Dendrites(4, 6, 2, self.key)
                             if correlated else input_gen.Normal(4, 6))
                instance = CountingDriver(neuron, generator, self.key, resample_every=4, n_tested_patterns=4)
                weights = []
                for repeat in range(2):
                    key, conn_key = runner.trial_keys(10, 4, 6, repeat)
                    dynamic = self.dynamic.replace(eta=jnp.float32(1.2 + repeat), beta_select=jnp.float32(0.7 + repeat))
                    if correlated:
                        dynamic = neurons.HyperDynConfigSoftmax_Correlated(
                            dynamic.b, dynamic.eta, dynamic.beta_select,
                            generator._get_connectivity_idx_mat(conn_key),
                        )
                    instance.re_init(key, dynamic)
                    instance.run(n_init_steps=8, n_steps_phase1=12, n_steps_phase2=8,
                                 n_steps_phase3=8, test_every_phase3=2, bs=jnp.array([0., 0.3]))
                    self.assertEqual(instance.transient_record.shape, (8, 2, 4))
                    self.assertEqual(instance.steady_record.shape, (4, 2, 4))
                    weights.append(np.array(instance.state.w))
                    if repeat == 0:
                        first_traces = traces.copy()
                    else:
                        self.assertEqual(traces, first_traces)
                self.assertFalse(np.array_equal(*weights))

    def test_dynamic_connectivity_and_seed_pairing(self):
        generator = input_gen.Normal_Corr_across_Dendrites(4, 6, 2, self.key)
        key, conn_key = runner.trial_keys(123, 4, 6, 0)
        connectivity = generator._get_connectivity_idx_mat(conn_key)
        axons = jax.random.normal(key, (3, 12))
        np.testing.assert_array_equal(generator.generate(key, 3, connectivity), axons[:, connectivity])
        np.testing.assert_array_equal(generator.generate(key, 3), axons[:, generator.conn_idx_mat])
        for row in np.asarray(connectivity):
            self.assertEqual(len(set(row)), 6)
        again = runner.trial_keys(123, 4, 6, 0)
        different = runner.trial_keys(123, 4, 6, 1)
        np.testing.assert_array_equal(again, (key, conn_key))
        self.assertFalse(np.array_equal(different, again))


class RunnerTests(unittest.TestCase):
    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.csv"
            path.write_text("4,6,1.5,0.1,0.375,-1,123\n")
            self.assertEqual(runner.load_params(path).shape, (1, 7))
            for content in ("4,6,0,0.1,0.3,-1,123", "4,6,1,0.1,-0.3,-1,123",
                            "4,6,1,0.1,0.3,0,123", "4.5,6,1,0.1,0.3,-1,123",
                            "4,6,1,nan,0.3,-1,123", "4,6,1,0.1,0.3,-1,1.5",
                            "4,6,1,0.1,0.3,-1"):
                path.write_text(content + "\n")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    runner.load_params(path)

    def test_runner_reuses_objects_restarts_and_roundtrips_output(self):
        starts, instances = [], []

        class RecordingDriver:
            def __init__(self, **kwargs):
                instances.append(self)

            def re_init(self, rng, hyperD, initial_state):
                starts.append((np.array(rng), hyperD, initial_state))

            def run(self, **kwargs):
                self.state = types.SimpleNamespace(w=np.ones((4, 6)))
                self.transient_record = np.zeros((8, 3, 4))
                self.steady_record = np.zeros((4, 3, 4))
                self.upper99_s = np.zeros(3)
                assert kwargs["n_init_steps"] == 8

        fake_utils = types.ModuleType("lib.utils")
        fake_utils.capacity_numba = lambda *args: np.array([10., 11., 12.])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            params = root / "params.csv"
            params.write_text("4,6,1.5,0.1,0.375,-1,123\n4,6,1.75,0.1,0.375,-1,123\n")
            settings = dict(params_file=params, output_root=root / "output", coeff=2, rule="int",
                            n_repeat=2, bs_size=3, resample_every=4, n_tested_patterns=4,
                            n_init_steps=8, n_steps_phase1=12, n_steps_phase2=8,
                            n_steps_phase3=8, test_every_phase3=2)
            with patch.multiple(runner, **settings), patch.object(
                runner.driver, "DriverSoftmax_bScan_Correlated", RecordingDriver,
            ), patch.dict("sys.modules", {"lib.utils": fake_utils}), patch(
                "lib.utils", fake_utils, create=True,
            ), redirect_stdout(io.StringIO()):
                runner.main()
                experiment = root / "output" / "int" / "coeff_2.0"
                grids = base.load_grids(experiment, base.Settings(n_trials=2), softmax=True)
                self.assertEqual(len(grids), 1)
                self.assertEqual(len(next(iter(grids.values()))), 6)
                self.assertEqual(len((experiment / "runs.jsonl").read_text().splitlines()), 2)
                with patch.object(runner, "n_init_steps", 12), self.assertRaisesRegex(ValueError, "settings differ"):
                    runner.main()
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(starts), 4)
        self.assertTrue(all(state is None for _, _, state in starts))
        self.assertFalse(np.array_equal(starts[0][0], starts[1][0]))
        np.testing.assert_array_equal(starts[0][0], starts[2][0])
        np.testing.assert_array_equal(starts[0][1].connectivity, starts[2][1].connectivity)
        self.assertFalse(np.array_equal(starts[0][1].connectivity, starts[1][1].connectivity))


class CalibrationTests(unittest.TestCase):
    def test_endpoints_and_reproducible_calibration(self):
        self.assertEqual(calibration.calibrate(4, 4, samples=100)["beta_select"], 0)
        self.assertEqual(calibration.calibrate(1, 1, samples=100)["beta_select"], 0)
        result = calibration.calibrate(8, 3, samples=1000, batch_size=100, rtol=1e-3)
        self.assertAlmostEqual(result["measured_ipr"], 1 / 3, delta=1e-3 / 3)
        self.assertEqual(result, calibration.calibrate(8, 3, samples=1000, batch_size=100, rtol=1e-3))
        endpoint = calibration.calibrate(4, 1, samples=300, rtol=0.05)
        self.assertTrue(np.isfinite(endpoint["beta_select"]))
        self.assertGreaterEqual(endpoint["measured_ipr"], 0.95)
        self.assertTrue(endpoint["approximate_n1"])

    def test_invalid_targets_and_unbracketed_endpoint(self):
        for nd, n in ((0, 1), (4, 0), (4, 5), (4, np.nan)):
            with self.assertRaises(ValueError):
                calibration.calibrate(nd, n)
        with self.assertRaisesRegex(ValueError, "bracket"):
            calibration.calibrate(4, 1, samples=100, max_beta=0.01)


class BatchCalibrationTests(unittest.TestCase):

    def test_reference_spacing_scales_by_target_n_not_mc_estimate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            source.write_text("0.3,1,2.5,8,6\n0.3,1.25,2.5,8,6\n")
            result = dict(nd=8, target_n=2.5, beta_select=1.2, effective_n=2.499)
            with patch.object(calibration, "calibrate", return_value=result), redirect_stdout(io.StringIO()):
                report = calibration.prepare_batch(source, output)
            params = runner.load_params(output)
            np.testing.assert_array_equal(params[:, 2], [2.5, 3.125])
            self.assertEqual(params[1, 2] - params[0, 2], 0.25 * 2.5)
            self.assertEqual(report["eta_scaling"], "target_n")
            self.assertEqual(report["input_columns"][1], "eta_reference")
            np.testing.assert_array_equal(report["output_rows"], params)
            self.assertEqual(source.read_text(), "0.3,1,2.5,8,6\n0.3,1.25,2.5,8,6\n")

    def test_scaled_eta_overflow_preserves_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            source.write_text("0.3,1e308,4,4,6\n")
            output.write_text("existing table")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Scaled eta"):
                calibration.prepare_batch(source, output, samples=100)
            self.assertEqual(output.read_text(), "existing table")

    def test_batch_preserves_order_and_calibrates_each_pair_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params_list_softmax.csv"
            source.write_text("0.3,1.5,2.5,8,6\n0.1,2,8,8,6\n-0.2,1.75,2.5,8,12\n")
            with patch.object(calibration, "calibrate", wraps=calibration.calibrate) as solve, redirect_stdout(io.StringIO()):
                report = calibration.prepare_batch(
                    source, output, simulation_seed=123, seed=456,
                    samples=200, batch_size=100, rtol=0.01,
                )
            self.assertEqual(solve.call_count, 2)
            params = runner.load_params(output)
            np.testing.assert_array_equal(params[:, :2], [[8, 6], [8, 6], [8, 12]])
            np.testing.assert_array_equal(params[:, 2:4], [[3.75, 0.3], [16, 0.1], [4.375, -0.2]])
            np.testing.assert_array_equal(params[:, 5:], [[-1, 123]] * 3)
            self.assertEqual(params[0, 4], params[2, 4])
            self.assertEqual(params[1, 4], 0)
            saved = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(saved, report)
            self.assertEqual([entry["seed"] for entry in saved["calibrations"]], [456, 456])
            # Reordering the input changes only row order, not calibration or seeds.
            source.write_text("-0.2,1.75,2.5,8,12\n0.1,2,8,8,6\n0.3,1.5,2.5,8,6\n")
            with redirect_stdout(io.StringIO()):
                calibration.prepare_batch(source, output, simulation_seed=123, seed=456,
                                          samples=200, batch_size=100, rtol=0.01)
            np.testing.assert_array_equal(runner.load_params(output), params[::-1])

    def test_batch_cli_single_row_and_legacy_json_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            source.write_text("0.3,1.5,4,4,6\n")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(calibration.main([
                    str(source), "--output", str(output), "--simulation-seed", "42",
                    "--seed", "99", "--samples", "100",
                ]), 0)
            np.testing.assert_array_equal(runner.load_params(output), [[4, 6, 6, 0.3, 0, -1, 42]])
            single = root / "single.json"
            with redirect_stdout(io.StringIO()):
                calibration.main(["--nd", "4", "--n", "4", "--samples", "100", "--output", str(single)])
            self.assertEqual(json.loads(single.read_text())["beta_select"], 0)

    def test_batch_defaults_resolve_from_script_and_keep_seeds_separate(self):
        with patch.object(calibration, "prepare_batch", return_value={"input_rows": []}) as prepare, redirect_stdout(io.StringIO()):
            calibration.main(["--seed", "123"])
        args, kwargs = prepare.call_args
        tmp_data = Path(calibration.__file__).resolve().parents[1] / "tmp_data"
        self.assertEqual(args, (tmp_data / "tmp.csv", tmp_data / "params_list_softmax.csv"))
        self.assertEqual(kwargs["seed"], 123)
        self.assertEqual(kwargs["simulation_seed"], 0)

    def test_invalid_input_or_failed_calibration_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            output.write_text("existing table")
            output.with_suffix(".report.json").write_text("existing report")
            for content in ("0,1,2,4", "0,0,2,4,6", "0,1,0.5,4,6", "0,1,5,4,6",
                            "0,1,2,4.5,6", "0,1,2,4,6.5", "nan,1,2,4,6",
                            "0,1,2,4,6\n0,-1,2,4,6"):
                source.write_text(content + "\n")
                with self.subTest(content=content), patch.object(calibration, "calibrate") as solve, self.assertRaises(ValueError):
                    calibration.prepare_batch(source, output)
                solve.assert_not_called()
                self.assertEqual(output.read_text(), "existing table")
            source.write_text("0,1,1,4,6\n")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "bracket"):
                calibration.prepare_batch(source, output, samples=100, max_beta=0.01)
            self.assertEqual(output.read_text(), "existing table")
            self.assertEqual(output.with_suffix(".report.json").read_text(), "existing report")
            for seed in (-1, 2**32, 0.5):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    calibration.prepare_batch(source, output, simulation_seed=seed)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                calibration.prepare_batch(source, source)
            source_report = output.with_suffix(".report.json")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                calibration.prepare_batch(source_report, output)

    def test_cli_rejects_ambiguous_modes(self):
        from contextlib import redirect_stderr
        for args in (["--nd", "4"], ["--n", "2"], ["tmp.csv", "--nd", "4", "--n", "2"],
                     ["--nd", "4", "--n", "2", "--simulation-seed", "1"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                calibration.main(args)
            self.assertEqual(error.exception.code, 2)


class CalibrationRegistryTests(unittest.TestCase):
    def test_two_batches_preserve_history_and_planner_discovers_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            source.write_text("0.3,1,2,4,6\n")
            with redirect_stdout(io.StringIO()):
                first = calibration.prepare_batch(source, output, samples=100, seed=123)
            source.write_text("0.3,1,4,4,6\n")
            with redirect_stdout(io.StringIO()):
                second = calibration.prepare_batch(source, output, samples=100, seed=123)
                calibration.prepare_batch(source, output, samples=100, seed=123)
            registry_path = root / registry.REGISTRY_NAME
            records = registry.read_calibrations(registry_path)
            self.assertEqual(len(records), 2)
            self.assertEqual({record["target_n"] for record in records}, {2., 4.})
            self.assertEqual(len(second["calibrations"]), 1)
            self.assertEqual(second["calibration_registry"], str(registry_path.resolve()))
            data_dir = root / "results" / "int"
            data_dir.mkdir(parents=True)
            keys = [base.SoftmaxGridKey(4, 6, record["beta_select"]) for record in records]
            sources = registry.discover_sources(data_dir)
            self.assertEqual(sources, [registry_path.resolve()])
            scales = planner.load_eta_scales(dict.fromkeys(keys), sources)
            self.assertEqual(set(scales.values()), {2., 4.})
            # Existing commands passing only the current report also find history.
            sources = registry.discover_sources(data_dir, [output.with_suffix(".report.json")])
            self.assertEqual(planner.load_eta_scales(dict.fromkeys(keys), sources), scales)

    def test_previous_report_migrates_and_conflicts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            previous = dict(nd=4, target_n=2., beta_select=1.5)
            report_path = output.with_suffix(".report.json")
            report_path.write_text(json.dumps(dict(calibrations=[previous])))
            source.write_text("0.3,1,4,4,6\n")
            with redirect_stdout(io.StringIO()):
                calibration.prepare_batch(source, output, samples=100)
            registry_path = root / registry.REGISTRY_NAME
            self.assertIn(previous, registry.read_calibrations(registry_path))
            before = {path: path.read_bytes() for path in (output, report_path, registry_path)}
            source.write_text("0.3,1,3,4,6\n")
            with patch.object(calibration, "calibrate", return_value=dict(previous, target_n=3.)), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    calibration.prepare_batch(source, output, samples=100)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_reference_cli_uses_registry_without_report_argument(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir = root / "results"
            geometry = data_dir / "nd8_ns6"
            geometry.mkdir(parents=True)
            (geometry / "beta_select1.2.csv").write_text("0.15,0.625,-1,10,12\n")
            registry.update_registry(root / registry.REGISTRY_NAME,
                                     [dict(nd=8, target_n=2.5, beta_select=1.2)])
            with redirect_stdout(io.StringIO()):
                planner.main([str(data_dir), "--eta-reference-spacing", "0.25",
                              "--b-size", "3", "--n-trials", "2"])
            params = runner.load_params(data_dir / "next_grid" / "params_list_softmax.csv")
            np.testing.assert_array_equal(params[:, 2], [1.25])

    def test_custom_registry_link_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "tmp.csv", root / "params.csv"
            source.write_text("0.3,1,4,4,6\n")
            custom = root / "history" / "custom.json"
            with redirect_stdout(io.StringIO()):
                calibration.prepare_batch(source, output, samples=100, registry_path=custom)
            self.assertIn(custom.resolve(), registry.discover_sources(
                root / "results", [output.with_suffix(".report.json")],
            ))
            for path in (source, output, output.with_suffix(".report.json")):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    calibration.prepare_batch(source, output, registry_path=path)


class PlannerTests(unittest.TestCase):

    def test_reference_spacing_uses_different_scale_per_grid(self):
        settings = base.Settings(top_k=1, b_size=3, bk_size=3, n_trials=2)
        keys = [base.SoftmaxGridKey(8, 6, 1.2), base.SoftmaxGridKey(8, 6, 0.5)]
        scales = dict(zip(keys, (2., 4.)))
        grids = {
            key: [base.Point(0.15 * b, 0.25 * scale * eta, -1,
                             10 if (b, eta) == (1, 3) else 1, 0, 2)
                  for b in range(3) for eta in (1, 2, 3)]
            for key, scale in scales.items()
        }
        reports = planner.analyze(grids, settings, eta_scales=scales)
        for report in reports:
            scale = next(scale for key, scale in scales.items() if key.beta_select == report["beta_select"])
            self.assertEqual(report["eta_spacing"], 0.25 * scale)
            self.assertEqual(report["proposals"][0]["eta"], scale)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            planner.write_outputs(root, grids, reports, settings, root, 123, eta_scales=scales)
            params = runner.load_params(root / "params_list_softmax.csv")
            for row in params:
                scale = next(scale for key, scale in scales.items() if key.beta_select == row[4])
                self.assertEqual(row[2], scale) # already scaled; no second multiplication
            audit = json.loads((root / "grid_search_report.json").read_text())
            self.assertEqual(audit["settings"]["eta_reference_spacing"], 0.25)
            self.assertEqual(audit["eta_scaling"], "target_n")

    def test_calibration_mapping_rejects_missing_and_conflicting_targets(self):
        key = base.SoftmaxGridKey(8, 6, 1.2)
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "params.report.json"
            calibration_row = dict(nd=8, target_n=2.5, beta_select=1.2, effective_n=2.499)
            report.write_text(json.dumps(dict(calibrations=[calibration_row])))
            self.assertEqual(planner.load_eta_scales({key: []}, [report]), {key: 2.5})
            with self.assertRaisesRegex(ValueError, "lack"):
                planner.load_eta_scales({base.SoftmaxGridKey(8, 6, 2.): []}, [report])
            report.write_text(json.dumps(dict(calibrations=[
                calibration_row, dict(calibration_row, target_n=3.),
            ])))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                planner.load_eta_scales({key: []}, [report])

    def test_reference_spacing_cli_writes_actual_eta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry = root / "nd8_ns6"
            geometry.mkdir()
            (geometry / "beta_select1.2.csv").write_text("0.15,0.625,-1,10,12\n")
            mapping = root / "calibration.json"
            mapping.write_text(json.dumps(dict(calibrations=[
                dict(nd=8, target_n=2.5, beta_select=1.2),
            ])))
            with redirect_stdout(io.StringIO()):
                planner.main([str(root), "--eta-reference-spacing", "0.25",
                              "--calibration-report", str(mapping), "--b-size", "3", "--n-trials", "2"])
            params = runner.load_params(root / "next_grid" / "params_list_softmax.csv")
            np.testing.assert_array_equal(params[:, 2], [1.25])

    def test_positive_eta_boundary_does_not_prevent_b_extension(self):
        points = [base.Point(0.15 * b, 0.25 * eta, -1, 100 if (b, eta) == (2, 1) else 1, 0, 2)
                  for b in range(3) for eta in (1, 2, 3)]
        settings = base.Settings(top_k=1, b_size=3, bk_size=3, n_trials=2)
        report = planner.analyze({base.SoftmaxGridKey(4, 6, 0.375): points}, settings)[0]
        self.assertEqual(report["status"], "extend_b")
        self.assertTrue(all(p["eta"] > 0 for p in report["proposals"]))

    def test_fractional_beta_loader_output_and_sentinel_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry = root / "nd4_ns6"
            geometry.mkdir()
            path = geometry / "beta_select0.375.csv"
            path.write_text("0.15,0.25,-1,10,12\n")
            settings = base.Settings(b_size=3, n_trials=2)
            grids = base.load_grids(root, settings, softmax=True)
            report = planner.analyze(grids, settings)
            planner.write_outputs(root / "next", grids, report, settings, root, 123)
            params = runner.load_params(root / "next" / "params_list_softmax.csv")
            self.assertTrue(np.all(params[:, 4] == 0.375))
            self.assertTrue(np.all(params[:, 5] == -1))
            self.assertTrue(np.all(params[:, 2] > 0))
            self.assertIn("beta_select", (root / "next" / "tmp_df.csv").read_text().splitlines()[0])
            audit = json.loads((root / "next" / "grid_search_report.json").read_text())
            self.assertIn("eta_spacing", audit["settings"])
            for row in ("0.15,0.25,0,10,12", "0.15,0,-1,10,12"):
                path.write_text(row + "\n")
                with self.assertRaises(ValueError):
                    base.load_grids(root, settings, softmax=True)


if __name__ == "__main__":
    unittest.main()
