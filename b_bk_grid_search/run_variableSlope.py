"""Grid search for b and bk with fixed random dendritic ReLU slopes.

Edit the settings and model/input/driver choices below, as in run.py.
Input rows: nd, ns, bk, bias_center, n, beta, seed (no header).
Outputs: b, bk, beta, capacity_0, ..., capacity_5 (no header).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import numpy as np
import jax
import jax.numpy as jnp

from lib import neurons, input_gen, driver


# Settings: paths resolve from this script, regardless of the working directory.
tmp_data_folder = Path(__file__).resolve().parents[1] / "tmp_data"
params_file = tmp_data_folder / "params_list_tmp.csv"
slope_min, slope_max = 0.333, 3.0
slope_seed = 20240305 # independent of row seeds; keep fixed when extending a grid
output_folder = tmp_data_folder / "int_variableSlope" / f"slope_{slope_min:g}_{slope_max:g}"
n_repeat = 6

bs_size = 9
b_spacing = 0.15
resample_every = 1000
n_tested_patterns = 1000
n_init_steps = 10000
n_steps_phase1 = 3000
n_steps_phase2 = 20000
n_steps_phase3 = 10000
test_every_phase3 = 50


def sample_slopes(nd, repeat_idx, seed, lower, upper):
    """Sample solely by slope seed, nd, and trial, independent of row ordering.

    Slopes are shared across n, ns, b, bk, and simulation seeds for a given nd
    and trial. Do not normalize the sample mean. Equal endpoints give a
    constant slope, useful for a baseline check.
    """
    if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper:
        raise ValueError("Slope bounds must be finite and satisfy 0 <= lower <= upper")
    key = jax.random.fold_in(jax.random.PRNGKey(seed), nd)
    key = jax.random.fold_in(key, repeat_idx)
    return jax.random.uniform(key, (nd,), minval=float(lower), maxval=float(upper))


def load_params(path):
    """Validate the standard seven-column schema before any simulation runs."""
    params = np.loadtxt(path, delimiter=",", ndmin=2)
    if params.size == 0 or params.shape[1] != 7:
        raise ValueError(
            "Expected seven columns: nd, ns, bk, bias_center, n, beta, seed. "
            "Parallel-state parameter tables with one beta per b are not supported."
        )
    if not np.isfinite(params).all():
        raise ValueError("Parameter values must be finite")
    integers = params[:, [0, 1, 4, 6]]
    if not np.equal(integers, np.floor(integers)).all():
        raise ValueError("nd, ns, n, and seed must be integers")
    if np.any(params[:, 0:2] < 1) or np.any(params[:, 4] < 1) or np.any(params[:, 4] > params[:, 0]):
        raise ValueError("Require nd >= n >= 1 and ns >= 1")
    if np.any(params[:, 6] < 0) or np.any(params[:, 6] > 2**32 - 1):
        raise ValueError("Seeds must be in [0, 2**32 - 1]")
    return params


def main():
    """Append six independent-restart capacity trials per grid point."""
    params_list = load_params(params_file)
    if bs_size < 1 or bs_size % 2 == 0 or b_spacing <= 0 or n_repeat < 1:
        raise ValueError("Use a positive odd bs_size, positive b_spacing, and n_repeat >= 1")
    if resample_every < 1 or n_tested_patterns < 1:
        raise ValueError("resample_every and n_tested_patterns must be positive")
    for steps in (n_init_steps, n_steps_phase1, n_steps_phase2, n_steps_phase3):
        if steps < 1 or steps % resample_every:
            raise ValueError("Phase lengths must be positive multiples of resample_every")
    if n_tested_patterns % resample_every or n_steps_phase1 <= n_tested_patterns:
        raise ValueError("Use block-aligned probes and n_steps_phase1 > n_tested_patterns")
    if test_every_phase3 < 1 or resample_every % test_every_phase3:
        raise ValueError("test_every_phase3 must be a positive divisor of resample_every")
    sample_slopes(1, 0, slope_seed, slope_min, slope_max) # validate bounds before running

    # Keep imports side-effect free for inspecting settings or testing helpers.
    from lib import utils

    old_geometry = None
    for line_idx, row in enumerate(params_list):
        nd, ns, bk, bias_center, n, beta, seed = row
        nd, ns, n, seed = int(nd), int(ns), int(n), int(seed)
        print(f"\n#############Line{line_idx + 1}/{len(params_list)}#####################", flush=True)
        print(f"nd: {nd}; ns: {ns}; n: {n}; bk: {bk:.3f}", flush=True)
        bs = bias_center + b_spacing * jnp.arange(-(bs_size // 2), bs_size // 2 + 1)

        # Geometry and top-k are static; beta, bk, thresholds, and slopes are dynamic.
        if (nd, ns, n) != old_geometry:
            old_geometry = (nd, ns, n)
            hyper = neurons.HyperConfig1(nd=nd, ns=ns, n=n, b=bias_center, bk=bk, beta=beta)
            neuron = neurons.Neuron_Decay_Int(hyperparameters=hyper)
            # neuron = neurons.Neuron_Decay_NonInt(hyperparameters=hyper)
            # Variable-target learning depends on b and needs a parallel-state driver.

            xs_gen = input_gen.Normal(nd=nd, ns=ns)
            # Fixed connectivity key makes optional correlated input independent of row order.
            # conn_key = jax.random.fold_in(jax.random.PRNGKey(slope_seed + 1), nd)
            # xs_gen = input_gen.Normal_Corr_across_Dendrites(nd, ns, coeff=2, key=conn_key)

            my_driver = driver.Driver1_bScan_VariableSlope(
                neuron=neuron, xs_generator=xs_gen, rng=jax.random.PRNGKey(seed),
                resample_every=resample_every, n_tested_patterns=n_tested_patterns,
            )
            # my_driver = driver.Driver1_bScan(neuron=neuron, xs_generator=xs_gen, rng=jax.random.PRNGKey(seed), resample_every=resample_every, n_tested_patterns=n_tested_patterns)

        capa_list = np.zeros((bs_size, n_repeat))
        for repeat_idx in range(n_repeat):
            print(repeat_idx + 1, end="    ", flush=True)
            slopes = sample_slopes(nd, repeat_idx, slope_seed, slope_min, slope_max)
            hyperD = neurons.HyperDynConfig1_VariableSlope(
                b=bias_center, bk=bk, beta=beta, slopes=slopes,
            )
            # Fresh states and streams across trials; common random numbers across
            # parameter rows with the same seed/geometry make comparisons less noisy.
            trial_key = jax.random.PRNGKey(seed)
            for value in (nd, ns, n, repeat_idx):
                trial_key = jax.random.fold_in(trial_key, value)
            my_driver.re_init(rng=trial_key, hyperD=hyperD, initial_state=None)
            my_driver.run(
                n_init_steps=n_init_steps,
                n_steps_phase1=n_steps_phase1,
                n_steps_phase2=n_steps_phase2,
                n_steps_phase3=n_steps_phase3,
                test_every_phase3=test_every_phase3,
                bs=bs,
            )
            capa_list[:, repeat_idx] = utils.capacity_numba(
                my_driver.transient_record, my_driver.steady_record, my_driver.upper99_s,
            )

        tmp_data = np.column_stack((np.asarray(bs), np.full(bs_size, bk), np.full(bs_size, beta), capa_list))
        folder_path = output_folder / f"nd{nd}_ns{ns}"
        folder_path.mkdir(parents=True, exist_ok=True)
        with (folder_path / f"n{n}.csv").open("a") as handle:
            np.savetxt(handle, tmp_data, delimiter=",", fmt=["%.3f", "%.3f", "%.6f"] + ["%.2f"] * n_repeat)

        my_driver.transient_record = None
        my_driver.steady_record = None
        my_driver.upper99_s = None
        del tmp_data, capa_list
        gc.collect()
        if (line_idx + 1) % 20 == 0:
            jax.clear_caches()


if __name__ == "__main__":
    main()
