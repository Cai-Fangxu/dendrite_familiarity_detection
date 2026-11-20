"""Grid search over b and eta for the two Softmax learning rules.

Edit the settings below. Input: nd, ns, eta, b_center, beta_select, -1, seed.
The eta column is the actual learning strength, already scaled during preparation.
Every trial restarts weights, inputs, and (when used) shared-axon connectivity.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from lib import driver, input_gen, neurons


tmp_data_folder = Path(__file__).resolve().parents[1] / "tmp_data"
params_file = tmp_data_folder / "params_list_softmax.csv"
output_root = tmp_data_folder / "softmax"
rule = "int" # "int" or "nonint"
coeff = 1 # None: independent inputs; a finite number >= 1: shared axons
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


def load_params(path):
    """Read and validate the full seven-column table before starting a sweep."""
    params = np.loadtxt(path, delimiter=",", ndmin=2)
    if params.size == 0 or params.shape[1] != 7:
        raise ValueError("Expected nd, ns, eta, b_center, beta_select, -1, seed")
    if not np.isfinite(params).all():
        raise ValueError("All parameters must be finite")
    integers = params[:, [0, 1, 6]]
    if not np.equal(integers, np.floor(integers)).all():
        raise ValueError("nd, ns, and seed must be integers")
    if np.any(params[:, :2] < 1) or np.any(params[:, 2] <= 0) or np.any(params[:, 4] < 0):
        raise ValueError("Require nd, ns >= 1, eta > 0, and beta_select >= 0")
    if np.any(params[:, 5] != -1):
        raise ValueError("The unused decay beta column must equal -1")
    if np.any(params[:, 6] < 0) or np.any(params[:, 6] > 2**32 - 1):
        raise ValueError("Seeds must be in [0, 2**32 - 1]")
    return params


def trial_keys(seed, nd, ns, repeat_idx):
    """Match randomness across parameter settings and rules, independently of row order."""
    key = jax.random.PRNGKey(seed)
    for value in (nd, ns, repeat_idx):
        key = jax.random.fold_in(key, value)
    return jax.random.split(key) # separate simulation and connectivity streams


def validate_settings():
    """Reject incompatible shapes and phase schedules before compiling or writing."""
    if rule not in ("int", "nonint"):
        raise ValueError("rule must be int or nonint")
    if coeff is not None and (not np.isfinite(coeff) or coeff < 1):
        raise ValueError("coeff must be None or finite and >= 1")
    counts = (n_repeat, bs_size, resample_every, n_tested_patterns,
              n_init_steps, n_steps_phase1, n_steps_phase2, n_steps_phase3, test_every_phase3)
    if any(not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("Trial counts, scan sizes, and phase lengths must be positive integers")
    if bs_size % 2 != 1 or not np.isfinite(b_spacing) or b_spacing <= 0:
        raise ValueError("Use an odd bs_size and finite positive b_spacing")
    for steps in (n_init_steps, n_steps_phase1, n_steps_phase2, n_steps_phase3):
        if steps % resample_every:
            raise ValueError("Phase lengths must be multiples of resample_every")
    if n_tested_patterns % resample_every or n_steps_phase1 <= n_tested_patterns:
        raise ValueError("Use block-aligned probes and n_steps_phase1 > n_tested_patterns")
    if resample_every % test_every_phase3:
        raise ValueError("test_every_phase3 must divide resample_every")


def experiment_settings():
    """Settings that must stay consistent when appending results to an experiment."""
    return dict(
        model_version=1, rule=rule, coeff=coeff, n_repeat=n_repeat,
        bs_size=bs_size, b_spacing=b_spacing, resample_every=resample_every,
        n_tested_patterns=n_tested_patterns, n_init_steps=n_init_steps,
        n_steps_phase1=n_steps_phase1, n_steps_phase2=n_steps_phase2,
        n_steps_phase3=n_steps_phase3, test_every_phase3=test_every_phase3,
        independent_restarts=True, jax_enable_x64=bool(jax.config.jax_enable_x64),
        schema=["b", "eta", "beta"] + [f"capacity_{i}" for i in range(n_repeat)],
    )


def prepare_output(folder):
    """Prevent accidental mixing of incompatible trial protocols in one directory."""
    settings = experiment_settings()
    path = folder / "settings.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"{path}: settings differ; choose a separate output_root")
    else:
        if folder.exists() and any(folder.glob("nd*_ns*/*.csv")):
            raise ValueError(f"{folder}: existing results have no settings.json")
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")


def make_driver(nd, ns):
    """Construct one reusable compiled object per geometry and input condition."""
    model = neurons.Neuron_Softmax_Int if rule == "int" else neurons.Neuron_Softmax_NonInt
    neuron = model(neurons.HyperConfigSoftmax(nd, ns))
    key = jax.random.PRNGKey(0)
    if coeff is None:
        generator = input_gen.Normal(nd, ns)
        driver_type = driver.Driver1_bScan
        sample_connectivity = None
    else:
        generator = input_gen.Normal_Corr_across_Dendrites(nd, ns, coeff, key)
        driver_type = driver.DriverSoftmax_bScan_Correlated
        sample_connectivity = jax.jit(generator._get_connectivity_idx_mat)
    instance = driver_type(
        neuron=neuron, xs_generator=generator, rng=key,
        resample_every=resample_every, n_tested_patterns=n_tested_patterns,
    )
    return instance, sample_connectivity


def main():
    """Run independently restarted trials and append capacities with seed provenance."""
    validate_settings()
    params = load_params(params_file)
    from lib import utils

    condition = "infty" if coeff is None else repr(float(coeff))
    folder = output_root / rule / f"coeff_{condition}"
    prepare_output(folder)
    # Retain compiled objects across nonadjacent rows of the same geometry.
    drivers = {}
    for row_idx, row in enumerate(params):
        nd, ns, eta, b_center, beta_select, _, seed = row
        nd, ns, seed = int(nd), int(ns), int(seed)
        if (nd, ns) not in drivers:
            drivers[nd, ns] = make_driver(nd, ns)
        instance, sample_connectivity = drivers[nd, ns]
        print(f"Row {row_idx + 1}/{len(params)}: nd={nd}, ns={ns}, beta_select={beta_select:g}, eta={eta:g}", flush=True)
        # Explicit dtypes prevent weak scalar types from causing extra traces.
        bs = jnp.asarray(b_center + b_spacing * np.arange(-(bs_size // 2), bs_size // 2 + 1), dtype=jnp.float32)
        capacities = np.empty((bs_size, n_repeat))
        for repeat_idx in range(n_repeat):
            key, conn_key = trial_keys(seed, nd, ns, repeat_idx)
            values = dict(b=jnp.asarray(b_center, dtype=jnp.float32),
                          eta=jnp.asarray(eta, dtype=jnp.float32),
                          beta_select=jnp.asarray(beta_select, dtype=jnp.float32))
            if sample_connectivity is None:
                hyperD = neurons.HyperDynConfigSoftmax(**values)
            else:
                hyperD = neurons.HyperDynConfigSoftmax_Correlated(
                    **values, connectivity=sample_connectivity(conn_key),
                )
            instance.re_init(rng=key, hyperD=hyperD, initial_state=None)
            instance.run(
                n_init_steps=n_init_steps, n_steps_phase1=n_steps_phase1,
                n_steps_phase2=n_steps_phase2, n_steps_phase3=n_steps_phase3,
                test_every_phase3=test_every_phase3, bs=bs,
            )
            if not all(np.isfinite(value).all() for value in (
                instance.state.w, instance.transient_record, instance.steady_record,
            )):
                raise FloatingPointError(f"Nonfinite simulation at row {row_idx + 1}, repeat {repeat_idx}")
            capacities[:, repeat_idx] = utils.capacity_numba(
                instance.transient_record, instance.steady_record, instance.upper99_s,
            )
            instance.transient_record = instance.steady_record = instance.upper99_s = None
            instance.state = None # release weights; the next trial always initializes afresh
            print(f"  trial {repeat_idx + 1}/{n_repeat}", flush=True)

        # Write the intended grid coordinates, rather than their float32 device approximations.
        output_bs = b_center + b_spacing * np.arange(-(bs_size // 2), bs_size // 2 + 1)
        rows = np.column_stack((output_bs, np.full(bs_size, eta), np.full(bs_size, -1), capacities))
        geometry_folder = folder / f"nd{nd}_ns{ns}"
        geometry_folder.mkdir(parents=True, exist_ok=True)
        result_path = geometry_folder / f"beta_select{float(beta_select)!r}.csv"
        with result_path.open("a") as handle:
            np.savetxt(handle, rows, delimiter=",", fmt="%.17g")
        with (folder / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(
                nd=nd, ns=ns, eta=eta, b_center=b_center, beta_select=beta_select,
                seed=seed, result_file=str(result_path.relative_to(folder)),
            )) + "\n")


if __name__ == "__main__":
    main()
