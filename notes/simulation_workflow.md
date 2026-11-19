# Simulation workflow

[Notes index](README.md) · [Architecture](architecture.md) · [Runner and data inventory](experiments_and_data.md) · [Caveats](caveats.md)

## Environment and launch conventions

There is no pinned environment, package build configuration, or automated test suite in the inspected tracked files. The simulation imports require Python, JAX/jaxlib, Flax, NumPy, and Numba. Analysis and plotting additionally use pandas, SciPy, Matplotlib, and sometimes seaborn; interactive notebooks need a suitable Jupyter kernel. The [root README](../README.md) lists a baseline installation command. Historical package versions and accelerator configuration are not recorded, so that list is not a verified reproducible environment.

At review time, the shell's `python` had JAX, jaxlib, Flax, NumPy, and SciPy installed, but package metadata for pandas, Matplotlib, Numba, ipykernel, and seaborn was absent. This describes the inspected interpreter, not necessarily the user's notebook environment. In particular, importing `lib.utils` requires Numba.

Simulation runners and most notebooks use paths relative to the current working directory and insert `..` into `sys.path`. Launch a runner from its own directory after preparing the matching input table and checking its output location. For example, the following is the launch convention for the parallel-state runner, not an instruction to start the existing queued sweep without review:

```bash
cd b_bk_grid_search
python run_parallel_states.py
```

The current standalone scripts in `plot_scripts/fig2`, `fig3`, `fig4`, and `cosyne2027_plots` resolve paths from `__file__` and can be invoked from the repository root. The older `fig4_old` scripts retain relative-path assumptions.

## Driver lifecycle

A standard simulation is assembled from `HyperConfig1`, `HyperDynConfig1`, a neuron class, an input generator, and a driver. Constructing the driver does not run the experiment. Calling `run(...)` updates its state and populates recordings; it does not return the capacity. Call `utils.capacity_numba(driver.transient_record, driver.steady_record, driver.upper99_s)` afterward.

| Phase | What happens | Common runner setting |
| --- | --- | --- |
| Initialization | If `state is None`, create random weights and learn fresh inputs without recording. | 10,000 patterns. |
| Phase 1 | Generate `n_tested_patterns` probe patterns; present each once at the beginning of the learning stream, then continue with new patterns. After every update, evaluate all probes. | 3,000 updates; 1,000 probes. |
| Phase 2 | Continue learning unrelated inputs without recording, to wash out probe familiarity. | 20,000 updates. |
| Phase 3 | Continue learning unrelated inputs and periodically test those same old probes as a steady-state reference. | 10,000 updates, sampled every 50 updates. |

The steady-state distribution comes from the original probe patterns after further learning; the driver does not replace them with a newly sampled unfamiliar test set in phase 3. Whether the washout duration suffices is an experimental assumption.

After phase 1, probe `i` is shifted by `-i` along time so rows correspond to age since that probe was learned. Age zero is the response immediately after its learning update. The driver conservatively keeps `n_steps_phase1 - n_tested_patterns` rows. With the usual settings and 13 readout thresholds, `transient_record.shape == (2000, 13, 1000)`, `steady_record.shape == (200, 13, 1000)`, and `upper99_s.shape == (13,)`.

The middle axis means readout thresholds for `Driver1_bScan`, parameter states for `Driver1_parallel_states`, or noise levels for noise-scan drivers. It is not the repeat axis. Repeats are performed by an outer Python loop.

A second `run` call retains the weights and advances the RNG. It skips initialization even if `n_init_steps` is supplied again. `re_init(rng=..., hyperD=..., initial_state=None)` resets the weights to be initialized on the next run and clears recorded results, but retains the existing neuron, generator, and constructor settings. The usual six repeats are consecutive trials along a continuing weight trajectory. The variable-synapse runner is an exception: it reinitializes every repeat with a different size mask and the same row-level simulation key.

## Choosing the correct scan driver

`Driver1_bScan` evolves one weight trajectory and evaluates it at multiple values in `bs`. This is appropriate when learning is independent of readout threshold, as in the standard n-WTA or non-interacting fixed-target rule at fixed `bk` and `beta`. It can evaluate a single fixed configuration of the variable-target rule, but scanning `bs` does not recompute that rule's learning target for each threshold.

`Driver1_parallel_states` evolves one state for each entry in the dynamic hyperparameter arrays. All fields of `HyperDynConfig1` need matching leading dimensions: `b, bk, beta` are vectors, even when some values are repeated. The states receive the same input stream and the same initialization key, allowing comparisons with shared randomness. This is the appropriate existing driver when changes to `b` or `beta` also change learning. Its `bs` argument is unused; runners pass `None`.

`Driver2_parallel_states` handles the variable-synapse model using the same general trial protocol. Its dynamic geometry arrays also have a leading state dimension, with geometry shared within the scan.

## Capacity calculation

[capacity_numba](../lib/utils.py) compares the lower tail of the familiar response to the upper tail of the steady-state response. For each scan column, it takes the order statistic at index `ceil(0.01*n_patterns)-1` across probes at every age. It compares this with the steady-state order statistic at `ceil(0.99*n_samples)-1`, pooled over retained times and probes, plus `1e-5`. Passing `upper99_s` uses the supplied thresholds directly, avoiding that recomputation.

The estimator finds every transition from a nonnegative difference to a negative difference, records the previous age index `t-1`, and returns the mean of those indices. It does not interpolate between ages, and it does not restrict itself to the first crossing. Thus capacities can be noninteger even though time is discrete.

If no downward crossing occurs, the return value is zero. That can mean responses were already below threshold, but it can also mean they stayed above threshold throughout the recorded window. Inspect response curves before treating all zero results as failed memory. With the usual phase-1 settings, there are only 2,000 retained age samples; a longer-lived memory needs a longer recording window.

The estimator uses order statistics rather than interpolated percentiles. Some plotting notebooks use NumPy percentile functions for visualization, which can produce slightly different displayed quantiles.

## Starting a new grid from an existing run

Use [initialize_grid_search.py](../common_data_processing/initialize_grid_search.py) to seed a new experiment, such as variable-slope simulations, around the best `(b, bk)` points of an earlier run. It selects one maximum mean-capacity point per `(nd, ns, n)` and generates a fresh grid around it. Use [plan_grid_search.py](../common_data_processing/plan_grid_search.py) afterward to extend the new experiment's grid based on its own results.

Run these commands from the repository root. First preview the selected points and scan ranges without writing files, then generate the table:

```bash
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty --dry-run
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty
```

The source directory must contain raw results at `nd{nd}_ns{ns}/n{n}.csv`. Alternatively, supply an aggregated `tmp_df.csv` or `max_points_df.csv`. Legacy summaries lack `ns`, so provide the total number of synapses explicitly; the script derives `ns = total_synapses / nd` and requires exact divisibility:

```bash
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty/max_points_df.csv --total-synapses 60000 --dry-run
```

Remove `--dry-run` to write the summary-based plan. Only configurations present in the source receive grids; the script does not extrapolate to new geometries. Summary columns must include `nd`, `n` (or `ndR`), `b`, `bk`, and `capacity`, with either `ns` or the command-line total synapse count.

The default output is the repository's `tmp_data/tmp.csv`, with no header and columns `b_center, bk, n, nd, ns`. Each invocation that writes files replaces this table and writes a sibling `tmp.report.json` recording source optima, settings, proposed scans, and skipped grids. An empty plan replaces stale candidates with an empty table. Use `--output tmp_data/initial_variableSlope/tmp.csv` to choose another location, and point Mathematica at that table when preparing parameters. The source results are left unchanged.

Defaults and options should match the intended runner:

| Option | Default | Effect |
| --- | --- | --- |
| `--bk-size` | `5` | Number of fixed-bk scan rows per configuration; use `1` for just the center. |
| `--bk-spacing` | `0.25` | Spacing between bk rows and rounding lattice for the bk center. |
| `--b-spacing` | `0.15` | Rounding lattice for the b center and assumed spacing within each runner scan. |
| `--b-size` | `9` | Number of b values assumed when reporting the scan range; does not change the number of output rows or configure the runner. |
| `--n-trials` | `6` | Expected capacity columns per raw-result row. |

With defaults, each selected point produces five bk rows at offsets `-0.50, -0.25, 0, 0.25, 0.50` from the rounded bk center. All five share a b center, which the runner expands into nine b values. Both scan sizes must be positive odd integers.

Exact capacity ties prefer the largest b, including plateaus at fixed bk; ties in both capacity and b prefer the smaller bk. All-zero grids warn and are skipped, while `n=1` is included. The selected b center is clipped to be nonnegative, then rounded to the nearest multiple of b spacing; bk is similarly rounded to its spacing. Rounding uses ties-to-even on a lattice anchored at zero, so an optimum from an offset source grid may move. Individual b values can still be negative near a zero center. The report preserves both the source optimum and the resulting scan centers.

The output is input for the Mathematica preparation step below, not a runner parameter table: it contains neither beta nor seeds. For standard n-WTA and variable-slope runs, use [weight_decay_gs_calculator.nb](../lib/Mathematica_notebooks/weight_decay_gs_calculator.nb), verify its input/output paths, and prepare the seven-column runner table `nd, ns, bk, b_center, n, beta, seed`. The notebook exports `tmp_data/params_list_tmp2.csv`; inspect it before placing it at the runner's expected `tmp_data/params_list_tmp.csv`. Configure and launch the [variable-slope runner](variable_slope.md) or other matching runner, keeping new simulation outputs in `tmp_data/` until manually promoting completed results to `data/`.

## Preparing and extending grid searches

1. Choose the rule, generator, noise model, geometry, and whether thresholds share a trajectory. Many standard scans keep `nd*ns = 60000`, using explicit matched `ndlist` and `nslist` arrays rather than a universally enforced constraint.
2. Prepare grid-center rows in the order `b_center, bk, n, nd, ns`. For a new experiment, use the [initialization script described above](#starting-a-new-grid-from-an-existing-run). For boundary extensions of an existing experiment, use [plan_grid_search.py](../common_data_processing/README.md#extending-an-existing-experiment). The processing notebook also retains these export branches.
3. Supply beta values appropriate to the chosen model. The Mathematica notebooks [weight_decay_gs_calculator.nb](../lib/Mathematica_notebooks/weight_decay_gs_calculator.nb) and [weight_decay_gs_calculator_variable_n.nb](../lib/Mathematica_notebooks/weight_decay_gs_calculator_variable_n.nb) define analytical moment calculations and root solvers. They import `tmp_data/tmp.csv` and export `tmp_data/params_list_tmp2.csv` through hard-coded absolute paths. Their source contains alternative parameter-generation definitions; verify the active definition and resulting column order.
4. Inspect the generated table before selecting it as `tmp_data/params_list_tmp.csv`, the filename all inspected runners read. No automatic handoff from `params_list_tmp2.csv` to that filename was found. In parallel scans, the number and order of beta values must match the generated `bs` vector.
5. Run the matching script from its directory. Its constants determine repetitions, phase lengths, model choice, and output path; these are not command-line options.
6. Aggregate raw outputs with [create_gridSearch_df.ipynb](../common_data_processing/create_gridSearch_df.ipynb), choosing the correct experiment root. Create the intermediate output directory if it does not exist.
7. Inspect [process_gridSearch_df.ipynb](../common_data_processing/process_gridSearch_df.ipynb) to select maxima, check grid alignment and edges, and generate the next search batch if needed. Use the relevant cells for the intended branch: several cells overwrite the same candidate-table variable.
8. Retain the raw results and final `max_points_df.csv` under an experiment-specific data location before pointing plotting scripts at them. The inspected code does not implement an automatic publishing or provenance stage.

The general processing notebook also contains a beta-free export branch with rows `nd, ns, bk, b_center, n, seed, -1`. This matches the variable-synapse runner's willingness to ignore supplied betas; it is not the seven-column standard schema, where beta precedes seed. A beta placeholder of `-1` must not be used as ordinary decay for the standard neuron.

## Practical constraints

Use positive phase lengths that are multiples of `resample_every`; implementations use integer division and can silently omit remainders. Keep `n_steps_phase1 > n_tested_patterns`. Existing callers normally use `n_tested_patterns = resample_every = 1000`. Probe padding exists, but staying with block-aligned sizes avoids depending on less-used paths.

Phase-3 sampling restarts its local index at each resampling block. Choose `test_every_phase3` to divide `resample_every` if uniformly spaced sampling across the whole phase is intended. Unrecorded rows are represented internally by `-100` and removed afterward.

Recordings and noise arrays can be large: before filtering, the standard 10,000-step phase-3 output with 13 thresholds and 1,000 probes has 130 million elements. A float32 version alone is about 520 MB, excluding other arrays and copies. Sparse recording reduces retained output but does not eliminate the full sentinel-filled scan output. Small pilot configurations should be checked before launching a large new sweep.
