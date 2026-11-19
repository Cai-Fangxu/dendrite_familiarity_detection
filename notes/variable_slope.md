# Variable-slope simulations

[Notes index](README.md) · [Runner](../b_bk_grid_search/run_variableSlope.py)

The variable-slope experiment uses the standard n-WTA update with equal numbers of synapses per dendrite and a weighted ReLU readout: $y=\sum_r s_r\max(\mathbf{w}_r\cdot\mathbf{x}_r-b,0)$. Each dendrite's slope is sampled uniformly from the configured range and held fixed throughout a trial. Slopes are not normalized to have sample mean one. Selection still ranks raw overlaps, and selected dendrites still target overlap `bk`; neither selection nor the weight update compensates for slopes.

## Implementation and efficiency

`HyperDynConfig1_VariableSlope` in [neurons.py](../lib/neurons.py) adds a dynamic `slopes` vector to the existing runtime configuration. No neuron learning or forward implementation is changed. `Driver1_bScan_VariableSlope` in [driver.py](../lib/driver.py) overrides the two recording methods and uses a shared weighted-readout helper. Direct `neuron.forward` calls retain the ordinary unweighted readout.

The driver inherits initialization, phase scheduling, blockwise input generation, JIT compilation, sequential `lax.scan`, transient age alignment, and steady-state quantiles. It evolves one weight trajectory per trial, computes probe overlaps once per recorded step, and broadcasts over b and slopes. Phase 3 evaluates probes only on sampled steps through `lax.cond`. Slopes are dynamic array leaves rather than mutable attributes on the JIT-static driver object; changing their values with unchanged shape/dtype does not require retracing.

The standard n-WTA beta calculation remains applicable because the weight dynamics are unchanged. Readout-only b scans require common bk and beta and a learning rule independent of b. The variable-target learning rules still need a separate-state approach when scanning b.

## Configuring and running

Edit the constants and commented model/input/driver alternatives in [run_variableSlope.py](../b_bk_grid_search/run_variableSlope.py). Defaults are n-WTA, independent Gaussian input, slopes in `[0.5, 1.5]`, six restarted trials, and nine b values at spacing 0.15. The phase lengths match the existing runners: 10,000 initialization steps, 3,000 phase-1 steps, 20,000 washout steps, and 10,000 phase-3 steps sampled every 50; 1,000 probes and input blocks of 1,000.

The parameter file is `tmp_data/params_list_tmp.csv`, with headerless rows `nd, ns, bk, bias_center, n, beta, seed`. Single-row files are supported. The currently inspected working table has the parallel-state format, so prepare a standard seven-column table before launching this runner. The script validates the whole parameter table before simulating.

Run from the repository root or from the runner directory:

```bash
python b_bk_grid_search/run_variableSlope.py
```

Paths resolve from the script rather than the working directory. Importing the runner does not start a sweep. As with other simulation runners, a full run requires JAX, Flax, NumPy, and Numba for capacity estimation.

## Trial randomness and reuse

Slope samples are keyed solely by `slope_seed`, `nd`, and the zero-based repeat index. For the same nd and repeat they are shared across n, ns, b, bk, and row-level simulation seeds. Different repeats draw different slopes. Keeping the slope seed and range fixed reproduces these samples when rows are reordered or the grid is extended.

Every parameter-row/repeat starts from a newly initialized state and runs the initialization phase. Simulation keys depend on the row seed, nd, ns, n, and repeat index. Trials have distinct random streams; rows with identical seed/geometry/n share random numbers across different b/bk/beta settings while restarting independently. Input and slope RNG streams are separate. The optional correlated-input lines use a reproducible connectivity key; default inputs are independent.

The runner reuses a driver when `(nd, ns, n)` is unchanged and supplies fresh dynamic hyperparameters through `re_init`. Include ns in this condition to avoid the shape-reuse issue in older runners.

## Output and subsequent planning

Default output is `tmp_data/int_variableSlope/slope_0.5_1.5/nd{nd}_ns{ns}/n{n}.csv`. Rows append in the existing format `b, bk, beta, capacity_0, ..., capacity_5`, with the same numeric precision as the other runners. The runner does not move results into `data/`.

Keep the slope seed, model/input choice, and simulation settings consistent within an output directory. If changing these conditions, also choose a distinct output folder; the CSV schema and default folder name do not encode all settings. Rerunning a completed parameter row appends another set of results.

The new planner can consume this experiment root directly:

```bash
python common_data_processing/plan_grid_search.py tmp_data/int_variableSlope/slope_0.5_1.5 --b-size 9 --dry-run
```

## Validation

Run the focused tests from the repository root with `python -B -m unittest tests.test_variable_slope -v`. They check all-one equivalence to the original driver through all phases, explicit weighted response calculations, sparse recording, unchanged learning under nonuniform slopes, dynamic JIT arguments, reproducible trial sampling, parameter validation, and runner restart/output behavior. The runner orchestration test substitutes a recording driver and capacity function; the numerical tests exercise the real JAX driver separately. Production capacity estimation still uses the unchanged Numba implementation in `lib/utils.py`.
