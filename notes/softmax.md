# Softmax learning rules and simulations

[Notes index](README.md) · [Architecture](architecture.md) · [Simulation workflow](simulation_workflow.md)

This is the maintained specification and workflow for the Softmax experiments.
The original proposal is archived in [tmp/angular_plasticity_implementation.md](tmp/angular_plasticity_implementation.md).

## Model and parameter names

Each of `nd` dendrites has `ns` synapses, with a unit-length weight vector.
For a pattern, compute every local activation from the weights before the update:
`u_r = dot(w_r, x_r)`. Readout is the standard sum of `max(u_r - b, 0)`.

The selection preference is named `beta_select >= 0`; it is unrelated to the
historical weight-decay parameter `beta`. The learning strength is `eta > 0`.
There is no decay parameter in the model configuration. The CSV decay slot must
contain `-1` and is never passed to the learning rule.

The implemented rules in [lib/neurons.py](../lib/neurons.py) are:

- `Neuron_Softmax_Int`: let `p = softmax(beta_select * u)`, then
  `delta_theta_r = eta / sqrt(ns) * p_r`.
- `Neuron_Softmax_NonInt`:
  `delta_theta_r = eta / (nd * sqrt(ns)) * exp(beta_select * u_r - beta_select**2 / 2)`.

The non-interacting rule uses local exponential allocation; it has no softmax
denominator. “Softmax” names the experiment family. At `beta_select=0`, both
rules assign the same angle to every dendrite.

Initialize independent Gaussian weights and normalize each dendrite to unit
length. Define the tangent direction as the normalized vector
`x_r - u_r * w_r`, and update with
`w_new = w * cos(delta_theta) + tangent_direction * sin(delta_theta)`.
The angles are used literally, without clipping. Zero tangents leave weights
unchanged. Numerically parallel inputs (tangent norm no greater than
`8 * machine_epsilon * ||x||`) also leave weights unchanged; this avoids amplifying
projection roundoff. A final normalization prevents accumulated norm drift.
With `ns=1`, the tangent space has dimension zero and learning leaves weights
unchanged.

`HyperConfigSoftmax` holds `nd, ns`; `HyperDynConfigSoftmax` holds
`b, eta, beta_select`. `State1` is reused. The correlated dynamic configuration
additionally carries the connectivity array.

## Input models and compiled driver reuse

Independent inputs use `Normal`. Shared-axon inputs use
`Normal_Corr_across_Dendrites`, with `int(ns * coeff)` axons and `coeff >= 1`.
Each dendrite selects `ns` distinct axons; axons may be shared across dendrites.
The connectivity stays fixed for every phase of a trial. At `coeff=1`, all
dendrites receive the same signals, possibly in different synapse order.

Learning is independent of b. The runner evolves one state per eta/beta_select
configuration and evaluates all b thresholds together using `Driver1_bScan`.
The existing `Driver1_parallel_states` also supports these neurons when all
dynamic scalar fields are batched; it is not needed for a threshold-only scan.

For independently restarted correlated trials,
`DriverSoftmax_bScan_Correlated` inherits the same phase implementation and passes
the dynamic connectivity to the generator. The base driver's input hook defaults
to the existing input behavior. The generator's optional connectivity argument
preserves its original fixed-map behavior for all old callers.

The runner caches one driver per geometry during a sweep. Seeds, eta,
beta_select, thresholds, and connectivity are dynamic arrays. Independent
restarts therefore reuse compiled phases with unchanged shapes and dtypes.
There is no periodic `jax.clear_caches()`. Changing geometry, phase lengths,
sampling intervals, scan size, input condition, or model can require compilation.
The shared driver's washout phase now has the same JIT/static-length wrapper
as the other phases, preventing repeated construction of its compiled scan.

## Independent trials and phase schedule

Every repeat resets the state, draws fresh weights and input streams, and runs
the full warm-up. Correlated trials also draw fresh connectivity. Keys depend
on the row seed, nd, ns, and repeat index, with separate simulation and connectivity
streams. For the same seed and geometry, randomness is matched across eta,
beta_select, b, and the two learning rules, independently of row order.
Different repeats have distinct streams; comparisons across parameter settings
are intentionally paired.

The default schedule matches the existing runners:

| Phase | Default |
| --- | --- |
| Warm-up, every repeat | 10,000 updates |
| Probe learning and transient recording | 3,000 updates, 1,000 probes |
| Washout | 20,000 updates |
| Steady-state recording | 10,000 updates, sampled every 50 |
| Input block size | 1,000 |
| Repeats | 6 |
| Readout scan | 9 thresholds, spacing 0.15 |

These lengths are configurable; they are not a guarantee of equilibration for
every eta or beta_select. The existing probe-age alignment, steady-state threshold,
and capacity estimator are unchanged. In particular, a capacity of zero may also
mean the recall window was too short to observe a downward crossing.

The old `run.py` and `run_parallel_states.py` retain their continuing-trial
behavior: only the first repeat per row warms up. Their restart policy is not
changed by these additions.

## Visualizing one trial

Open [test_softmax.ipynb](../test_softmax.ipynb), select the `jax-cuda12` environment,
and edit the first code cell: nd, ns, b, eta, beta_select, rule, coeff, seed,
trial index, and phase lengths. Run All to initialize one trial, warm up, and
plot neural response versus pattern age. The example parameters are not optimized.

The plot shows the familiar mean and 1–99% response range across probe patterns,
the pooled steady-state threshold, and capacity when a downward crossing occurs.
It uses the capacity estimator's order statistics and reports no-crossing cases
explicitly. The band is probe variability within a trial, not uncertainty across
independent trials. Age zero is immediately after learning.

Rerunning the trial cell resets weights and connectivity reproducibly, while
reusing the driver when its configuration is unchanged. Change seed or trial_index
for a different trial. Plot controls can be changed without another simulation.
The notebook leaves grid-search files untouched; figure export is optional.

## Preparing and running a search

Edit the constants in [run_softmax.py](../b_bk_grid_search/run_softmax.py).
Set `rule` to `"int"` or `"nonint"`, and `coeff` to `None` for independent
inputs or a finite value >= 1 for shared axons. Choose initial eta and beta_select
values yourself. Each runner row fixes the actual eta and beta_select and specifies
a b scan center. When preparing from target n, the eta column in tmp.csv is a
reference value: actual eta = eta_reference * n_eff, with n_eff equal to the
requested target n. The runner, raw data, summaries, and single-trial notebook
all use actual eta; they do not multiply it again.

The dedicated input path defaults to `tmp_data/params_list_softmax.csv`.
It is headerless, with seven columns:

```text
nd, ns, eta, b_center, beta_select, -1, seed
```

This retains the old column positions, substituting eta for bk and beta_select
for n. The old working parameter table is not modified. Fractional beta_select
and single-row tables are supported. The whole table is validated before running.

Run from any directory:

```bash
python b_bk_grid_search/run_softmax.py
```

Default outputs are:

```text
tmp_data/softmax/{int|nonint}/coeff_{infty|value}/nd{nd}_ns{ns}/beta_select{value}.csv
```

Finite coefficient and beta_select labels use Python float representations
(e.g. `coeff_2.0`, `beta_select0.375.csv`). Rows contain:

```text
b, eta, -1, capacity_0, ..., capacity_5
```

CSV values use 17 significant digits to preserve parameter distinctions. JAX
dynamic scalar parameters and b scans use float32, so grids finer than float32
resolution cannot resolve distinct simulated settings. Weight/input precision
follows JAX's default floating-point configuration.

`settings.json` records the trial protocol and prevents appending incompatible
settings. `runs.jsonl` records completed parameter rows and seeds.
Rerunning a completed row appends it again; there is no automatic resume or
deduplication. If its seed is unchanged, those are duplicate trials, not new
independent samples. Run one writer per output directory. Results remain in
`tmp_data/`; nothing is automatically promoted to `data/`.

## Exporting maximum points only

Use [export_softmax_max_points.py](../common_data_processing/export_softmax_max_points.py)
to generate a maximum-point summary from one Softmax experiment folder:

```bash
python common_data_processing/export_softmax_max_points.py tmp_data/softmax/int/coeff_infty --output tmp_data/softmax/int/coeff_infty/next_grid/max_points_df.csv
```

The input folder must directly contain `nd*_ns*` subfolders with
`beta_select*.csv` results. Keep different rule variants and correlation
coefficients in separate experiment folders. The script pools repeated parameter
rows and computes mean capacity and sample SEM, with six capacity columns per
raw row by default (override with `--n-trials`).

For each (nd, ns, beta_select), it keeps the points with the largest mean
capacity. Exact ties at a fixed eta retain the largest b; ties across eta values
are retained. All-zero grids are included. The output has columns
`nd,ns,beta_select,b,eta,beta,capacity,sem,n_samples`, with actual eta and
decay beta=-1. No calibration registry or grid spacing is required.
The output parent directory is created if needed; an existing summary is
overwritten after input validation. This summary can be passed directly to
`initialize_softmax_grid_search.py`.

## Initializing a new grid from existing Softmax results

Use [initialize_softmax_grid_search.py](../common_data_processing/initialize_softmax_grid_search.py)
to center a fresh experiment on the best existing (b, eta) points. It accepts a
raw Softmax experiment directory or a summary such as tmp_df.csv/max_points_df.csv.
The summaries contain actual eta. The script finds target n_eff in the persistent
calibration registry and converts eta back to reference units before planning.

Preview a new grid from the current summary:

```bash
python common_data_processing/initialize_softmax_grid_search.py tmp_data/max_points_df.csv --dry-run
```

To write both the reference table and a runner table that preserves the exact
source beta_select values:

```bash
python common_data_processing/initialize_softmax_grid_search.py tmp_data/max_points_df.csv --output tmp_data/initial_softmax/tmp.csv --params-output tmp_data/initial_softmax/params_list_softmax.csv --seed 123
```

The default output is tmp_data/tmp.csv, with headerless rows
b_center, eta_reference, n_eff, nd, ns. The companion tmp.report.json records
source optima, registry sources, rounding, actual eta values, and scan ranges.
With --params-output, the additional table has runner rows
nd, ns, eta_actual, b_center, beta_select, -1, seed. Point the runner at this
file directly; no recalibration or second eta scaling is needed. Alternatively,
the reference table can be passed through calibrate_softmax.py; use the original
calibration settings to reproduce its beta values.

Defaults match the original initializer: five eta rows, reference spacing 0.25,
nine b values per scan at spacing 0.15, and six capacity columns in raw files.
Use --eta-size, --eta-reference-spacing, --b-size, --b-spacing, and --n-trials
to match your intended search. --eta-size 1 exports only a center row.
The chosen b center is clamped to be nonnegative, and both axes are rounded
onto new zero-anchored lattices using ties-to-even. Nonpositive eta candidates
are omitted, so there can be fewer than five rows near the lower eta limit.
The minimum eta-reference center is one spacing step. Individual b values in
a scan can still be negative near b_center=0.

Selection is independent for each (nd, ns, beta_select). Maximum-capacity ties
prefer larger b, then smaller eta. All-zero grids warn and are skipped. This
initializes a new experiment, so existing source points are not removed from
the proposed grid. It does not extrapolate to new geometries or compare boundary
conditions as the extension planner does. Empty plans clear stale output tables.

Registry discovery works as in the extension planner; --registry and repeated
--calibration-report options provide explicit sources. Summaries require nd,
beta_select, b, eta, capacity, and ns; use --total-synapses only when ns is absent.
If the optional decay beta column is present it must be -1. Source results and
calibration history are protected from being used as output paths.

## Extending the grid

Use [plan_softmax_grid_search.py](../common_data_processing/plan_softmax_grid_search.py)
on one rule/input condition:

```bash
python common_data_processing/plan_softmax_grid_search.py tmp_data/softmax/int/coeff_infty --eta-reference-spacing 0.25 --seed 123 --dry-run
```

Here 0.25 is the reference spacing. The planner reads target n_eff for each
(nd, beta_select) from persistent calibration history and uses actual spacing
0.25 * n_eff separately for each grid. It automatically finds the nearest
softmax_calibrations.json in the data directory or its parents (normally
at tmp_data/softmax_calibrations.json). Use --registry PATH to select a different
registry. Existing --calibration-report commands still work: the planner includes
the report, its linked/sibling registry, and automatically discovered history.
Additional reports may be supplied by repeating --calibration-report.
Every grid must have a consistent target-n mapping; conflicts are rejected.
The measured Monte Carlo effective_n is diagnostic and is not the multiplier.

Alternatively, --eta-spacing specifies one common spacing directly in actual
eta units, preserving the earlier planner interface; omit --calibration-report
in that mode. Choose the seed to match the initial sweep. Set b spacing, scan
size, and repeat count to match the runner.
Remove `--dry-run` to write into `DATA_DIR/next_grid/`:

- `tmp_df.csv`: pooled mean capacity and SEM, explicitly labeled with beta_select and eta.
- `max_points_df.csv`: best representatives using the existing plateau policy.
- `params_list_softmax.csv`: ready-to-run seven-column parameter rows.
- `grid_search_report.json`: settings and proposed extensions.

Proposed runner rows and all summary eta columns contain actual eta. The audit
report records each grid's n_eff and actual spacing when reference mode is used.
Point the runner's `params_file` at the generated table directly, without passing
it through calibration or scaling it again. No Mathematica step is
needed. The planner reuses the original strip/boundary logic, with eta replacing
bk, and supports a linear eta lattice. It never proposes eta <= 0. When the
next lower eta is outside the domain, that edge does not block checking b.
“Complete” is only a boundary-policy result, not proof of global optimality.
Repeated raw rows are pooled with a warning, just as in the existing planner;
check the seed provenance before treating them as independent replicates.

## Optional calibration from target n

The runner always accepts direct beta_select values. To prepare a batch from
target effective dendrite counts, use
[calibrate_softmax.py](../common_data_processing/calibrate_softmax.py) in place of
the Mathematica parameter-preparation step. Create a headerless input table with:

```text
b_center, eta_reference, n, nd, ns
```

The input shape matches the old five-column preparation table, but its second
column is eta_reference. Choose reference values for this model before using a
legacy bk table. Preparation multiplies by the target effective count n from
the same row: eta = eta_reference * n. For example, n=2.5 and reference values
1.00, 1.25 produce actual eta values 2.5, 3.125; reference spacing 0.25 becomes
actual spacing 0.625. This uses the requested target, not its slightly noisy
Monte Carlo estimate, and applies to both learning rules.
Running without arguments reads the repository's `tmp_data/tmp.csv` and writes
`tmp_data/params_list_softmax.csv`, regardless of the working directory:

```bash
python common_data_processing/calibrate_softmax.py
```

Specify another input/output or seeds as needed:

```bash
python common_data_processing/calibrate_softmax.py tmp_data/tmp.csv --output tmp_data/params_list_softmax.csv --seed 123 --simulation-seed 456
```

The generated rows are `nd, ns, eta, b_center, beta_select, -1, seed`, ready for
`run_softmax.py`. Every input row is retained in its original order, including
repeated rows. Calibration is performed once per distinct (nd, n), reusing the
result across all b_center, eta_reference, and ns values. Fractional targets n are accepted.
The whole input is validated and all calibrations finish before outputs are
replaced. The input file cannot be used as either output path.

`--seed` controls Monte Carlo calibration; `--simulation-seed` is written to every
runner row. Both default to zero. The shared simulation seed preserves matched
randomness across parameter settings; the runner derives separate repeat keys.
Changing the calibration seed does not change the simulation seed. Keep both
settings consistent when extending an experiment.

A companion `params_list_softmax.report.json` records the input rows, target-n
mapping, unique calibration results, simulation seed, eta_scaling="target_n",
and the resulting output rows. The input column is explicitly labeled
eta_reference; the output eta column always contains the scaled value. Both output files are
replaced on each successful conversion. The script does not launch simulations.

Calibration history accumulates in softmax_calibrations.json beside the output
CSV; use --registry PATH to choose another location. Each batch merges its new
calibrations with existing history and the previous batch report before replacing
that report. Identical records are deduplicated; different beta values and
calibration settings remain available. Conflicting target_n assignments for the
same (nd, beta_select) stop preparation before any output is replaced. The registry
is written atomically; run one calibration writer per registry at a time.
The batch report links to its registry but continues to describe only its own
input/output rows and calibrations. Keep the registry when replacing working
parameter tables. Single-pair JSON mode does not update it automatically.

For inspecting one target without preparing a table, the original JSON mode is
still available (do not combine it with a batch input):

```bash
python common_data_processing/calibrate_softmax.py --nd 100 --n 5 --samples 20000 --seed 123 --output tmp_data/softmax_calibration/nd100_n5.json
```

It solves `E[sum(p_r**2)] = 1/n` for independent standard Gaussian activations,
using stable softmax and bisection with common Monte Carlo samples at every
candidate beta. The same seed regenerates batches, keeping memory bounded.
The reported effective count is `1 / E[sum(p_r**2)]`, not `E[1/sum(p_r**2)]`.

For n=nd it returns beta_select=0. For n=1 and nd>1 it returns a finite
approximation satisfying the empirical `E[sum(p_r**2)] >= 1-rtol`.
The output records tolerance, sample size, seed, measured IPR, and an independent
validation estimate with standard error. The root tolerance is not a guarantee
of Monte Carlo accuracy; increase sample size to improve calibration accuracy.
A failed bracket raises an error rather than silently returning an endpoint.

Batch mode fills beta_select in the runner table automatically; single-pair mode
requires copying the value yourself. Retain the JSON as provenance. Hold
beta_select fixed across rules, ns, eta, and correlation conditions for the chosen
nd and target n.

## Validation

```bash
python -B -m unittest tests.test_softmax tests.test_variable_slope tests.test_plan_grid_search tests.test_initialize_grid_search tests.test_initialize_softmax_grid_search -v
```

The focused tests compare one-step updates to a NumPy reference, test norm
preservation and degenerate inputs, compare scalar and parallel recordings, and
check that independently restarted trials reuse traced phases even with changed
connectivity. Additional tests cover the runner's CSV/provenance behavior,
calibration, and positive-eta grid extensions.
