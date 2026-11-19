# Grid-search planning

[plan_grid_search.py](plan_grid_search.py) replaces the aggregation and boundary-extension portions of `create_gridSearch_df.ipynb` and `process_gridSearch_df.ipynb`. It prepares the next simulation scans without calculating beta or running simulations. It requires NumPy and the Python standard library, with no pandas, SciPy, or JAX dependency.

## Initializing a new experiment

[initialize_grid_search.py](initialize_grid_search.py) implements the notebook's separate initialization branch: choose the best existing (b, bk) for each (nd, ns, n), then center a fresh rectangular search there. It does not perform the extension planner's boundary checks and does not exclude points already sampled in the source experiment, because the destination is a new experiment.

From an existing raw-result directory:

```bash
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty --dry-run
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty
```

Alternatively, use a summary such as `tmp_df.csv` or `max_points_df.csv`. Legacy summaries lack ns, so explicitly supply the total synapse count:

```bash
python common_data_processing/initialize_grid_search.py data/gridSearch_standard/60000/coeff_infty/max_points_df.csv --total-synapses 60000 --dry-run
```

The default output is the repository's `tmp_data/tmp.csv`, which matches the Mathematica notebooks' input path. Use `--output tmp_data/initial_variableSlope/tmp.csv` for a separate working table. A non-dry run replaces the output CSV and writes a sibling `tmp.report.json` containing the selected source points, settings, and proposed rows. An empty result replaces stale candidates with an empty file. Source data are not modified.

Each output row is `b_center, bk, n, nd, ns`, with no header. With defaults, the selected bk is surrounded by five rows at offsets {-0.50, -0.25, 0, 0.25, 0.50}, all sharing the selected b center. The runner later expands each row into nine b values at spacing 0.15. Configure `--bk-size`, `--bk-spacing`, `--b-spacing`, and `--b-size` to match the new experiment; b-size controls the reported scan range rather than the number of output rows. `--bk-size 1` exports just the centers. For raw results, `--n-trials` defaults to six.

Selection uses maximum mean capacity, preferring larger b for exact ties, including plateaus at a fixed bk. If capacity and b both tie across bk values, the smaller bk wins deterministically. All-zero groups warn and are skipped; n=1 is included. SEM does not affect selection. Different ns values are kept as separate geometries. Raw-source loading uses the same validation and pooling of repeated parameter rows as the extension planner; summary rows are treated as already aggregated estimates.

Following the notebook's initialization branch, the selected b center is clipped to be nonnegative and rounded to the nearest multiple of b-spacing; the selected bk is rounded to the nearest multiple of bk-spacing. Rounding uses ties-to-even, like NumPy's round in the notebook. The new lattice is anchored at zero even if the source search used a different offset. The report retains the original optimum as well as the resulting centers. Individual b values in a scan may still be negative when the center is near zero.

A summary needs nd, ndR (or n), b, bk, and capacity. If ns is present it is preserved; otherwise `--total-synapses` is required and must be exactly divisible by each nd. No geometry extrapolation is performed: only configurations present in the source receive initial grids. The generated table contains neither beta nor seeds; continue through the existing Mathematica preparation step before running simulations.

Run the initialization tests with `python -B -m unittest tests.test_initialize_grid_search -v`.

## Extending an existing experiment

Run from the repository root, supplying one experiment directory containing `nd{nd}_ns{ns}/n{n}.csv` files:

```bash
python common_data_processing/plan_grid_search.py tmp_data/int_variable_n_min1 --dry-run
python common_data_processing/plan_grid_search.py tmp_data/int_variable_n_min1 --top-k 4
```

Paths supplied on the command line are relative to the current working directory. The script discovers available geometries and winner counts, including `n=1`, instead of using the notebooks' hard-coded lists. One experiment root must contain only one model/input/noise condition.

Default settings match the nine-point parallel-state runner: `--b-size 9 --b-spacing 0.15 --bk-size 5 --bk-spacing 0.25 --n-trials 6`. For a runner scanning thirteen b values, pass `--b-size 13`. Scan sizes must be positive odd integers. `--top-k` controls how many ranked representatives are checked; its default is four.

## Outputs

The default output directory is `DATA_DIR/next_grid/`. Override it with `--output-dir PATH`. These four generated files are replaced on subsequent runs; raw result files and notebooks are left unchanged. `--dry-run` prints decisions and writes nothing.

| File | Contents |
| --- | --- |
| `tmp_df.csv` | Aggregated mean capacity and sample SEM, with the notebook columns `nd, ndR, b, bk, beta, capacity, sem`, plus `ns, n_samples`. |
| `max_points_df.csv` | Maximum-capacity representatives per geometry and n, after the plateau policy below. Ties at different bk values are retained. |
| `tmp.csv` | Headerless rows `b_center, bk, n, nd, ns`, suitable for the existing Mathematica parameter-preparation step. An empty plan produces an empty file, replacing any stale candidates. |
| `grid_search_report.json` | Settings, source directory, per-grid status, top points, and proposed scans with their trigger, actual b range, and counts of new/already-sampled points. |

The Mathematica notebooks currently read the shared `tmp_data/tmp.csv`. This script deliberately defaults to an experiment-specific output directory; choose the appropriate input table in that preparation workflow. The generated table does not contain beta or seeds and is not a simulation runner's `params_list_tmp.csv`.

## Decision procedure

Each `(nd, ns, n)` grid is handled independently. Raw rows must have exactly `b, bk, beta` followed by the configured number of repeat capacities. Single-row files are supported. Repeated rows at exactly the same `(b, bk, beta)` are pooled before calculating the mean and SEM, with a warning to check that those trials are intended repeats. Different beta values at the same b/bk coordinate are rejected because this planner assumes a two-dimensional search.

An all-zero grid produces a warning and no proposals. Otherwise, equal mean capacities at the same bk are represented by the largest b, before selecting the top K representatives. Equality here is exact, not an SEM-based or approximate tie. This policy applies to equal values even if their sampled b positions are separated. Remaining ranking ties prefer larger b, then smaller bk. The unfiltered samples remain available in `tmp_df.csv`.

The bk check runs first. Each top point is checked against the bk extremes within its b strip, using the full dataset independently for every strip. The rightmost strip is preferred when it contains the point; otherwise the leftmost strip is used when it contains the point. For grids wider than two scan blocks, interior points use a centered strip. Each upper/lower boundary hit requests another row beyond that boundary, with counters maintained separately for each strip and direction. If any bk boundary is hit, no b check is performed for that grid in this iteration.

Only after bk passes does the b check examine global b extremes. The first actionable top point on a b boundary generates an adjacent block of b values, repeated over the configured number of bk rows centered on that point's bk. With the default settings, an upper extension has center `b_max + 0.75`, covers `b_max + 0.15` through `b_max + 1.35`, and uses five bk values separated by 0.25. Lower extensions are symmetric before enforcing the center constraint.

All proposed centers are nonnegative. If necessary, a center is clamped to the smallest nonnegative point on the existing b lattice; this is zero for the usual zero-aligned grids. A nonnegative center can still produce negative individual b values. Existing grid offsets are preserved rather than rounded onto a different lattice.

Identical proposed scans are deduplicated, and fully sampled scans are omitted. A partially sampled scan is retained because the existing runners accept full b blocks; overlap counts are reported. When the lower-center constraint leaves no new points, the status is `b_center_limit` and a warning is emitted instead of repeatedly proposing the same scans.

## Interpretation and validation

`complete` means the selected top K representatives passed these boundary checks, not a proof of a global optimum. The procedure follows the notebook's strip-level bk extremes and global b extremes; it does not fill arbitrary holes in an irregular sampled grid or assess statistical uncertainty in the optimum. Inspect the report if a grid is sparse or unusually shaped.

Misaligned coordinates, malformed schemas, nonfinite values, negative capacities, or conflicting beta values cause an explicit error before outputs are written. The all-zero warning does not distinguish intrinsically low capacity from a too-short recall window; the existing capacity estimator can return zero for either reason.

Run the focused tests from the repository root:

```bash
python -B -m unittest tests.test_plan_grid_search -v
```
