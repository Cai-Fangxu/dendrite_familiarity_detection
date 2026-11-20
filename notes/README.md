# Codebase reference notes

These notes describe the repository as inspected on 2026-11-18. They are based on the Python source, notebook source cells, representative local data files, Mathematica notebook definitions and export paths, and [the manuscript](main.tex). This was a documentation review: no simulations, notebooks, Mathematica calculations, or figure generation were executed, and no source code was changed.

## Reading order

| Document | What it answers |
| --- | --- |
| [Architecture and model](architecture.md) | Where is the implementation, how do the mathematical quantities map to code, and which classes implement each learning rule? |
| [Simulation workflow](simulation_workflow.md) | How do drivers work, how is capacity measured, and how are parameter sweeps prepared and run? |
| [Experiments and data](experiments_and_data.md) | What do the existing runners actually run, what do the CSVs contain, and how do results become plots? |
| [Caveats and restart checklist](caveats.md) | What needs attention before adding simulations or interpreting existing results? |

## Subsequent additions

[Initial grid generation](simulation_workflow.md#starting-a-new-grid-from-an-existing-run) describes how to seed a new experiment from an existing run using `initialize_grid_search.py`, before extending it with `plan_grid_search.py`.

[Variable-slope simulations](variable_slope.md) documents the new weighted-ReLU driver and runner, including configuration, shared slope sampling, restarted trials, and output paths.

## Repository at a glance

| Location | Role |
| --- | --- |
| [lib/](../lib/) | Reusable input generators, neuron models, recall drivers, and capacity estimator. |
| [lib/Mathematica_notebooks/](../lib/Mathematica_notebooks/) | Analytical calculations and numerical root finding for weight-decay parameters; exports used by Python sweeps. |
| [b_bk_grid_search/](../b_bk_grid_search/) | Main sweeps over dendritic threshold `b` and target activation `bk`, including parallel states, variable synapse counts, and step activation. |
| [b_bk_grid_search_noisy/](../b_bk_grid_search_noisy/) | Recall-noise sweeps and threshold optimization with noise. |
| [b_bk_grid_search_sparse_binomial/](../b_bk_grid_search_sparse_binomial/) | Sparse input experiments and associated analysis notebooks. |
| [no_nonlinearit_bk_grid_search/](../no_nonlinearit_bk_grid_search/) | Linear-readout control; the spelling is the actual directory name. |
| [common_data_processing/](../common_data_processing/) | Aggregate raw capacities, identify optimal grid points, inspect grid boundaries, and prepare further parameter searches. |
| [plot_notebooks/](../plot_notebooks/) | Exploratory analysis and generation of fixed-parameter simulation data as well as plotting. |
| [plot_scripts/](../plot_scripts/) | Standalone figure panels, mostly loading existing data; `fig4_old/` contains earlier scripts. |
| [cosyne2027_plots/](../cosyne2027_plots/) | Manuscript panels d/e/f and locally available figure assembly assets. |
| [data/](../data/) | Curated experiment results and fixed-parameter arrays used by plots; ignored by Git. |
| [tmp_data/](../tmp_data/) | Working parameter tables, append-only raw outputs, and intermediate summaries; ignored by Git. |
| [figures/](../figures/), [tmp_figures/](../tmp_figures/) | Existing figure collections and scratch figures; current standalone plotting scripts generally save next to themselves. |
| [tmp_code_files/](../tmp_code_files/) | Ignored local scratch analysis, including additional synapse-count scans and conversion utilities. |
| Root `test*.ipynb` | Interactive experiments and examples, not an automated test suite; some refer to retired APIs. |
| [notes/main.tex](main.tex) | Project manuscript. It references equations in a figure rather than including their full definitions in the TeX body. |

The normal research cycle is: choose a model and input distribution → prepare `b/bk` grid centers and beta values → run a Python sweep → aggregate six capacity measurements per grid point → inspect and extend the grid → retain optimized summaries → generate figures. Fixed-parameter comparisons also use a shorter path through simulation cells in `plot_notebooks/` and saved NumPy arrays.

## Quick orientation for future work

The manuscript's three central rules correspond to `Neuron_Decay_NonInt` (non-interacting), `Neuron_Decay_Int` (n-WTA), and `Neuron_Decay_Int_Variable_n_min1` (variable target activation with a fallback winner). `Int` is not a numerical integer dtype designation. In processed CSVs, `ndR` is the historical name for `n`, the number or nominal mean number of selected dendrites.

The current working parameter table is a 15-column parallel-state table, while the main `b_bk_grid_search/run.py` expects seven columns. Read the runner inventory and caveats before using the root README's example command.

The root [README](../README.md) gives a useful short introduction. These notes add implementation details and describe the active configuration of the inspected checkout; they are not a claim that every historical dataset can be regenerated by running the current scripts unchanged.

[Softmax learning rules and simulations](softmax.md) documents unit-norm learning, independent restarts, direct beta_select grids, and optional target-n calibration.

[Softmax single-trial notebook](../test_softmax.ipynb) provides editable parameters and a neural-response-versus-age visualization.
