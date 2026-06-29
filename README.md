# Dendrite Familiarity Detection

Simulation and plotting code for studying familiarity detection in dendritic neuron models. The repository contains JAX/Flax implementations of neuron update rules, input generators, grid-search scripts, data-processing notebooks, and scripts used to generate figure panels.

## Repository Layout

- `lib/` - Core simulation code.
  - `neurons.py` defines neuron states, hyperparameter dataclasses, and update rules.
  - `input_gen.py` defines normal, correlated, sparse binomial, and variable-synapse input generators.
  - `driver.py` runs transient and steady-state recall simulations.
  - `utils.py` contains capacity-estimation utilities.
- `b_bk_grid_search*/` - Grid-search entry points for threshold and target-activation scans.
- `no_nonlinearit_bk_grid_search/` - Grid-search scripts for no-nonlinearity controls.
- `common_data_processing/` - Notebooks for aggregating raw grid-search outputs into summary data frames.
- `plot_scripts/` - Figure-generation scripts grouped by figure.
- `plot_notebooks/` - Exploratory plotting notebooks.
- `data/` - Processed or curated data used by plotting scripts.
- `tmp_data/` - Generated intermediate grid-search outputs.
- `figures/` and `tmp_figures/` - Figure outputs.

## Setup

This project does not currently include a pinned environment file. A minimal Python environment should include:

```bash
pip install jax jaxlib flax numpy scipy pandas matplotlib numba ipykernel
```

For GPU or TPU runs, install the `jax`/`jaxlib` build that matches the target accelerator.

## Typical Workflow

Run scripts from their own directories so their relative paths resolve correctly:

```bash
cd b_bk_grid_search
python run.py
```

Most grid-search scripts read parameter rows from `../tmp_data/params_list_tmp.csv` and append results under `../tmp_data/`. Each row specifies simulation settings such as `nd`, `ns`, `bk`, threshold center, selected dendrite count `n`, decay `beta`, and random seed.

After raw simulations finish, use the notebooks in `common_data_processing/` to build processed CSV summaries. Plotting scripts in `plot_scripts/` then read from `data/` or `tmp_data/` and write figure files next to the script being run. For example:

```bash
cd plot_scripts/fig4
python fig4a.py
```

## Development Notes

- JAX compilation depends on static simulation dimensions. The grid-search scripts reuse drivers when possible and recompile when dimensions such as `nd` or `n` change.
- Many runners assume `n_init_steps`, `n_steps_phase1`, `n_steps_phase2`, and `n_steps_phase3` are multiples of `resample_every`.
- Capacity is computed from transient and steady-state response records using `lib.utils.capacity_numba`.
- Generated CSVs can become large; keep reproducible inputs and processed summaries in `data/`, and use `tmp_data/` for scratch or rerunnable outputs.

