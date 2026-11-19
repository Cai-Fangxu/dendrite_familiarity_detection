# Experiments, data formats, and plotting

[Notes index](README.md) · [Workflow](simulation_workflow.md) · [Caveats](caveats.md)

## Active runner configurations

This table records executable assignments in the inspected source, not commented alternatives or inferred provenance of historical data. All listed scripts read `../tmp_data/params_list_tmp.csv`, use six repeats, and append headerless CSV output. Their common phase lengths are documented in [Simulation workflow](simulation_workflow.md).

| Runner | Active model and input | Scan and destination below `tmp_data/` |
| --- | --- | --- |
| [b_bk_grid_search/run.py](../b_bk_grid_search/run.py) | n-WTA; correlated Gaussian input, `coeff=2`. | 13 thresholds, spacing 0.15; `nd{nd}_ns{ns}/n{n}.csv`. |
| [run_parallel_states.py](../b_bk_grid_search/run_parallel_states.py) | Variable target with minimum one winner; correlated input, `coeff=1`. | Nine thresholds with paired betas; `int_variable_n_min1/nd{nd}_ns{ns}/n{n}.csv`. |
| [run_parallel_states_StepFunNonLinearity.py](../b_bk_grid_search/run_parallel_states_StepFunNonLinearity.py) | Local subclasses of the variable-target/minimum-one model and parallel driver; independent Gaussian input; Heaviside readout. | Nine thresholds with paired betas; same destination as the preceding ReLU runner. |
| [run_parallel_states_variableNs.py](../b_bk_grid_search/run_parallel_states_variableNs.py) | Variable-synapse n-WTA model; independent masked Gaussian input. Counts are drawn uniformly from `int(0.5*ns)` through `int(1.5*ns)`, inclusive. | Nine thresholds; beta forced to `-1` and unused; `int_variableNs/nd{nd}_ns{ns}/n{n}.csv`. |
| [run_get_thetaY.py](../b_bk_grid_search/run_get_thetaY.py) | n-WTA; correlated input, `coeff=2`; dendritic recall noise 0.15. | One threshold; saves six theta-Y values to `coeff_infty/dendrite_noise_0.15.csv`. The folder label disagrees with the active input correlation. |
| [b_bk_grid_search_noisy/run.py](../b_bk_grid_search_noisy/run.py) | n-WTA; combined dendritic and somatic recall noise. Correlation and noise settings depend on row index. | 13 thresholds; condition-specific folders. Rows 0–519 use independent input and dendrite/soma noise 0.25/0.1; 520–1039 use `coeff=4`, 0.2/0.1; 1040–1559 use `coeff=4`, 0.25/0.1; later rows use `coeff=4`, 0.3/0.1. |
| [b_bk_grid_search_noisy/run_noiseScan.py](../b_bk_grid_search_noisy/run_noiseScan.py) | n-WTA; dendritic noise scan at fixed `b`; independent input in the first 520 rows, then `coeff=4`. | 15 noise levels from 0 through 1; later folder names still mention fixed dendrite/soma noise although the active driver scans dendritic noise only. |
| [b_bk_grid_search_sparse_binomial/run.py](../b_bk_grid_search_sparse_binomial/run.py) | n-WTA with synaptic bounds `reset_from=reset_to=3`; sparse input `fA=0.30, mu=4, n_burst=4`. | 13 thresholds; `nd{nd}_ns{ns}/n{n}.csv`, shared with the basic runner. |
| [no_nonlinearit_bk_grid_search/run.py](../no_nonlinearit_bk_grid_search/run.py) | n-WTA learning with linear readout and matching driver; independent Gaussian input. | One readout column; `nd{nd}_ns{ns}/n{n}.csv`, shared with the basic runner. |

The nine-threshold scans span `bias_center ± 0.6`; the 13-threshold scans span `bias_center ± 0.9`. There is no dedicated currently active non-interacting sweep in this inventory: some runners contain commented alternatives, and historical non-interacting results are present in `data/`.

## Parameter and result schemas

| File or runner family | Headerless column order |
| --- | --- |
| Ordinary runners, including noise and theta-Y | `nd, ns, bk, bias_center, n, beta, seed` |
| Parallel-state and step-function runners | `nd, ns, bk, bias_center, n, seed, beta_0, ..., beta_8` |
| Variable-synapse runner | Parses `nd, ns, bk, bias_center, n, seed, *betas`, then replaces betas with dummy values. |
| Standard raw capacity output | `b, bk, beta, capacity_0, ..., capacity_5` |
| Noise-scan raw output | `noise_level, b, bk, beta, capacity_0, ..., capacity_5` |
| Theta-Y output | `nd, n, b, bk, beta, thetaY_0, ..., thetaY_5` |
| Candidate grid-center table `tmp.csv` | `b_center, bk, n, nd, ns` |

Standard raw output has nine columns, noise-scan output ten, and theta-Y output eleven. Standard outputs usually encode `nd/ns/n` only in their paths. They do not contain the seed, generator class, connectivity, phase lengths, software environment, or learning-rule name. Capacities are saved to two decimal places; `b/bk` to three and beta to six.

At review time, the first row of `tmp_data/params_list_tmp.csv` began with `2,30000,4,1.35,2,20240305` followed by nine beta values. It is therefore a parallel-state table, incompatible with the standard seven-column runner. This is a snapshot of local working data, not a permanent project default.

## Aggregation and grid optimization

[create_gridSearch_df.ipynb](../common_data_processing/create_gridSearch_df.ipynb) loops over hard-coded dendrite counts and selected-dendrite counts, reads each `nd{nd}_ns{ns}/n{n}.csv`, and computes the mean and `scipy.stats.sem` across the last six columns. Its current source root is `../tmp_data/int_variable_n_min1/`. It writes a header-bearing `tmp_data/plot_data/tmp_df.csv` with columns:

```text
nd, ndR, b, bk, beta, capacity, sem
```

Here `ndR` means `n`. The summary drops `ns`, relying on the chosen geometry list and experiment context. With default settings, those lists encode 60,000 total synapses. Broad exception handling silently skips files or rows that fail to load, so the existence of a summary does not establish complete experiment coverage.

[process_gridSearch_df.ipynb](../common_data_processing/process_gridSearch_df.ipynb) selects all rows tied for maximum capacity within each `(nd, ndR)` group and writes `max_points_df.csv`. It also plots the grid, checks all-zero groups and grid alignment, and proposes further `b/bk` points when high-capacity points sit at a boundary. Later cells offer alternative ways to generate candidates around previous maxima. Several plotting consumers break ties by choosing the largest `b` for a repeated capacity and dendrite count.

Raw append-only output is not deduplicated by this aggregation. A repeated parameter point becomes another summary row; model differences cannot be recovered if they were appended into the same file without an external record. The ordinary aggregation code assumes the first three columns are `b, bk, beta` and must not be pointed at noise-scan or theta-Y output unchanged.

Grid plots commonly use `2.571 * sem` for a nominal 95% interval with six measurements. Other fixed-parameter figures use standard deviation; inspect the consumer before comparing error bars. The statistical independence of consecutive driver repeats is not established by the plotting convention.

## Curated data organization

The local `data/` tree includes `gridSearch_standard`, `gridSearch_NonInt`, `gridSearch_int_variable_n_min1`, `gridSearch_StepActivation`, `gridSearch_sparse_binomial`, `gridSearch_variableNs`, `gridSearch_SomaNoise`, `gridSearch_SynapseNoise`, `no_nonlinearity`, `no_gridSearch_noise`, `thetaY_record_no_gridSearch`, and `fixed_params`. Names encode experiment families, often followed by total synapse count, correlation, sparsity, or noise settings.

For example, `data/gridSearch_standard/60000/coeff_infty/max_points_df.csv` supplies optimized independent-input n-WTA results. Folder names are useful organizational context, but the current runners demonstrate why they cannot be treated as complete verified provenance. Some synaptic and dendritic noise plotting inputs both live under `gridSearch_SomaNoise`.

`data/`, `tmp_data/`, `tmp_code_files/`, `figures/`, `tmp_figures/`, and common image/PDF outputs are ignored by the current [.gitignore](../.gitignore). The manuscript is also ignored. A clean clone does not imply these local inputs will be available. Markdown notes under `notes/` are not ignored.

## Fixed-parameter simulations and arrays

The notebooks in [plot_notebooks/](../plot_notebooks/) are an additional data-generation route. Many combine a simulation section, optional or commented-out `np.save` calls, and a plotting section loading previously curated data. Existing saved figures or notebook outputs do not establish that rerunning the current cells will reproduce the same data.

| Notebook | Driver or purpose | Main data directory under `data/fixed_params/` |
| --- | --- | --- |
| `response_age.ipynb` | Recall response versus age, including mean, 1st/99th percentiles, steady threshold, and capacity. | `memory_response_vs_age/` |
| `capa_vs_coeff_fixed_params.ipynb` | Repeat capacity measurements across axon-pool ratios. | `capa_vs_coeff/` |
| `response_distribution_vs_coeff_fixed_params.ipynb` | Response distributions across correlation conditions. | `response_distribution_vs_coeff/` |
| `w_len_distribution.ipynb` | `Driver1_wLens_record`. | `wLens_distribution/` |
| `stepSize_distribution.ipynb` | `Driver1_stepSize_record`. | `stepSize_distribution/` |
| `weight_corr_vs_coeff.ipynb` | `Driver1_WeightCorr_record`. | `weight_corr_vs_coeff/` |
| `nUpdated_distributionvs_coeff.ipynb` | `Driver1_nUpdated_record`. | `nUpdated_vs_coeff/` |
| `capa_vs_nd.ipynb` | Compare existing optimized grid-search summaries. | Uses the grid-search CSV trees instead. |

Common prefixes are `nonInt`, `int_variableN_min1`, and `int`; these identify the three manuscript rules. Suffixes such as `params3` are informal parameter-set labels, not structured metadata. Several consumers assume correlation values `[1, 2, 4, 8, 16, 32, 64]` in that exact order. Response-age consumers load files ending in `upper99.npy`, `transient_upper99.npy`, `transient_lower01.npy`, `transient_mean.npy`, and `capa.npy`.

## Figure production and manuscript connection

[plot_scripts/fig2/](../plot_scripts/fig2/) covers independent-input response curves, weight norms, update sizes, and pre-learning activation distributions. [plot_scripts/fig3/](../plot_scripts/fig3/) covers correlated-input responses, capacity and variability versus correlation, weight correlations, and capacity versus dendrite count. These scripts generally load saved arrays; `fig2/u_distribution_before_learning.py` instead samples Gaussian activations directly for its illustration.

[plot_scripts/fig4/](../plot_scripts/fig4/) uses small panel wrappers and shared helpers. Panels a–f show standard capacity, step activation, sparse input, synaptic noise, dendritic noise, and variable synapse counts. Panels g/h compare maximum capacity across dendrite counts versus synaptic/dendritic noise; panel i shows capacity versus synapse-count variability at fixed `nd=100`. The plotting helpers consume existing optimized summaries, so noise curves here can represent reoptimized parameters rather than fixed-parameter noise scans.

[cosyne2027_plots/panel_d.py](../cosyne2027_plots/panel_d.py) loads fixed-parameter independent-input response curves. [panel_e.py](../cosyne2027_plots/panel_e.py) combines capacity and weight correlation versus axon-pool ratio. [panel_f.py](../cosyne2027_plots/panel_f.py) uses the independent-input standard grid-search summary to plot capacity versus dendrite count. These match the manuscript's d/e/f descriptions.

Current figure scripts save PDFs next to the script. Their style conventions are recorded in [plot_scripts/STYLE_NOTES.md](../plot_scripts/STYLE_NOTES.md). Colors generally identify non-interacting as C0, variable target as C1, and n-WTA as C2.

The local `cosyne2027_plots/` directory includes assembly SVGs and `figure1.pdf`. The manuscript requests `figure1.pdf` by a relative path, but that file was not present under `notes/` during review. Figure assembly and manuscript compilation are separate from the Python simulation workflow; no automatic build pipeline was found.
