# Caveats and restart checklist

[Notes index](README.md) · [Architecture](architecture.md) · [Workflow](simulation_workflow.md) · [Experiments and data](experiments_and_data.md)

These observations come from source inspection, not an executed regression suite. They identify behavior to account for before future work; no fixes were made during this review.

## Before launching a new simulation

1. Record the exact neuron and driver classes, input distribution, geometry, correlation/noise parameters, beta source, random seeds, phase lengths, sampling interval, and environment. The raw CSV schema does not retain most of this information.
2. Check the actual parameter-table shape and column order against the selected runner. The current working table has 15 columns; the basic runner expects seven. A single-row `np.genfromtxt` result is one-dimensional with the current calls, so the existing row loop also does not handle a one-row table as intended.
3. Confirm that the learning rule permits the chosen scan strategy. Readout-only `b` scans do not retrain a variable-target rule for each `b`. Custom readout changes must be reflected in the driver's recording methods.
4. Review output destinations before running. Several distinct experiment types append to identical paths, and none of the runners deduplicates completed rows or resumes from a recorded checkpoint.
5. Check which settings cause a runner to rebuild its objects. Most only compare `nd` and `n`; changes in other constructor-held settings need deliberate handling.
6. Use a pilot to inspect finite outputs, actual weight norms and update counts, recall curves, and whether a capacity crossing lies inside the recorded age window before committing to a large sweep.
7. After aggregation, verify the expected geometry/parameter groups are present, inspect grid edges and all-zero capacities, and retain the data needed to trace each plot back to a specific experiment.

## Object reuse and randomness

Most runner rebuild conditions are `nd != old_nd or n != old_n`. They do not check `ns`. If adjacent rows change `ns` while preserving `nd/n`, the existing neuron and generator retain their earlier shapes even though the new output folder uses the new `ns` label. The common fixed-total-synapse lists avoid this particular mismatch, but a new scan need not.

The noisy runners also change correlation and noise settings at fixed row-index boundaries. Those changes take effect on the actual objects only if the rebuild condition fires at that boundary. Row ordering is therefore part of the experiment configuration, and rearranging a parameter table can alter the conditions.

Correlated-input connectivity is generated only when its generator is constructed. `re_init` replaces the simulation RNG and state but does not rebuild connectivity. A row's seed alone is consequently insufficient to reconstruct the connectivity if that row reused an earlier generator. Parameter row order and object-rebuild history matter.

The optional `Driver1_NoisyRecall_SomaNoise3Scan` reads its fixed readout threshold from `neuron.hyper.b`, rather than `hyperD.b`. Reinitializing only the dynamic hyperparameters therefore does not change that driver's readout threshold. The currently active noise-scan runner uses the dendritic-noise driver instead.

Parallel-state drivers share input patterns and initialization keys across parameter values. Most repeated trials continue one state. In the variable-synapse runner, geometry arrays are regenerated when `nd` changes, reused across other settings, and selected by repeat; each repeat resets with the same row-level key. These designs provide shared randomness, not six automatically independent network/seed draws.

## Result interpretation

The capacity estimator returns zero if it sees no downward crossing, including when the recording window is too short to observe forgetting. It also averages all downward crossings. A zero-capacity filter in an analysis notebook cannot distinguish these cases without consulting the response curves.

The non-interacting threshold is calculated under an approximate unit-normal overlap distribution. Weight norms, sparse inputs, and shared input can change the realized update statistics. The variable-target classes omit the fixed-target upper gate. Do not infer identical selection or update behavior solely from a common nominal `n`.

`Driver1_nUpdated_record` is a specialized path that should be checked before reuse: its recording method returns one scalar per update, but it inherits `record_phase1`, whose final reshape assumes two trailing response axes. The specialized run method flattens that result back into one sequence of update counts. Its `run` method also ignores the later-phase arguments.

`Driver1_WeightCorr_record` measures cosine similarity in axon coordinates, not mean-subtracted Pearson correlation between local weight arrays. Its division by `nd*(nd-1)` requires more than one dendrite.

Sparse and variable-synapse inputs must have valid nonzero norms where updates divide by those norms. The variable-synapse initialization also divides by each masked weight norm, so a dendrite with zero valid synapses is not supported by the current formula.

## Noise means several different things here

All inspected noise drivers add noise during recall; learning still uses the underlying clean input and neuron update. The driver class, not just a numeric noise level, determines the noise model.

| Driver suffix | Implemented noise |
| --- | --- |
| `SynapseNoise` | Gaussian overlap noise with scale `noise_level * ||w_r||`, before ReLU. |
| `DendriteNoise` | Gaussian noise added only to positive dendritic activations, with scale `noise_level * (bk-b) * ||w_r||`. No second rectification is applied afterward. |
| `SomaNoise` | Gaussian somatic noise with scale `noise_level * ||W||_F / sqrt(nd)`; the source includes a commented older scale without the normalization. |
| `SomaNoise2` | Gaussian somatic noise scaled by `noise_level * n * (bk-b)`. |
| `SomaNoise3` | Clip the clean response to `[0, max_Y*theta_Y]`, then add Gaussian noise scaled by `noise_level * theta_Y`; final noisy responses are not clipped again. |
| `DendriteSomaNoise` | Dendritic noise plus somatic noise scaled by the steady threshold measured with dendritic noise. |
| `SomaNoise3Scan` | Scan theta-Y-scaled additive noise over saved responses; unlike the similarly named non-scan class, its noise helper does not clip responses. |
| `DendriteNoiseScan` | Scan dendritic noise at one fixed `hyperD.b`. |

Dendritic noise can make a previously positive activation negative because it is added after ReLU without reclipping. This is an implemented modeling choice, not a conventional always-nonnegative noisy ReLU.

Phase-3 record filtering removes any row containing a value close to the sentinel `-100`. For unbounded linear or noisy outputs, an actual response near that value could in principle also be removed; the filter does not store a separate validity mask.

## Historical and operational limitations

The root `test2.ipynb` refers to the old adaptive-threshold `Neuron_Decay_Int2`, `Driver2_shift_freq_record`, and hyperparameters such as `gamma`. Their definitions are commented out in the library. The live `HyperConfig2` instead describes variable synapse counts, so this notebook is not a current runnable example.

The step-function runner and the ReLU variable-target runner use the same output tree. The basic, sparse-binomial, and linear-control runners also share an output pattern. `run_get_thetaY.py` writes beneath `coeff_infty` while actively using `coeff=2`; the noise-scan runner carries condition labels that do not fully match its driver. These are concrete reasons to verify source settings before interpreting folders as experiment metadata.

Several analysis notebooks silently catch loading or selection failures and overwrite intermediate tables in later cells. Their cells represent interactive workflows and alternative branches. A blanket “Run All” is not a verified reconstruction procedure.

The Mathematica notebooks supply a significant part of parameter preparation but contain saved state, alternative definitions, and absolute filesystem paths. Their equations and numerical solutions were not fully validated in this documentation review. The next simulation should identify which beta calculation is appropriate rather than assume the current export applies to every neuron variant.

There is no checked-in environment lock, checkpoint/resume mechanism, general command-line configuration, or automated regression harness in the inspected code. Many datasets and figure assets are local ignored files. The current shell interpreter is also missing some required packages, as noted in [Simulation workflow](simulation_workflow.md). These notes establish where to resume, not that the current environment has already been validated for numerical reproduction.
