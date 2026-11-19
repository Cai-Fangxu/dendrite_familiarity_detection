# Architecture and model

[Notes index](README.md) · [Simulation workflow](simulation_workflow.md) · [Experiments and data](experiments_and_data.md)

## Core modules and interfaces

[lib/base.py](../lib/base.py) defines the abstract `Neuron` and `Xs_Generator` interfaces. A neuron exposes `init_state(rng, hyperD)`, `update(hyperD, state, x)`, and `forward(hyperD, state, x)`. Its constructor binds `update` and `forward` to the corresponding underscored implementations. An input generator exposes `generate(rng, n_patterns, ...)`.

[lib/neurons.py](../lib/neurons.py) contains Flax dataclasses and learning rules. The standard `State1` contains weights `w` of shape `(nd, ns)` and an RNG key. `HyperConfig1` is held by the neuron and includes `nd, ns, n, b, bk, beta`; `HyperDynConfig1` supplies `b, bk, beta` at runtime. Although these three values appear in both configurations, the standard implementations read them from `hyperD`. Many runners explicitly treat their values in `hyper` as placeholders.

[lib/driver.py](../lib/driver.py) owns simulation state, random streams, learning phases, and recordings. It uses `jax.lax.scan` for sequential learning and `jax.vmap` for parallel parameter states. [lib/utils.py](../lib/utils.py) supplies the Numba capacity estimator. [lib/__init__.py](../lib/__init__.py) is empty; callers import the submodules directly.

JIT-compiled methods generally make `self` static. Geometry, the selected-dendrite count used by `top_k`, and constructor-held settings therefore belong to a particular compiled object. Dynamic hyperparameter values can change without recompilation when their shapes and dtypes stay compatible. Driver methods also mark phase lengths and sampling intervals static. Changing array shapes can trigger recompilation even when values are passed dynamically.

## Mathematical dictionary

| Manuscript quantity | Code | Meaning |
| --- | --- | --- |
| $N_d$ | `nd` | Number of dendrites. |
| $N_s$ | `ns` | Synapses per dendrite in the uniform model; nominal average in the variable-size model. |
| $n$ | `n`, historical CSV column `ndR` | Winner count or nominal average selection count. |
| $b$ | `b`, `bs`, `bias_center` | Dendritic readout threshold, an array of thresholds, or its scan center. |
| $u_{\mathrm{target}}=b+\kappa$ | `bk` | Target overlap for the fixed-target rules; `bk` is not kappa alone. |
| $\kappa$ | `bk - b` | Target contribution above threshold. |
| $\beta$ | `beta` | Weight decay parameter; standard updates use the ratio `beta/ns`. |
| $b-\lambda$ | `bla` | Selection threshold derived from `n/nd` in threshold-based rules. |
| $c=N_a/N_s$ | `coeff` | Axon-pool size relative to synapses per dendrite; smaller values mean more shared inputs. |
| $\theta_Y$ | `upper99_s` | Steady-state 99th-percentile response plus a small epsilon. |

One input pattern has shape `(nd, ns)`; a batch has shape `(B, nd, ns)`. For dendrite $r$, the standard model computes $u_r=\mathbf{w}_r\cdot\mathbf{x}_r$, then $z_r=\max(u_r-b,0)$, and neuronal response $y=\sum_r z_r$. Initial standard weights are independent Gaussian samples scaled by $1/\sqrt{N_s}$, giving expected squared dendritic norm one, not exactly normalized norms.

For a selected dendrite, the implemented standard increment is:

$$\Delta\mathbf{w}_r=\frac{\mathbf{x}_r}{\|\mathbf{x}_r\|^2}\left[u_{\mathrm{target}}-\left(1-\frac{\beta}{N_s}\right)u_r\right]-\frac{\beta}{N_s}\mathbf{w}_r.$$

Applying this increment makes the overlap with the current pattern equal to the target, assuming a nonzero input norm. Selection masks decide whether the increment is applied. Unselected dendrites do not receive global decay. The full equations here are reconstructed from the code; [main.tex](main.tex) provides the scientific interpretation but places its numbered equations in the referenced figure.

## Learning-rule implementations

| Class in `neurons.py` | Selection and target |
| --- | --- |
| `Neuron_Decay_Int` | Select the top `n` raw overlaps; apply the increment only where `overlap < bk`; target `bk`. This implements n-WTA. |
| `Neuron_Decay_NonInt` | Update where `bla < overlap < bk`, with `bla = sqrt(2) * erfinv(1 - 2*n/nd)`; target `bk`. This implements the non-interacting rule. |
| `Neuron_Decay_Int_Variable_n` | Select all overlaps above `bla`; target `b + (bk-b)*n/n_selected`. If none pass, leave weights unchanged. |
| `Neuron_Decay_Int_Variable_n_min1` | Same variable target, but if none pass, select the largest overlap and target `b + (bk-b)*n`. This is the manuscript's variable-target rule. |
| `Neuron_Decay_Int_SynapticBound` | Standard n-WTA update followed by resetting selected weights whose absolute value exceeds `reset_from/sqrt(ns)` to signed `reset_to/sqrt(ns)`. |
| `Neuron_Decay_Int_NoNonLinearity`, `Neuron_Decay_NonInt_NoNonLinearity` | Inherit the corresponding learning rule but define a linear readout, `sum(u)`. A matching driver is also required. |
| `Neuron_Decay_Int_VariableNs` | Separate model for unequal dendritic synapse counts, described below. |

The variable-target classes do not apply the fixed-target rules' `overlap < bk` gate: a selected dendrite can be moved downward to its adjusted target. The immediate total response is approximately `n*(bk-b)`, subject to contributions from unselected active dendrites and the readout conditions. For n-WTA, exactly `n` candidates are selected, but fewer may actually change because of the upper gate. For threshold selection, `n` is a nominal mean under a unit-normal overlap approximation; the realized update count need not equal it.

The standard class names do not provide complete support for fractional selection counts at the script level: threshold-based implementations can use fractional `n`, but the inspected grid runners all cast `n` to `int`. Top-k implementations require an integer count within the number of dendrites.

## Input distributions

| Generator in [input_gen.py](../lib/input_gen.py) | Behavior |
| --- | --- |
| `Normal(nd, ns)` | Independent standard normal synaptic inputs. |
| `Normal_Corr_across_Dendrites(nd, ns, coeff, key)` | Create a fixed connectivity map from `int(ns*coeff)` axons, choosing `ns` distinct axons per dendrite. Generate one Gaussian value per axon per pattern and gather through that map. |
| `Normal_random_scaled(nd, ns, rng)` | Multiply Gaussian inputs by a fixed nonnegative random scale matrix whose rows have norm `sqrt(ns)`. |
| `Sparse_Binomial(nd, ns, fA, mu, n_burst, key)` | Multiply a Bernoulli activity gate by a binomial burst count, subtract `fA*mu`, and rescale to unit variance. Generation uses its call-time RNG; the stored constructor key is not used by `_generate`. |
| `Normal_VariableNs(nd, ns_max)` | Generate padded Gaussian arrays and multiply by a supplied binary synapse mask. |

For correlated input, connectivity is fixed for the lifetime of the generator. `coeff=1` means every dendrite receives the same axon pool, potentially in a different local synapse order. `coeff` is not a Pearson correlation coefficient. The expected fraction of shared axons is approximately `1/coeff`; the exact finite-pool fraction is `ns/int(ns*coeff)`. Independent input is represented by `Normal` and often labeled `coeff_infty` in filenames. Sampling without replacement requires `int(ns*coeff) >= ns`.

Sparse input has variance before rescaling `fA * (mu*(1-mu/n_burst) + mu**2*(1-fA))`. Valid settings must give a positive variance and a binomial success probability between zero and one. With `mu=n_burst=4`, an active burst is always four spikes; randomness then comes from the activity gate.

## Variable synapse counts

`HyperConfig2`, `HyperDynConfig2`, and `State2` currently belong to the variable-synapse model. Weights and inputs have shape `(nd, ns_max)`; `ns_mask` marks valid synapses, and `l_array[r]=sqrt(ns_r)`. Initial dendritic norms are normalized exactly to `sqrt(ns_r/ns)`. Each readout threshold is scaled by that same factor.

The update ranks normalized overlaps `u_r / ||w_r||`. It rescales selected weight vectors toward their prescribed norms and adjusts the target according to the summed square roots of selected synapse counts. Although `beta` is still a dataclass field, this update does not use it. `Driver2_parallel_states` supplies masked inputs and the scaled readout.

The driver generates shared input using `hyperD.ns_mask[0]`, so parallel states are intended to share the same dendritic geometry. Different thresholds can be compared in parallel; different masks should not be assumed to work as independent geometries in that same call.

## Readout and diagnostic drivers

Most recall drivers directly calculate overlaps and responses instead of calling `neuron.forward`. Changing only a neuron's forward method therefore does not change recorded recall responses. The linear-control driver and the step-function runner explicitly override recording methods as well.

Beyond the main drivers, `Driver1_bScan_activationScan` scans both threshold and saturation window, using `min(ReLU(u-b), window_len)`. The threshold-only recorders retain `upper99_s`. The weight-norm and step-size recorders store sampled per-dendrite values in `steady_record`. `Driver1_WeightCorr_record` embeds weights into a shared axon coordinate system and averages off-diagonal cosine similarities; it requires a correlated-input generator and `nd > 1`. `Driver1_nUpdated_record` counts weight changes above `1e-6`, not simply selected candidates; it stores counts in `transient_record` and only runs initialization and phase 1.
