# Dendritic Angular-Plasticity Model

## Model

A neuron has $N_d$ dendrites, each with $N_s$ synapses. For dendrite $r$:

- $\mathbf{x}_r\in\mathbb{R}^{N_s}$ is its input vector for the current pattern.
- $\mathbf{w}_r\in\mathbb{R}^{N_s}$ is its weight vector, with $\|\mathbf{w}_r\|=1$.
- Its **local activation** is $u_r=\mathbf{w}_r\cdot\mathbf{x}_r$.

The neuronal response is

$$
Y=\sum_{r=1}^{N_d}\max(u_r-b,0),
$$

where $b$ is the dendritic activation threshold.

Initialize each weight vector with independent standard Gaussian entries, then normalize it to unit length. Present patterns sequentially and update the weights once per pattern.

## Learning rules

The two rules use the same parameters: $\eta>0$ (learning strength) and $\beta\ge0$ (preference for dendrites with higher activation). Compute all $u_r$ **before** updating any weights.

### Interacting

For every pattern, assign a share of the total angular update to each dendrite:

$$
p_r=\frac{\exp(\beta u_r)}{\sum_{s=1}^{N_d}\exp(\beta u_s)},
\qquad
\boxed{\Delta\theta_r=\frac{\eta}{\sqrt{N_s}}p_r.}
$$

### Non-interacting

Each dendrite's update depends only on its own local activation:

$$
\boxed{
\Delta\theta_r
=\frac{\eta}{N_d\sqrt{N_s}}
\exp\!\left(\beta u_r-\frac{\beta^2}{2}\right).
}
$$

Use the same $\beta$ for both rules at a given target $n$. The learning strength $\eta$ can be optimized numerically for each experiment.

## Choosing $\beta$ from a target $n$

Use a fixed target $n$ to label each experimental curve, analogous to the number of updated dendrites in the original model. In the new soft-allocation rules, $n$ means an **effective number of updated dendrites**, not a hard count.

For a pattern, compute the relative plasticity shares

$$
p_r=\frac{e^{\beta u_r}}{\sum_s e^{\beta u_s}}.
$$

Define the effective number for that pattern as

$$
n_{\mathrm{eff}}(\mathbf u)=\frac{1}{\sum_r p_r^2}.
$$

This relative-allocation measure is **the same for the interacting and non-interacting rules**. To assign one $\beta$ to a fixed target $n$, use the ensemble-level calibration

$$
\boxed{
\left\langle\sum_{r=1}^{N_d}p_r^2\right\rangle_\beta=\frac1n,
\qquad u_r\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1).
}
$$

**Calibration procedure:** For each pair $(N_d,n)$, generate batches of independent standard-normal vectors of length $N_d$, compute their softmax shares for a trial $\beta$, average $\sum_r p_r^2$ over the batches, and solve the boxed equation with a one-dimensional search (e.g. bisection). Use a numerically stable softmax by subtracting $\max_r(\beta u_r)$ before exponentiating. Hold the resulting $\beta$ fixed for the corresponding $n$ curve, for both learning rules and all input-correlation conditions. Recalibrate if $N_d$ or the target $n$ changes.

A simple approximation for choosing the initial search range is

$$
\boxed{\beta\approx\sqrt{\log(N_d/n)}}.
$$

**Do not use this approximation as the final calibration** when $n$ is small or $N_d$ is finite; solve the expectation equation numerically instead. The exact endpoints are $n=N_d\Rightarrow\beta=0$ and $n\to1\Rightarrow\beta\to\infty$, so use a sufficiently large finite $\beta$ and a specified tolerance when approximating $n=1$.

### Update the weight vector

For either rule, rotate the weight vector toward its current input while maintaining unit norm:

$$
\mathbf{e}_r
=\frac{\mathbf{x}_r-u_r\mathbf{w}_r}
{\|\mathbf{x}_r-u_r\mathbf{w}_r\|},
$$

$$
\boxed{
\mathbf{w}_r^{\mathrm{new}}
=\mathbf{w}_r\cos(\Delta\theta_r)
+\mathbf{e}_r\sin(\Delta\theta_r).
}
$$

If $\|\mathbf{x}_r-u_r\mathbf{w}_r\|=0$, leave $\mathbf{w}_r$ unchanged for that update. Apply all updates using the activations calculated from the weights before the learning step.

## Input models

All input values are standard Gaussian. Different patterns are generated independently. Input correlation across dendrites comes only from reusing signals from the same presynaptic axons.

### Uncorrelated inputs

For each new pattern, independently sample

$$
\mathbf{x}_r\sim\mathcal N(\mathbf0,I_{N_s})
\qquad (r=1,\ldots,N_d).
$$

No dendrites share input values within a pattern.

### Correlated inputs: shared axons

Define $N_a\ge N_s$ available presynaptic axons and the shared-input coefficient

$$
c=\frac{N_a}{N_s}.
$$

At the start of a simulation, **for each dendrite**, choose $N_s$ distinct axon indices uniformly from $\{1,\ldots,N_a\}$. Keep this connectivity fixed throughout the simulation. Different dendrites may select the same axon.

For each new pattern:

1. Sample one axonal signal vector $\mathbf{x}\sim\mathcal N(\mathbf0,I_{N_a})$.
2. Construct $\mathbf{x}_r$ by reading the entries of $\mathbf{x}$ at the fixed axon indices connected to dendrite $r$, in the same order as the corresponding entries of $\mathbf w_r$.
3. Compute all $u_r$ and apply either learning rule above.

At $c=1$, all dendrites receive the same $N_s$ axonal signals (possibly in a different synapse ordering). As $c$ increases, fewer axons are shared between dendrites.

## Four configurations

Run the same learning procedure for each combination:

| Learning rule | Input model |
|---|---|
| Interacting | Uncorrelated |
| Non-interacting | Uncorrelated |
| Interacting | Correlated, with chosen $c\ge1$ |
| Non-interacting | Correlated, with chosen $c\ge1$ |

For each target $n$ curve, calculate $\beta$ using the calibration above. Hold $N_d$, $N_s$, $n$ (hence $\beta$), and the input-model settings consistent across the learning-rule comparison; optimize $\eta$ and $b$ numerically as needed.

If tracking a pattern learned at time $0$, denote its later local activation by

$$
u_r^{(0)}(t)=\mathbf{w}_r(t)\cdot\mathbf{x}_r^{(0)}.
$$
