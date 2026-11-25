# Dendritic computation — simulation validation plan

**Current scope: Figures 1 and 2 only.** Figure 1 retains the frozen-state memory-overlap drift/diffusion protocol below. Figure 2, described after Figure 1, tests the deterministic evolution and local drift of individual weight overlaps $Q_{rs}$. Figures beyond Figure 2 are **not specified**. The goal is to obtain and inspect simulation results, not to design a manuscript layout.

**Figure 1 question:** Do the stochastic drift and diffusion coefficients of an old memory's dendritic overlaps agree with the theoretical predictions (R4 for independent dendritic inputs, R8 for fully shared inputs), when measured locally at *actual trained weights*?

**Core design (updated after implementation discussion):** Train normally to time \(t\), freeze \(W_t\), choose real patterns learned at different ages \(\tau\), and generate many independent hypothetical **multistep forks** starting from that *same* \(W_t\). Each fork evolves sequentially for a configurable \(\ell\) updates. Compare its rescaled endpoint moments with numerical Gaussian expectations evaluated at the initial frozen state. These finite-duration estimates can differ from local R4/R8 coefficients; compare \(\ell=10\) and \(20\) to inspect this bias. Setting \(\ell=1\) recovers the original one-step protocol.

**Implemented entry points:** [`run_fig1.py`](../../theory_check/run_fig1.py), [`fig1_config.example.json`](../../theory_check/fig1_config.example.json), and [`plot_coefficients.py`](../../plot_scripts/theory_fig1/plot_coefficients.py). Fill the required learning parameters in a copy of the example JSON; `null` is deliberately not a default learning parameter. Rules can override `nd` and `n` separately or be explicitly disabled with `enabled: false`. Initial architecture: \(N_d=100,n=10,N_s=600\). Eligibility is derived as \(L=\Phi^{-1}(1-n/N_d)\). The original five entries use leading-angle unit-sphere rotations; the optional `*_decay` entries reuse the legacy weight-decay updates (see §6). Variable-target amplitudes are signed. Fixed defaults are **50,000 forks and 10 updates per fork**, configurable in JSON or with `--forks` / `--fork-steps`. Age zero is immediately after learning.

Each invocation runs **one input condition**: set `"condition": "independent"` or `"condition": "shared"`. Use a separate configuration for each condition to choose its learning parameters independently. `stages` is a flat list for that condition, for example `[6000]` for independent inputs or `[600, 3000, 12000]` for shared inputs. The former `conditions` list and condition-keyed stage dictionaries are no longer accepted. Saved runs from both conditions can still be passed together to the plotting script, including when their learning parameters differ.

`"seeds": [0, 1, 2]` selects three independent training replicates per enabled rule: different initial weights, training histories, and fork streams. These are RNG seed identifiers, not training stages or fork counts. Three is a configurable default for assessing variation between trained states, not a theoretical requirement; use `"seeds": [0]` for a single replicate. The fixed number of forks applies separately to each seed and snapshot. Within a condition/architecture, matching seeds pair the random streams across rules. `sampling.theory_seed` separately controls numerical Gaussian integration.

```bash
python theory_check/run_fig1.py --config /path/to/fig1_config.json --validate-only
python theory_check/run_fig1.py --config /path/to/fig1_config.json
python theory_check/run_fig1.py --config /path/to/fig1_config.json --fork-steps 20
python plot_scripts/theory_fig1/plot_coefficients.py /path/to/fig1_run_10 /path/to/fig1_run_20
```

Plot all saved rules by default, select a subset with `--rules nwta variable_target`, or hide rules with `--exclude-rules softmax_nonint`. These mutually exclusive options apply to every coefficient panel, diagnostic and legend, retaining the same rule colors and leaving saved simulation data unchanged. Use `--output` to save a filtered plot set separately from existing figures.

Display controls: `--max-points 100` caps displayed coefficient points **per rule, stage, and panel**, sampling across seeds and ages. `--sample-seed 0` controls a reproducible selection independent of observed values, rule filtering, and input-file order. The same selected points appear in the combined and stage-row figures. `--point-size 2` sets marker size in points; `--no-error-bars` hides both horizontal and vertical point uncertainty bars. All default to the original display (all points, size 3, errors shown). These controls do not change saved coefficients, age averages, numerical theory, or uncertainties. Snapshot-level Gram diagnostics retain all snapshots. `plot_options.json` records the display settings beside the figures.

Shared-input plotting saves both the combined `shared_CDE` figure and `shared_CDE_by_stage`, whose rows are training stages in increasing order and columns are C/D/E. All selected rules appear together within each row. Each column uses the same axis limits across stages and as the corresponding combined panel. Both figures retain the same uncertainty estimates, identity lines, and age aggregation for D/E.

Outputs go to a new timestamped directory under `/media/fangxu/T7/dendritic_computation`. Each completed snapshot has `metadata.json`, `snapshot.npz`, `empirical.npz`, `theory.npz`, and `points.csv`. Full frozen weights and selected acquisition patterns are retained; fork trajectories are discarded. Run metadata records seeds, configuration, software/device information, and Git revision/status. Plotting reads completed snapshots only and separates architectures and fork lengths. PDF/PNG figures are saved under the first run's `figures/` directory unless `--output` is supplied. NumPy, SciPy, and JAX are required for simulation; Matplotlib is required for plotting. Float64 is the default, with `dtype: "float32"` available explicitly.

## 1. Notation and theoretical predictions

- \(N_d\): number of dendrites. \(N_s\): synapses per dendrite. In the current codebase, \(N_dN_s=60{,}000\).
- \(t\): training step of the frozen weights \(W_t\). \(\tau\): age of a selected stored pattern at that step. Fix a convention for whether age zero is immediately after learning.
- \(\ell\): updates per fork (`fork_steps`), distinct from the eligibility threshold \(L=b-\lambda\).
- \(u_r=\mathbf w_r\cdot\mathbf x_r^{\mathrm{new}}\): **local activation** for a newly presented input, not the stored overlap.
- \(m_r=m_r(t;\tau)=\mathbf w_r(t)\cdot\mathbf x_r^{\mathrm{old}}\): current overlap with a particular stored pattern of age \(\tau\). \(\mathbf m\) is its full vector across dendrites.
- \(a_r(\mathbf u)\): rule-dependent angular learning amplitude, with angle \(\theta_r=a_r(\mathbf u)/\sqrt{N_s}\) for the asymptotic spherical update.
- \(Q_t\): actual Gram matrix of the frozen unit weight vectors in a common axonal coordinate system; for full sharing, \((Q_t)_{rs}=\mathbf w_r(t)\cdot\mathbf w_s(t)\).
- \(\xi=t_{\mathrm{additional}}/N_s\): *rescaled additional-learning time* used in the diffusion theory. **Do not confuse \(\xi\) with the pattern age \(\tau\).**

For \(K\) independent forks, perform \(\ell\) hypothetical sequential learning updates from \(W_t\) in each fork. For a chosen stored pattern, calculate the endpoint change

\[
\Delta m_r^{(k)}=m_r^{\prime(k)}-m_r,\quad k=1,\dots,K.
\]

Estimate the **conditional rescaled drift** and **conditional diffusion covariance**:

\[
\boxed{\widehat A_r=\frac{N_s}{\ell}\,\overline{\Delta m_r},\qquad
\widehat D_{rs}=\frac{N_s}{\ell}\,\widehat{\operatorname{Cov}}_k(\Delta m_r^{(k)},\Delta m_s^{(k)})}. \tag{1}
\]

Use a centered sample covariance, with \(K-1\) in its denominator. One data point in a single-dendrite plot represents a specific **rule, seed, frozen state \(t\), stored pattern/age \(\tau\), and dendrite \(r\)**. Cross-diffusion points additionally identify a dendrite pair \((r,s)\).

### 1A. Independent dendritic inputs: R4

For \(\mathbf z\sim\mathcal N(0,I_{N_d})\), evaluate the Gaussian moment for each learning rule, using its **user-provided fixed hyperparameters**:

\[
\boxed{D_r^{\mathrm{G}}=\mathbb E_{\mathbf z}[a_r(\mathbf z)^2]}.\tag{2}
\]

For exchangeable rules, \(D_r^{\mathrm G}=D^{\mathrm G}\) for every \(r\). The predictions are

\[
\boxed{A_r^{\mathrm{th}}=-\tfrac12 D_r^{\mathrm G}m_r,\qquad
D_{rr}^{\mathrm{th}}=D_r^{\mathrm G},\qquad D_{r\ne s}^{\mathrm{th}}=0}.\tag{3}
\]

The theoretical coefficient is independent of training time \(t\), memory age \(\tau\), and current overlap \(m_r\), apart from the explicit linear dependence of drift on \(m_r\). **One frozen training stage** with many old-pattern probes is sufficient for the primary test. Multiple independent training seeds are still useful.

### 1B. Maximally shared dendritic inputs: R8

At each **actual** frozen state \(W_t\), use the actual \(Q_t\) to draw independent Gaussian activation vectors \(\mathbf u\sim\mathcal N(0,Q_t)\). Numerically estimate

\[
\begin{aligned}
C_{rs}(Q_t)&=\mathbb E[a_r(\mathbf u)a_s(\mathbf u)],\\
F_{rs}(Q_t)&=\mathbb E[a_r(\mathbf u)u_s],\\
J(Q_t)&=F(Q_t)Q_t^+,\\
H(Q_t)&=J(Q_t)-\operatorname{diag}\left(F_{rr}(Q_t)+\tfrac12 C_{rr}(Q_t)\right),
\end{aligned}\tag{4}
\]

where \(Q_t^+\) is the Moore–Penrose pseudoinverse. For each tested stored pattern, predict

\[
\boxed{\mathbf A^{\mathrm{th}}=H(Q_t)\mathbf m,\qquad
\mathbf D^{\mathrm{th}}=C(Q_t)}.\tag{5}
\]

**Do not average \(Q_t\) across training time** and do not substitute a single average correlation for the measured Gram matrix. The same \(C(Q_t),H(Q_t)\) can be reused for all old patterns at a fixed snapshot, but \(H(Q_t)\mathbf m\) must be recomputed for *each old pattern*. For an interacting rule, the drift of dendrite \(r\) can depend on the full old-pattern overlap vector \(\mathbf m\), not only \(m_r\). The **off-diagonal** diffusion coefficients \(C_{rs}\) are important predictions, not optional consequences of the diagonal coefficients.

The formulas are leading large-\(N_s\) predictions at fixed \(N_d\); finite-\(N_s\) corrections can matter, particularly when \(N_d\ge N_s\).

## 2. Experiment: freeze, probe, branch, measure

**Step 1 — train and freeze.** For a specified rule and input condition, initialize the weights randomly, train sequentially on genuinely independent Gaussian patterns, and save frozen weights \(W_t\) at selected stages. Keep the actual training history or sampled old patterns with their acquisition times.

**Step 2 — choose old patterns.** For a fixed \(t\), select actual previously learned patterns at multiple ages \(\tau<t\) spanning recent, intermediate and old memories. Configured ages \(\tau\ge t\) are automatically skipped at that stage and reported in the console. If no ages are valid, skip that stage's measurement; training to later measured stages still includes all preceding updates. Saved snapshot arrays and point tables contain only that stage's valid ages; the run configuration retains the full requested age list. If all combinations are unavailable, no snapshots are produced. Compute \(\mathbf m(t;\tau)\) for each valid age. Across patterns and dendrites, this should give a broad range of \(m_r\). Store the **full** \(\mathbf m\) for the correlated-input theory.

**Step 3 — generate hypothetical multistep forks.** Draw \(K\) independent input sequences of length \(\ell\), from the same distribution and with the same axonal connectivity as ordinary training. Start each sequence at the original \(W_t\), evolve its weights sequentially for \(\ell\) updates, and compute the endpoint \(\Delta\mathbf m^{(k)}\) for all selected old patterns. Discard that trajectory and restart the next fork from the original \(W_t\). Reuse the same trajectories across old patterns. Never reset weights between steps *within* one fork.

**Step 4 — extract empirical coefficients.** Calculate Eq. (1), plus sampling standard errors / confidence intervals, for each old pattern and dendrite. Calculate cross-covariances for a prespecified selection of dendrite pairs, including pairs with positive and negative predicted covariance when available. **Do not pool forks from different \(t\)** when estimating a conditional coefficient.

**Step 5 — compare with numerical theory.** For independent inputs, compute \(D^{\mathrm G}\) once per rule/parameter setting and apply Eq. (3). For full sharing, compute \(C(Q_t),F(Q_t),H(Q_t)\) **separately for every frozen snapshot**, then apply Eq. (5) to each sampled old pattern. Estimate theory moments with independent random draws (and report their own numerical uncertainty).

**Step 6 — vary training stage for correlated inputs.** Repeat at early, intermediate and late \(t\), so the actual \(Q_t\) explores states with different weight correlations. Stages should be prespecified or selected by an exploratory look at \(Q_t\), **not** selected for agreement with the theory. For independent inputs, this step can be skipped.

**Step 7 — repeat for learning rules.** Use the **same architecture, input condition, age-selection protocol, and fixed user-supplied parameter setting** wherever definitions permit. Reuse training input random seeds across rules if convenient, while maintaining independent fork-vs-theory Gaussian random streams. All learning rules will be shown **together** in each relevant plot, encoded by color.

## 3. Plots to generate for Figure 1

**Plot rules together, not in separate rule-specific plots.** Keep the same color for a given rule across the entire analysis. Use point markers (or marker edge style) for training stage \(t\); use opacity or subtle marker sizes for individual old-pattern ages \(\tau\). Do **not** assign different colors to training stages, because color is reserved for the learning rule.

Proposed consistent color mapping (color-blind-friendly, adjust only once for the whole project):

| Rule | Color | Curves/data encoding |
|---|---|---|
| Baseline, non-interacting threshold | Blue `#0072B2` | Blue theoretical line + blue measured points |
| n-WTA | Vermilion `#D55E00` | Vermilion theoretical line + measured points |
| Variable-\(u_{\mathrm{target}}\) | Bluish green `#009E73` | Green theoretical line + measured points |
| Softmax interacting | Reddish purple `#CC79A7` | Purple theoretical line + measured points |
| Softmax non-interacting | Golden orange `#E69F00` | Gold theoretical line + measured points |
| Threshold with decay | Light blue `#56B4E9` | Separate legacy-rule estimates |
| n-WTA with decay | Light orange `#F3A366` | Separate legacy-rule estimates |
| Variable target with decay | Light green `#73C9A8` | Separate legacy-rule estimates |

A common black dashed identity line \(y=x\) is a **reference**, not a learning rule. For a single plot, **all five rules may appear simultaneously**; if crowded, use small/transparent points and emphasized median-with-uncertainty markers, *not* separate colors for time or a new rule-specific layout. Avoid choosing different axis ranges across rules.

### Plot A — independent-input drift vs old-pattern overlap

- **X:** observed \(m_r(t;\tau)\).
- **Y:** empirical \(\widehat A_r=(N_s/\ell)\overline{\Delta m_r}\).
- **Prediction:** for each rule, the straight line \(A=-D^{\mathrm G}m/2\), plotted in that rule's color with **no fitted slope**.
- **Empirical:** individual frozen-state estimates, colored by rule; show error bars or binned/uncertainty-weighted summaries so the noise is interpretable.
- **Main question:** is the drift linear in \(m\), with the predicted rule-dependent slope, independent of old-pattern age?

### Plot B — independent-input diffusion vs old-pattern overlap

- **X:** observed \(m_r(t;\tau)\).
- **Y:** empirical \(\widehat D_{rr}\).
- **Prediction:** for each rule, a horizontal line at \(D^{\mathrm G}\), in that rule's color.
- **Empirical:** the same probe points and consistent colors as Plot A.
- **Main question:** is diffusion independent of the old-pattern overlap and age, and does its magnitude match the Gaussian integral?

### Plot C — shared-input drift: predicted vs measured

- **X:** snapshot- and pattern-specific \((H(Q_t)\mathbf m)_r\).
- **Y:** empirical \(\widehat A_r\).
- **Prediction:** **one** black \(y=x\) line; each rule has its **own colored points**, never separate fitted theory lines.
- **Stages:** early/intermediate/late \(t\) represented by different markers **within each color**.
- **Main question:** does the full-vector drift prediction work at varying \(Q_t\), for memories of different ages and overlap values?

### Plot D — shared-input diagonal diffusion: predicted vs measured

- **X:** \(C_{rr}(Q_t)\).
- **Y:** empirical \(\widehat D_{rr}\).
- **Prediction:** \(y=x\); points colored by rule and marker-shaped by \(t\).
- **Note:** at fixed \(t\), the predicted diffusion is independent of old-pattern age in the asymptotic model. Multiple ages can land at the same predicted x-value; use an age-aggregated point per dendrite/snapshot for readability, while retaining age-resolved values for the diagnostic tests.

### Plot E — shared-input cross-diffusion: predicted vs measured

- **X:** \(C_{rs}(Q_t)\), selected \(r\ne s\).
- **Y:** empirical \(\widehat D_{rs}\) from the **same stored pattern** across two dendrites.
- **Prediction:** \(y=x\); include \(x=0\) and \(y=0\) references.
- **Main question:** is the *correlated noise* prediction correct, beyond marginal single-dendrite diffusion? Prefer a prespecified subset of pairs with a discernible spread in predicted values.

### Diagnostics to accompany the plots (not future figures)

- Drift residual \(\widehat A-A^{\rm th}\) and diffusion residual \(\widehat D-D^{\rm th}\) vs **age** \(\tau\), **current overlap** \(m_r\), and **training stage** \(t\). Use the same rule colors.
- A record of \(R_t=N_d^{-1}\sum_{rs}(Q_t)_{rs}\), off-diagonal rms \(Q_t\), and \(\operatorname{rank}(Q_t)\) by stage, to show what states were actually tested.
- Optional finite-\(N_s\) reference markers, particularly if the leading diffusion limit disagrees with measured coefficients.

This is a list of diagnostic **result plots**, not a fixed manuscript panel layout. Nothing beyond Figure 1 is defined yet.

## 4. Statistical precision: conditional drift is the expensive observable

Single-step drift is \(O(1/N_s)\) while one-step noise is typically \(O(N_s^{-1/2})\). For forks short enough that coefficients change little, with \(D\) a representative diffusion coefficient,

\[
\operatorname{SE}(\widehat A_r)\approx\sqrt{\frac{N_sD}{K\ell}},
\qquad
\operatorname{SE}(\widehat D_{rr})/D\approx\sqrt{\frac{2}{K-1}}
\]

for approximately Gaussian increments. **User-selected policy: fixed \(K\), with no adaptive stopping.** Report measured errors for every rule even when their amplitudes differ. Defaults \(K=50{,}000,\ell=10\) give drift SE approximately \(0.0346\sqrt D\) at \(N_s=600\); \(\ell=20\) gives \(0.0245\sqrt D\). Diffusion covariance and drift both scale by \(1/\ell\); noise standard deviation scales by \(1/\sqrt\ell\). Longer forks introduce finite-duration bias and cost \(K\ell\) total updates. Near \(m_r=0\), keep data for diffusion/residual checks but do not divide drift by \(m_r\).

Empirical covariance uses centered, merged moments with \(K-1\) denominator. Independent near-equal fork groups provide covariance uncertainty; averaging ages happens within each group before estimating its SE. The saved full group covariances retain the dependence needed for further summaries. Theory uses independent Gaussian MC groups, with snapshot-specific drift uncertainty propagated through the full overlap vector. Plot intervals are conditional on each frozen state, not a population estimate treating dendrites as independent training seeds. Independent-input Gaussian moments are reused across snapshots and seeds, so their numerical uncertainty is shared.

The following **one-step** variance-reduction identities are background only; the multistep implementation measures raw fork endpoints:

- **Independent inputs, angular update:** the new-pattern tangent direction is independent of the amplitudes, giving the exact conditional mean \(\mathbb E[\Delta m_r\mid\mathbf u]=m_r[\cos(a_r(\mathbf u)/\sqrt{N_s})-1]\). This removes the zero-mean perpendicular projection noise and gives an accurate finite-angle drift estimate by Gaussian sampling.
- **Fully shared inputs:** Gaussian conditioning gives \(\mathbb E[y\mid\mathbf u]=\mathbf m^\top Q_t^+\mathbf u\), where \(y=\mathbf x^{\rm old}\cdot\mathbf x^{\rm new}\). Use this in the **leading-order** overlap update to reduce variance; substituting it directly into the exact finite-angle denominator is not in general an exact conditional expectation.
- Forks are independent *within a frozen state*, but points sharing \(W_t\), its old patterns, and its forked inputs are statistically dependent. For figure-wide uncertainty, replicate independent training seeds and use clustered uncertainty estimates; do not mistake the number of scatter points for the number of independent neurons.

Numerical Gaussian-moment estimates should use independent random streams from the forked updates; record numerical MC confidence intervals alongside empirical uncertainties. No fitted multiplicative correction to theoretical drift, diffusion, or the time axis is allowed. Optional regression slopes (measured vs predicted) are **diagnostics**, not fitted theory coefficients.

## 5. Large-\(N_s\) benchmark versus exact finite-\(N_s\) geometry

**Excluded from the current implementation at the user's request.** The local Gaussian R4/R8 predictions remain included. The construction below is retained as theoretical background; no optional finite-\(N_s\) geometric benchmark is computed.

All predictions above assume strictly normalized weights and the fixed-\(N_d\), large-\(N_s\) limit. It is useful to compute a **finite-\(N_s\) benchmark from spherical geometry** when interpreting discrepancies, without altering the user-supplied learning parameters.

**For independent inputs** and \(\theta=a_r(\mathbf u)/\sqrt{N_s}\), let \(L=\|\mathbf x_r^{\rm old}\|\), \(m=m_r\). The exact conditional moment reference is

\[
A_{\mathrm{finite}}=N_sm\bigl(\mathbb E[\cos\theta]-1\bigr),
\]
\[
D_{\mathrm{finite}}=N_s\left[m^2\operatorname{Var}(\cos\theta)
+\frac{L^2-m^2}{N_s-1}\mathbb E[\sin^2\theta]\right]. \tag{6}
\]

This requires only Gaussian sampling of the new activations. It explains finite-size deviations without simulating entire new weight vectors. For an **exact-target** angle that depends on the new input norm, use the corresponding exact geometry rather than applying Eq. (6) without adjustment.

**For fully shared inputs**, a finite-\(N_s\) geometric reference can be generated from the joint Gaussian variables \(\mathbf u=W_t^\top\mathbf x^{\rm new}\) and \(y=\mathbf x^{\rm old}\cdot\mathbf x^{\rm new}\), together with the conditional length of the new input. This accounts for rank effects when \(N_d\) is comparable to or larger than \(N_s\). The full construction is in the earlier frozen-state protocol file, which may be reused for implementation details; the present file, not that earlier file, is the current figure plan.

Compare: (i) raw-fork estimate, (ii) leading theoretical coefficient, and (iii) finite-\(N_s\) geometric benchmark when needed. Agreement with (iii) but not (ii) is evidence of a controlled finite-size correction, not necessarily a failure of the model.

## 6. Architectures, rules, and update implementations

**Only user-supported architectures are permitted:**

\[
N_d\in\{1,2,3,4,5,10,12,15,20,30,50,60,75,100,150,300,400,750,1000,1500\},
\qquad N_s=60{,}000/N_d.
\]

The user supplies and fixes all actual **learning hyperparameters** for this figure: \(N_d\), \(n\), \(b\), \(\kappa\), eligibility threshold \(\lambda\) or \(L=b-\lambda\), and, for softmax, \(\eta\) and selectivity \(\beta_{\rm sm}\). Distinguish \(\beta_{\rm decay}\) in the legacy weight-decay implementation. Training stages, ages, fork count, seeds and input condition are **experimental sampling choices**, not fitted learning parameters; report them explicitly. Do not silently select or retune parameters from old grid searches.

**Rules included in the shared-color plots:** baseline independent threshold, n-WTA target, variable-\(u_{\mathrm{target}}\), interacting softmax, non-interacting softmax. A rule may be absent only if the user explicitly excludes it or its required hyperparameters have not been supplied; do not fill missing parameters with invented defaults.

**Primary implementation:** strict unit-norm spherical updates with the leading-angle update \(\theta=a_r/\sqrt{N_s}\). Amplitudes contain no decay parameter. For variable-target learning use the signed difference between adjusted target and activation, with fallback winner selection. Fixed-target n-WTA and threshold rules retain their upper-target gate.

**Optional original weight-decay rules:** add `threshold_decay`, `nwta_decay`, or `variable_target_decay` to the JSON `rules` list, supplying their own `b`, `kappa`, and **`beta_decay`** (the library's `beta`, distinct from softmax `beta_select`). Both example configurations include disabled entries with unset parameters. Enable and fill them explicitly; no grid-search values are inferred. These entries call `Neuron_Decay_NonInt`, `Neuron_Decay_Int`, and `Neuron_Decay_Int_Variable_n_min1` directly. They initialize Gaussian weights with scale \(1/\sqrt{N_s}\), preserve the original eligibility/upper-target/fallback behavior, and apply **no unit-norm renormalization**. No legacy softmax-decay variant exists in the current library.

For mask \(M_r\), target \(T_r\), and \(\beta=\beta_{\rm decay}\), the simulated update is exactly

\[
\Delta\mathbf w_r=M_r\left[\frac{\mathbf x_r}{\|\mathbf x_r\|^2}
\left(T_r-(1-\beta/N_s)u_r\right)-\frac{\beta}{N_s}\mathbf w_r\right].
\]

The **separate leading-order decay reference**, derived by expanding this update at fixed learning parameters, is

\[
a_r=M_r(T_r-u_r),\quad p_r=\mathbb E[M_r],\quad
C=\mathbb E[\mathbf a\mathbf a^\top],\quad F=\mathbb E[\mathbf a\mathbf u^\top],
\qquad H_{\rm decay}=FQ^+-\beta\operatorname{diag}(\mathbf p).
\]

For full sharing, \(Q=W W^\top\) includes the **actual squared norms on its diagonal**, and \(\mathbf A=H_{\rm decay}\mathbf m,\ D=C\). For independent inputs use \(u_r\sim N(0,q_{rr})\), \(A_r=(\mathbb E[a_ru_r]/q_{rr}-\beta p_r)m_r\), \(D_{rr}=\mathbb E[a_r^2]\), and zero leading cross-diffusion. Here `F` stores only the local diagonal needed for independent-input drift. For a zero-norm branch, the pseudoinverse defines the drift on the realizable overlap subspace. These predictions are evaluated **at each snapshot, even with independent inputs**; exchangeability pooling and reuse across snapshots would incorrectly discard norm differences. The finite-input-length and \(\beta u_r/N_s\) corrections remain in the exact simulation but are subleading in this reference. The existing finite-duration caveat still applies.

The decay formulas are not labeled R4/R8, and the spherical radial term \(-C_{rr}/2\) is not substituted for selected weight decay. Saved metadata records the implementation and reference, `beta_decay`, and minimum/mean/maximum frozen norms; `snapshot.npz` stores per-dendrite norms. \(R\) and Gram diagnostics for decay include changes in weight magnitude as well as alignment. In A/B, decay predictions appear as x markers at the corresponding overlaps because their slopes and diffusion levels vary by dendrite/snapshot. Spherical predictions remain lines; C/D/E compare each implementation against its own local reference.

**Architecture warning:** some supported configurations have \(N_d\ge N_s\) (e.g. \(N_d=300,N_s=200\)), where a full-sharing Gram matrix \(Q\) is necessarily rank deficient and fixed-\(N_d\) diffusion asymptotics can be inaccurate. Do not artificially force \(Q=I\) in such cases. Use the actual realized initial weights and, when appropriate, the finite-\(N_s\) reference.

## 7. Minimum outputs to save

For every independent training run: rule, fixed hyperparameters, update implementation, input condition, \(N_d,N_s\), seed, saved stage \(t\), and frozen \(Q_t\) or its sufficient stored representation. For each selected old pattern: \(\tau\), full \(\mathbf m\), old-pattern norms, and its identity/index in the training history. For each frozen-state fork batch: \(K\), conditional mean \(\overline{\Delta m}\), covariance of increments, batch/bootstrap error estimates. For the theory: \(D^{\mathrm G}\), or \(C(Q_t),F(Q_t),H(Q_t)\), plus numerical integration/sample uncertainty. Save point-level predicted/observed drift, diagonal diffusion, and chosen pairwise diffusion, so all rule-colored plots are generated from the same table.

**Figure 1 execution order:** verify configurable fork machinery with one user-selected supported architecture and one fixed parameter set; complete independent-input drift/diffusion; then full-sharing snapshot-specific drift/diffusion; compare fork lengths and add more stages and rules using the *same plotting conventions*. The separate Figure 2 protocol follows; it does not change any Figure 1 data or plotting specifications.

---

# Figure 2 — Determinism and local evolution of individual weight overlaps

**Scope:** Only the fully shared-input, strict unit-norm version of the learning rules is used in the primary Figure 2 test. Here all dendrites see the same newly drawn Gaussian input pattern and

\[
Q_{rs}(t)=\mathbf w_r(t)^\top\mathbf w_s(t),\qquad Q_{rr}=1.
\]

**Figure 2 questions, in this order:**

1. **Determinism first:** If multiple training runs start from the *exact same* weight matrix but use independent input-pattern sequences, how much do their individual \(Q_{rs}\) trajectories differ?
2. **Evolution equation second:** Does the measured finite-window change in each selected \(Q_{rs}\) agree with the right-hand side of R7, computed through **independent numerical Gaussian sampling** at the actual simulated Gram matrix?

There will be **no plots of \(R\), average pairwise correlations, or other collective order parameters**, and **no numerical integration of R6**. Figure layout in a manuscript is not a design constraint. The main output is interpretable numerical results.

## F2.1. Notation and theory

- \(N_d\): number of dendrites; \(N_s\): number of synapses per dendrite. Use the **supported** architectures and fixed, user-provided rule parameters from Figure 1. A reasonable existing starting option, *not a new required choice*, is \(N_d=100, N_s=600\).
- \(t\): number of learned input patterns. Figure 2 does not select or analyze old memory patterns, so there is no pattern-age variable.
- \(s=t/N_s\): the **theoretical evolution time** for R6/R7. In Figure 1, \(\tau\) denotes old-pattern age and \(\xi\) denotes additional-learning time; Figure 2 uses \(s\) to avoid overloading either.
- \(W_t^{(k)}\), \(Q_t^{(k)}\): trained weights and Gram matrix for training-sequence replicate \(k\), with **identical \(W_0\)** across all replicates \(k\) within a rule and architecture. Different learning rules should also use the same starting \(W_0\) when possible.
- \(L\): separation in *actual learned-pattern steps* between time points used to estimate the derivative; choose **tens of steps**, rather than adjacent steps. Do not confuse this \(L\) with the eligibility threshold \(L_{\rm elig}=b-\lambda\) in Figure 1.
- \(a_r(\mathbf u)\): actual angular-update amplitude for the chosen unit-norm rule, with \(u_r=\mathbf w_r^\top\mathbf x\) and \(\Delta\theta_r=a_r(\mathbf u)/\sqrt{N_s}\).

At any actual Gram matrix \(Q\), draw fresh Gaussian activation samples \(\mathbf u\sim\mathcal N(0,Q)\) and calculate, **without training or fitting**, the R5 moments

\[
\begin{aligned}
C_{rs}(Q)&=\mathbb E_{\mathbf u\sim\mathcal N(0,Q)}[a_r(\mathbf u)a_s(\mathbf u)],\\
F_{rs}(Q)&=\mathbb E_{\mathbf u\sim\mathcal N(0,Q)}[a_r(\mathbf u)u_s].
\end{aligned}
\]

The off-diagonal theoretical rate in R7 is

\[
\boxed{\begin{aligned}
G_{rs}(Q):=\frac{dQ_{rs}}{ds}
={}&C_{rs}(Q)+F_{rs}(Q)+F_{sr}(Q)\\
&-Q_{rs}\left[F_{rr}(Q)+F_{ss}(Q)
+\frac{C_{rr}(Q)+C_{ss}(Q)}{2}\right],\quad r\ne s.
\end{aligned}} \tag{F2.1}
\]

**Use R7 directly for selected pairs.** There is no need to compute \(H\), \(Q^+\), a full \(C\) or \(F\) matrix, or an integrated theoretical trajectory. For each pair only \(C_{rs},C_{rr},C_{ss},F_{rs},F_{sr},F_{rr},F_{ss}\) are needed. Gaussian samples can be generated as \(\mathbf u=W_t\mathbf z\), \(\mathbf z\sim\mathcal N(0,I_{N_s})\), with weight vectors stored as rows of \(W_t\), which gives exactly \(\operatorname{Cov}(\mathbf u)=Q_t\) and works even when \(N_d>N_s\). This is a separate, independent **numerical Gaussian integral**, not an estimate extracted from the learning patterns.

## F2.2. Experiment A — Determinism of individual \(Q_{rs}\) trajectories (**first results**)

**Training protocol**

1. For each rule and fixed user-supplied parameter set, draw **one** initial unit-norm weight matrix \(W_0\). Save it. Use the same matrix as the initial state for every training replicate and, preferably, for every rule being compared.
2. Start \(K\) copies of \(W_0\). Train every copy through the same prespecified number of learning steps, using a **different independent sequence of Gaussian input patterns** for each copy. The connectivity is maximally shared among dendrites *within each input pattern*, but successive patterns and independent training replicates are independent.
3. Record selected off-diagonal \(Q_{rs}^{(k)}(t)\) at a common list of checkpoints, including \(t=0\). The values at \(t=0\) must be **exactly identical** across the copies for each pair; differing initial values indicate an implementation bug.
4. Start with approximately **10–20 independent training streams** if practical, and show the number of replicates in the saved metadata. The number is a sampling choice, not a fitted learning parameter.

**Pair selection**

- Prespecify a small number of distinct pairs, e.g. **four to six**, *once* using \(W_0\) and random selection—**not** using their later trajectories, agreement with theory, or the sign of later drifts.
- If it makes the comparison informative, choose from different **initial** \(Q_{rs}(0)\) ranges (positive, near zero, negative), with a random-pair control. Retain their exact index identities \((r,s)\) across all rules and replicates. Selection by initial overlap is acceptable because \(W_0\) is identical and frozen before learning.
- If the selected pairs have little observable separation, add a few other *predeclared/random* pairs; do not post-select visually favorable trajectories.

**Plot A — selected-pair trajectories and run-to-run spread**

- **Small panel per selected pair \((r,s)\)**; panel title identifies that pair and its initial \(Q_{rs}(0)\).
- **X-axis:** \(t\) or \(s=t/N_s\) (choose one and use it consistently).
- **Y-axis:** individual overlap \(Q_{rs}(t)\). Use the **same y-axis limits across pair panels** if this does not hide the dynamics; otherwise label the differing ranges prominently.
- **Colors:** exactly the Figure 1 rule colors. Show **all enabled rules together** on each pair's axes.
- **Per rule:** draw individual replicate trajectories as very thin, low-opacity lines; draw their average as a thicker solid line of the **same color**; optionally show a lightly shaded \(\pm1\) sample standard-deviation band. Avoid making each replicate a new color.
- **No theoretical ODE curve:** this plot is solely a test of concentration across independent random training sequences from the same \(W_0\).

**Numerical summary, alongside the trajectories or in a compact inset:** for each chosen pair and rule, compute

\[
\overline Q_{rs}(t)=\frac{1}{K}\sum_{k=1}^KQ_{rs}^{(k)}(t),\qquad
S_{rs}(t)=\sqrt{\frac{1}{K-1}\sum_{k=1}^K\left(Q_{rs}^{(k)}(t)-\overline Q_{rs}(t)\right)^2}.
\]

A small \(S_{rs}(t)\) relative to the actual drift of \(Q_{rs}\) supports approximate determinism at that architecture. **Finite \(N_s\) does not imply exactly identical paths**, and the theoretical deterministic statement is a large-\(N_s\) limit at fixed \(N_d\). The comparison can therefore show approximate concentration without requiring zero variance. Do not infer a particular finite-size scaling law from a single supported architecture.

## F2.3. Experiment B — Measured versus theoretical local evolution (**second results**)

**Empirical derivative over tens of steps**

For each replicate \(k\), selected pair \((r,s)\), and interval \([t,t+L]\), estimate

\[
\boxed{
\widehat G_{rs}^{(k)}(t;L)
=\frac{N_s}{L}\left[Q_{rs}^{(k)}(t+L)-Q_{rs}^{(k)}(t)\right].
} \tag{F2.2}
\]

A practical **initial choice is \(L=50\)** learning steps, with **\(L=20\)** as a finite-window sensitivity check. These are **measurement-window choices**, not model or learning-rule hyperparameters. They can be changed if the resulting noise or interval bias warrants it. Prefer nonoverlapping derivative windows for statistical error calculations; overlapping windows can still be plotted but their estimates are correlated.

**Numerical theory at the same trained state**

- For each actual replicate separately, use the **midpoint** Gram matrix \(Q^{(k)}_{t+L/2}\) (if \(L\) even) to evaluate \(G_{rs}^{\rm th}=G_{rs}(Q^{(k)}_{t+L/2})\) through R5/R7 using fresh Gaussian samples. This approximates the interval-averaged theoretical rate without integrating R6.
- For odd \(L\), use the nearest recorded midpoint, or average predictions from two symmetrically placed snapshots; record the convention.
- If finite-window bias is visible, sample \(G_{rs}(Q)\) at several **actual intermediate simulation snapshots** and average using a prespecified quadrature rule (e.g. trapezoid or midpoint). **This is numerical averaging along an observed simulation interval, not numerical integration of the self-consistent R6 ODE.**
- Do **not** replace \(Q^{(k)}\) by a mean matrix across training replicates before evaluating \(G\): R7 is nonlinear in \(Q\). Likewise, avoid averaging \(Q\) over distant training steps.
- The primary model uses exact unit-norm updates. Legacy decay rules may only be included in this direct R7 test if their **separately derived non-spherical overlap equation** is used. Otherwise they are out of scope for Figure 2, even if enabled in Figure 1.

**Plot B — direct R7 test**

- **X-axis:** \(G_{rs}(Q^{(k)}_{t+L/2})\), or the prespecified intermediate-snapshot average.
- **Y-axis:** \(\widehat G_{rs}^{(k)}(t;L)\).
- **Black dashed reference:** \(y=x\), with **no fitted shift, vertical scale, or time rescaling**.
- **Rules:** all enabled spherical rules on the **same axes**, with colors reused from Figure 1. A marker style may distinguish chosen \((r,s)\) pairs, and opacity may indicate individual time windows; **colors are not reassigned to pairs or times**.
- Include multiple training intervals to span early, middle, and later states. Avoid selecting only time windows with an evidently favorable comparison.
- Because pointwise finite differences may be noisy, also show one transparent, uncertainty-bearing summary per pair and interval or group of nearby intervals. Compute the replicate-averaged empirical slope and the replicate-averaged **individually evaluated** theoretical slope separately:

\[
\overline {\widehat G}_{rs}(t;L)=\frac1K\sum_k\widehat G_{rs}^{(k)}(t;L),\qquad
\overline G^{\rm th}_{rs}(t;L)=\frac1K\sum_kG_{rs}(Q^{(k)}_{t+L/2}).
\]

Do not substitute \(G(\overline Q)\) for the last expression. Compute replicate-level uncertainty of the empirical mean and independent Gaussian-sampling uncertainty of the theoretical coefficient; do not treat multiple windows from the same training run as independent replicates.

**Optional readability plot:** if combined scatter is too dense, add a **small plot per selected pair** with exactly the same axes and the same rule colors, plus the black \(y=x\) reference. This is only a breakdown of the R7 comparison, **not** an additional observable or figure proposal.

**Interpretation of disagreements**

- **Wide trajectory spread but matching average drift:** finite-\(N_s\) noise is still visible, while the mean local R7 evolution may be correct.
- **Tight replicate trajectories but systematic deviation from R7:** the dynamics may be reproducible while the proposed large-\(N_s\) coefficient is biased (or the simulated update does not match the theoretical one).
- **Window dependence (\(L=20\) versus \(50\)):** likely a mixture of finite-window bias and statistical error; compare midpoint and intermediate-snapshot theory before modifying the theory.
- **Theory Monte Carlo error comparable to derivative signal:** increase *independent Gaussian samples only*; do not smooth or calibrate the predicted right-hand side to agree with the simulation.

## F2.4. Execution conditions, outputs, and sequencing

**Keep the same user-selected learning parameters and supported architecture policy as Figure 1.** No new grid search, no inferred optimum, and no retuning across training stages. Each learning rule has its own supplied fixed parameters, and the selection of trained time checkpoints is prespecified. The experiment is **fully shared inputs only** because R6/R7 close in the ordinary Gram matrix in that case; with partially shared axonal supports, additional restricted-overlap quantities are needed.

Suggested saved outputs (one run directory for each user-specified architecture/condition and immutable parameter configuration):

- **Initialization:** the exact shared \(W_0\), chosen pair indices, initial \(Q_{rs}(0)\), implementation and all rule parameters.
- **Training:** weights or sufficient per-checkpoint state for each independent training-sequence seed, plus selected pair trajectories \(Q_{rs}^{(k)}(t)\), common checkpoint times, and seeds.
- **Determinism:** \(\overline Q_{rs}(t)\), \(S_{rs}(t)\), number of training replicates, and trajectory plots.
- **R7 test:** window start/end/midpoint, \(L\), empirical slope, Gaussian-theory slope from the **individual midpoint \(Q\)**, independent numerical-integration Monte Carlo error, and replicate mean/SE.
- **Provenance:** fixed training inputs distribution, \(N_d,N_s\), code revision, numerical precision, Gaussian-sampling count/seed, and the chosen colors/plot filters.

**Run order:** (1) set and save the single common \(W_0\) and selected pairs; (2) generate the independent learning sequences, recording pairwise trajectories; (3) inspect trajectory concentration (Experiment A); (4) use the **same saved training runs** to compute finite-window slopes and numerical R5/R7 estimates (Experiment B); (5) compare \(L=20\) and \(50\) before adding further conditions. No second training simulation is needed for the local derivative test if the necessary checkpoints are saved.

**Out of scope for Figure 2:** any plot of \(R\) or another collective order parameter; numerical integration of R6; memory-age, familiarity response or capacity panels; a new architecture or hyperparameter sweep; any figure beyond Figures 1 and 2.
