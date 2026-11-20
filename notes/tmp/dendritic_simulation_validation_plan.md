# Dendritic computation — simulation validation plan

**Current scope: Figure 1 only.** This file replaces the previous multi-figure proposals. Future figures are intentionally **not specified**. The goal is to obtain and inspect simulation results, not to design a manuscript layout.

**Figure 1 question:** Do the stochastic drift and diffusion coefficients of an old memory's dendritic overlaps agree with the theoretical predictions (R4 for independent dendritic inputs, R8 for fully shared inputs), when measured locally at *actual trained weights*?

**Core design:** Train normally to time \(t\), freeze \(W_t\), choose real patterns learned at different ages \(\tau\), and generate many independent hypothetical **one-step** updates starting from that *same* \(W_t\). Compare the resulting conditional moments with numerical Gaussian expectations evaluated at the actual frozen state.

## 1. Notation and theoretical predictions

- \(N_d\): number of dendrites. \(N_s\): synapses per dendrite. In the current codebase, \(N_dN_s=60{,}000\).
- \(t\): training step of the frozen weights \(W_t\). \(\tau\): age of a selected stored pattern at that step. Fix a convention for whether age zero is immediately after learning.
- \(u_r=\mathbf w_r\cdot\mathbf x_r^{\mathrm{new}}\): **local activation** for a newly presented input, not the stored overlap.
- \(m_r=m_r(t;\tau)=\mathbf w_r(t)\cdot\mathbf x_r^{\mathrm{old}}\): current overlap with a particular stored pattern of age \(\tau\). \(\mathbf m\) is its full vector across dendrites.
- \(a_r(\mathbf u)\): rule-dependent angular learning amplitude, with angle \(\theta_r=a_r(\mathbf u)/\sqrt{N_s}\) for the asymptotic spherical update.
- \(Q_t\): actual Gram matrix of the frozen unit weight vectors in a common axonal coordinate system; for full sharing, \((Q_t)_{rs}=\mathbf w_r(t)\cdot\mathbf w_s(t)\).
- \(\xi=t_{\mathrm{additional}}/N_s\): *rescaled additional-learning time* used in the diffusion theory. **Do not confuse \(\xi\) with the pattern age \(\tau\).**

For \(K\) independently sampled new inputs, perform one hypothetical learning update from \(W_t\) for each input. For a chosen stored pattern, calculate

\[
\Delta m_r^{(k)}=m_r^{\prime(k)}-m_r,\quad k=1,\dots,K.
\]

Estimate the **conditional rescaled drift** and **conditional diffusion covariance**:

\[
\boxed{\widehat A_r=N_s\,\overline{\Delta m_r},\qquad
\widehat D_{rs}=N_s\,\widehat{\operatorname{Cov}}_k(\Delta m_r^{(k)},\Delta m_s^{(k)})}. \tag{1}
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

**Step 2 — choose old patterns.** For a fixed \(t\), select actual previously learned patterns at multiple ages \(\tau<t\) spanning recent, intermediate and old memories. Compute \(\mathbf m(t;\tau)\) for each. Across patterns and dendrites, this should give a broad range of \(m_r\). Store the **full** \(\mathbf m\) for the correlated-input theory.

**Step 3 — generate hypothetical one-step forks.** Draw \(K\) independent new inputs from the same distribution and with the same axonal connectivity as ordinary training. For each draw, perform **one** update from the *unchanged* frozen \(W_t\), obtaining \(W_t^{\prime(k)}\), and compute \(\Delta\mathbf m^{(k)}\) for all selected old patterns. Discard that hypothetical state and restart the next fork from the original \(W_t\). Reuse the same forked updates across the selected old patterns at that snapshot for efficiency.

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

A common black dashed identity line \(y=x\) is a **reference**, not a learning rule. For a single plot, **all five rules may appear simultaneously**; if crowded, use small/transparent points and emphasized median-with-uncertainty markers, *not* separate colors for time or a new rule-specific layout. Avoid choosing different axis ranges across rules.

### Plot A — independent-input drift vs old-pattern overlap

- **X:** observed \(m_r(t;\tau)\).
- **Y:** empirical \(\widehat A_r=N_s\overline{\Delta m_r}\).
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

Single-step drift is \(O(1/N_s)\) while one-step noise is typically \(O(N_s^{-1/2})\). With \(D\) a representative diffusion coefficient,

\[
\operatorname{SE}(\widehat A_r)\approx\sqrt{\frac{N_sD}{K}},
\qquad
\operatorname{SE}(\widehat D_{rr})/D\approx\sqrt{\frac{2}{K-1}}
\]

for approximately Gaussian increments. Therefore **do not prescribe an arbitrary fixed \(K\)** for every state. Pilot raw fork standard errors and increase \(K\) adaptively; use the same precision target across learning rules rather than the same raw number of forks, if their update amplitudes differ substantially. Near \(m_r=0\), the drift is intrinsically difficult to resolve; keep those data for diffusion/residual checks but do not divide drift by \(m_r\).

Variance-reduced drift estimators are helpful, but keep a direct raw-fork check:

- **Independent inputs, angular update:** the new-pattern tangent direction is independent of the amplitudes, giving the exact conditional mean \(\mathbb E[\Delta m_r\mid\mathbf u]=m_r[\cos(a_r(\mathbf u)/\sqrt{N_s})-1]\). This removes the zero-mean perpendicular projection noise and gives an accurate finite-angle drift estimate by Gaussian sampling.
- **Fully shared inputs:** Gaussian conditioning gives \(\mathbb E[y\mid\mathbf u]=\mathbf m^\top Q_t^+\mathbf u\), where \(y=\mathbf x^{\rm old}\cdot\mathbf x^{\rm new}\). Use this in the **leading-order** overlap update to reduce variance; substituting it directly into the exact finite-angle denominator is not in general an exact conditional expectation.
- Forks are independent *within a frozen state*, but points sharing \(W_t\), its old patterns, and its forked inputs are statistically dependent. For figure-wide uncertainty, replicate independent training seeds and use clustered uncertainty estimates; do not mistake the number of scatter points for the number of independent neurons.

Numerical Gaussian-moment estimates should use independent random streams from the forked updates; record numerical MC confidence intervals alongside empirical uncertainties. No fitted multiplicative correction to theoretical drift, diffusion, or the time axis is allowed. Optional regression slopes (measured vs predicted) are **diagnostics**, not fitted theory coefficients.

## 5. Large-\(N_s\) benchmark versus exact finite-\(N_s\) geometry

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

**Primary implementation:** strict unit-norm spherical updates (the model behind R4/R8). For target-based rules choose either the leading-angle update \(\theta=a_r/\sqrt{N_s}\) or an exact-sphere target-hitting rotation, and *label which*; these coincide to leading order but differ at finite \(N_s\). The original weight-decay implementation is not assumed to obey the same finite-\(N_s\) coefficients. If included, show it as a **second marker style or line style within the same rule color**, record actual norms and angular increments, and do not pool its points with the spherical-update points.

**Architecture warning:** some supported configurations have \(N_d\ge N_s\) (e.g. \(N_d=300,N_s=200\)), where a full-sharing Gram matrix \(Q\) is necessarily rank deficient and fixed-\(N_d\) diffusion asymptotics can be inaccurate. Do not artificially force \(Q=I\) in such cases. Use the actual realized initial weights and, when appropriate, the finite-\(N_s\) reference.

## 7. Minimum outputs to save

For every independent training run: rule, fixed hyperparameters, update implementation, input condition, \(N_d,N_s\), seed, saved stage \(t\), and frozen \(Q_t\) or its sufficient stored representation. For each selected old pattern: \(\tau\), full \(\mathbf m\), old-pattern norms, and its identity/index in the training history. For each frozen-state fork batch: \(K\), conditional mean \(\overline{\Delta m}\), covariance of increments, batch/bootstrap error estimates. For the theory: \(D^{\mathrm G}\), or \(C(Q_t),F(Q_t),H(Q_t)\), plus numerical integration/sample uncertainty. Save point-level predicted/observed drift, diagonal diffusion, and chosen pairwise diffusion, so all rule-colored plots are generated from the same table.

**Execution order:** verify the one-step-fork machinery with one user-selected supported architecture and one fixed parameter set; complete independent-input drift/diffusion; then full-sharing snapshot-specific drift/diffusion; add more stages and rules using the *same plotting conventions*. Only after inspecting these results decide what the next figure should test.
