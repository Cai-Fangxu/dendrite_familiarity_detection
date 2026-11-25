# Weak-correlation theory for interacting and non-interacting dendritic learning

**Main results and physical interpretation**  
Companion file: [`dendritic_weak_correlation_proofs.md`](dendritic_weak_correlation_proofs.md)  
**Scope:** Gaussian input patterns; exact unit-norm synaptic weights; the limit of large numbers of synapses per dendrite, $N_s\to\infty$, with $N_d$ fixed unless a further limit is explicitly indicated. Interacting rules may be nonsmooth, including hard winner-take-all selection. The focus is **weak but nonzero weight correlations**, not an assumption that all pairwise overlaps are equal.

## 1. Model and notation

Each neuron has $N_d$ dendrites, with $N_s$ synapses per dendrite. In the common presynaptic axonal space $\mathbb R^{N_a}$, dendrite $r$ has a diagonal, binary connectivity projector $P_r$ satisfying

$$
P_r^2=P_r=P_r^\top,\qquad \operatorname{tr}P_r=N_s,
\qquad P_r\mathbf w_r=\mathbf w_r,\qquad \|\mathbf w_r\|=1.
$$

A new pattern is $\mathbf x\sim\mathcal N(0,I_{N_a})$; its **local dendritic activation** is

$$u_r=\mathbf w_r^\top P_r\mathbf x=\mathbf w_r^\top\mathbf x.$$

The inputs are independent across dendrites if their synaptic supports do not overlap. They are maximally shared when all $P_r$ are identical; partial sharing corresponds to overlapping supports. In the manuscript's random-connectivity construction, $c=N_a/N_s$ and the typical sharing fraction is $\alpha_{rs}\simeq 1/c$. For the latter define the **anatomical pairwise sharing fraction**

$$\boxed{\alpha_{rs}=\frac{\operatorname{tr}(P_rP_s)}{N_s}\in[0,1].} \tag{R0}$$

The **weight-overlap matrix** and **collective alignment** are

$$
Q_{rs}=\mathbf w_r^\top\mathbf w_s,\quad Q_{rr}=1,
\qquad
\boxed{R=\frac1{N_d}\sum_{r,s=1}^{N_d}Q_{rs}
=\frac1{N_d}\left\|\sum_{r=1}^{N_d}\mathbf w_r\right\|^2.}
$$

$R=1$ for initially uncorrelated weight directions (though opposing correlations can also cancel to give $R=1$), $R>1$ for net alignment, and $R<1$ for net anticorrelation. Define $E=Q-I_{N_d}$; **weak weight correlations** means $\epsilon=\max_{r\ne s}|E_{rs}|\ll1$, with $N_d$ fixed when the perturbation expansion is used. Individual $Q_{rs}$ need not be equal.

The learning rule is an angular rotation towards the new input, with

$$
\boxed{\Delta\theta_r=\frac{a_r(\mathbf u)}{\sqrt{N_s}},\qquad
\mathbf u=(u_1,\ldots,u_{N_d})^\top.} \tag{R1}
$$

Here $a_r(\mathbf u)$ is the **learning amplitude**, equal to the leading-order increment of local activation: $u_r'-u_r=a_r(\mathbf u)+o(1)$. The update always remains within dendrite $r$'s synaptic subspace and preserves its weight norm exactly. Time $t$ counts subsequently learned patterns; $\tau=t/N_s$ is the diffusion timescale. Denote by $\langle\cdot\rangle_Q$ expectation with $\mathbf u\sim\mathcal N(0,Q)$, and by $\langle\cdot\rangle_0$ expectation with independent standard Gaussian local activations $\mathbf z=(z_1,\ldots,z_{N_d})$.

**General rule class.** The amplitude $a_r(\mathbf u)$ may be any measurable function with sufficiently many finite Gaussian moments. An **interacting** rule can depend on all $u_s$; a **non-interacting** rule has $a_r=f(u_r)$. A rule is **permutation-equivariant** if relabeling dendrites relabels its output amplitudes. The original rules in the manuscript have the following leading-order angular representations:

$$
\begin{aligned}
\text{Independent threshold:}\quad
&a_r=\mathbf1\{u_r>b-\lambda\}[b+\kappa-u_r]_+,\\
\text{$n$-WTA target:}\quad
&a_r=\mathbf1\{r\in\operatorname{Top}_n(\mathbf u)\}[b+\kappa-u_r]_+,\\
\text{Variable target:}\quad
&a_r=\mathbf1\{r\in\mathcal S(\mathbf u)\}
\left[b+\frac{n\kappa}{\hat n}-u_r\right].
\end{aligned} \tag{R2}
$$

Here $b$ is the dendritic output threshold, $\lambda$ is the eligibility margin, $\kappa$ the original target activation margin, $n$ the nominal number of winners, $\hat n=|\mathcal S|$, $[y]_+=\max(y,0)$, and $\mathcal S$ is the above-eligibility-threshold set, with the largest-activation dendrite selected if that set is empty. The variable-target amplitude is **signed**, matching the existing implementation: selected branches above their adjusted target may move downward. The fixed-target rules retain their upper-target gate. Ties may be broken in a permutation-symmetric way. These are large-$N_s$ equivalents of the target updates, not identities at finite $N_s$; the original target-based rules **do not generally have a constant sum of learning amplitudes**.

## 2. Universal uncorrelated-input forgetting law

For any rule $a_r(\mathbf u)$, including interacting and hard-selection rules, define its single-dendrite **plasticity variance parameter**

$$\boxed{D_r=\langle a_r(\mathbf z)^2\rangle_0.} \tag{R3}$$

For an old pattern $\mathbf x^{(0)}$, denote the overlap at pattern age $t$ by $m_r(t)=\mathbf w_r(t)^\top\mathbf x_r^{(0)}$. With **disjoint input supports**, the large-$N_s$ limit yields

$$
\boxed{dm_r=-\frac{D_r}{2}m_r\,d\tau+\sqrt{D_r}\,dB_r(\tau),}
\qquad
\boxed{P(m_r(\tau)=m\mid m_r(0)=m_0)
\simeq\mathcal N\!\left(m; m_0e^{-D_r\tau/2},\ 1-e^{-D_r\tau}\right).} \tag{R4}
$$

$B_r$ is standard Brownian motion and $\mathcal N(m;\mu,V)$ denotes a Gaussian density at $m$ with mean $\mu$ and variance $V$. Thus **the rule enters single-dendrite forgetting only through $D_r$**, even if the amplitudes are mutually coupled across dendrites. This does not by itself determine the neuronal output variance or memory capacity.

For example, for target-based $n$-WTA, $D_r$ is a **one-dimensional Gaussian order-statistic integral** (given explicitly in the proofs). The variable-target rule can be evaluated by finite-$N_d$ Gaussian sampling without simulating synapses.

## 3. Maximally shared inputs: universal closed Gram-matrix theory

For full sharing ($P_r=I_{N_s}$ after identifying common coordinates), introduce the **co-plasticity matrix** $C(Q)$, the **activation–plasticity matrix** $F(Q)$, and the **forgetting drift matrix** $H(Q)$:

$$
\boxed{\begin{aligned}
C_{rs}(Q)&=\langle a_r(\mathbf u)a_s(\mathbf u)\rangle_Q,\\
F_{rs}(Q)&=\langle a_r(\mathbf u)u_s\rangle_Q,\\
J(Q)&=F(Q)Q^+,\\
H(Q)&=J(Q)-\operatorname{diag}\!\left(F_{11}+\tfrac12C_{11},\ldots,F_{N_dN_d}+\tfrac12C_{N_dN_d}\right).
\end{aligned}} \tag{R5}
$$

$Q^+$ denotes the Moore–Penrose pseudoinverse; for invertible $Q$ it is $Q^{-1}$. $C_{rs}$ quantifies simultaneous changes on dendrites $r,s$; $F_{rs}$ quantifies how plasticity on dendrite $r$ is associated with activation on dendrite $s$. Both are ordinary Gaussian expectations, **not derivatives of $a_r$**.

The deterministic overlap evolution, to leading order in $1/N_s$, is

$$\boxed{\frac{dQ}{d\tau}=HQ+QH^\top+C.} \tag{R6}$$

Equivalently, for $r\ne s$,

$$
\boxed{\frac{dQ_{rs}}{d\tau}
=C_{rs}+F_{rs}+F_{sr}
-Q_{rs}\!\left(F_{rr}+F_{ss}+\frac{C_{rr}+C_{ss}}2\right).} \tag{R7}
$$

The old-pattern overlap vector $\mathbf m(\tau)$ obeys a linear diffusion process conditioned on $Q(0)$ and $\mathbf m(0)$. Here $\mathbf B_\tau$ is a vector of independent standard Brownian motions and $C^{1/2}$ is any matrix square root satisfying $C^{1/2}(C^{1/2})^\top=C$:

$$
\boxed{d\mathbf m=H(Q)\mathbf m\,d\tau+C(Q)^{1/2}d\mathbf B_\tau.} \tag{R8}
$$

Let $\dot\Phi=H(Q(\tau))\Phi$ and $\Phi(0)=I_{N_d}$. Then

$$
\boxed{\mathbf m(\tau)\mid\mathbf m_0,Q(0)
\simeq\mathcal N\!\left(\Phi(\tau)\mathbf m_0,\ 
Q(\tau)-\Phi(\tau)Q(0)\Phi(\tau)^\top\right).} \tag{R9}
$$

This is a **finite-$N_d$ closure**, requiring no equality of off-diagonal $Q_{rs}$. It applies to hard $n$-WTA or variable-target selection as long as the Gaussian moments exist. The law conditioned only on one $m_r(0)$ can be a mixture, because the other initial overlaps are not specified.

## 4. Weak but nonzero overlaps: universal Gaussian perturbation

For any sufficiently integrable $G(\mathbf u)$,

$$
\boxed{\langle G\rangle_{I+E}
=\langle G\rangle_0+\sum_{a<b}E_{ab}\langle G(\mathbf z)z_az_b\rangle_0+O(\epsilon^2).} \tag{R10}
$$

This is a **density expansion**, valid for discontinuous $G$: no differentiability of the learning rule is needed. All expectations on the right use independent Gaussian variables.

### 4.1 Non-interacting learning, arbitrary anatomical sharing

For $a_r=f(u_r)$ define

$$
\boxed{\mu=\mathbb E[f(Z)],\quad D=\mathbb E[f(Z)^2],\quad
\chi=\mathbb E[Zf(Z)],\quad Z\sim\mathcal N(0,1).} \tag{R11}
$$

$\mu$ measures mean potentiation, $D$ its squared magnitude (individual forgetting/interference), and $\chi$ its sensitivity to local activation in a covariance sense.

For **any fixed connectivity masks**, at leading order in $N_s^{-1}$ the pairwise equation is closed:

$$
\boxed{\dot Q_{rs}=\alpha_{rs}K(Q_{rs})-DQ_{rs},\qquad
K(q)=\mathbb E[f(Z_1)f(Z_2)],\quad\operatorname{Corr}(Z_1,Z_2)=q,\quad r\ne s.} \tag{R12}
$$

The weak-correlation expansion is

$$
\boxed{\dot Q_{rs}=\alpha_{rs}\mu^2-\bigl(D-\alpha_{rs}\chi^2\bigr)Q_{rs}+O(Q_{rs}^2).} \tag{R13}
$$

For **homogeneous** pairwise anatomical sharing $\alpha_{rs}=\alpha$,

$$
\boxed{\dot R=(N_d-1)\alpha\mu^2-
(D-\alpha\chi^2)(R-1)+O(\epsilon^2).} \tag{R14}
$$

The linearized trajectory is $R(\tau)\simeq 1+\frac{(N_d-1)\alpha\mu^2}{D-\alpha\chi^2} + \left[R(0)-1-\frac{(N_d-1)\alpha\mu^2}{D-\alpha\chi^2}\right]e^{-(D-\alpha\chi^2)\tau}$ when the denominator is nonzero. These do not require statistical exchangeability of the actual weight-overlap matrix. If $D-\alpha\chi^2>0$, the linearized stationary pairwise overlap is $\alpha\mu^2/(D-\alpha\chi^2)$, **provided it remains small**. The initial growth from zero is always nonnegative for $f\geq0$.

### 4.2 Interacting learning, maximally shared inputs

Assume **permutation-equivariant** $a_r(\mathbf u)$. For two distinct indices 1,2 define

$$
\begin{aligned}
C_o&=\langle a_1a_2\rangle_0,& F_o&=\langle a_1z_2\rangle_0,\\
F_d&=\langle a_1z_1\rangle_0,&D&=\langle a_1^2\rangle_0,\\
G_{12}(\mathbf z)&=a_1a_2+a_1z_2+a_2z_1,
&\mathcal T(\mathbf z)&=\sum_{a<b}z_az_b.
\end{aligned}
$$

The **initial correlation-generation coefficient** and **linear feedback coefficient** are

$$
\boxed{B=C_o+2F_o,\qquad
\Lambda=\langle G_{12}\mathcal T\rangle_0-2F_d-D.} \tag{R15}
$$

$C_o$ is simultaneous plasticity; $F_o$ is the cross-dendrite activity–selection coupling (possibly negative under competition); $B$ is their net pairwise correlation drive from $Q=I$. $\Lambda$ quantifies how that drive responds to pre-existing weak collective alignment.

The collective order parameter satisfies

$$
\boxed{\dot R=(N_d-1)B+\Lambda(R-1)+O(\epsilon^2).} \tag{R16}
$$

For $\Lambda\ne0$, its linearized solution is $R(\tau)\simeq 1-\frac{(N_d-1)B}{\Lambda}+\left[R(0)-1+\frac{(N_d-1)B}{\Lambda}\right]e^{\Lambda\tau}$, valid only while the overlaps remain weak. **Crucially, this does not impose $Q_{rs}=q$**. Only the learning rule is permutation-symmetric; the individual small overlaps may differ. A negative $B$ means initial anticorrelation, and $\Lambda<0$ gives linear damping of a collective perturbation. Neither condition alone proves improved memory capacity.

### 4.3 Interacting learning, partially shared inputs

Under **independently and isotropically initialized** weights, the initial large-$N_s$ correlation generation satisfies

$$
\boxed{\left.\dot Q_{rs}\right|_{Q=I}=\alpha_{rs}B.} \tag{R17}
$$

If all pairwise sharing fractions equal $\alpha$, then $\left.\dot R\right|_{Q=I}=(N_d-1)\alpha B$.

**Important boundary of the theory:** weak ordinary $Q_{rs}$ does not alone close subsequent partially shared interacting dynamics. They also involve *connectivity-restricted overlaps* $T_{srk}=\mathbf w_s^\top P_r\mathbf w_k$ (see the proof). Under an **additional, explicitly conditional random-mask mixing assumption**, with $\alpha_{rs}=\alpha$, the collective equation becomes

$$
\boxed{\dot R\simeq(N_d-1)\alpha B+
\underbrace{\bigl[\alpha(\Lambda+D)-D\bigr]}_{\Lambda_\alpha}(R-1)
+O(\epsilon^2)+\text{mask-mixing error}.} \tag{R18}
$$

The $\alpha=1$ limit is (R16), and the non-interacting specialization reproduces (R14). **(R18) is a closure hypothesis, not a theorem from weak $Q$ alone**; measurements of restricted overlaps should accompany its use.

## 5. Constant-budget corollary (optional rule subclass)

For fully shared inputs, consider a rule satisfying $\sum_r a_r(\mathbf u)=\eta$ for every pattern, and set $k_r=\sum_sQ_{rs}$ (the $r$th row sum). The collective order parameter has the exact leading-order large-$N_s$ identity

$$
\boxed{\dot R=\frac{\eta^2}{N_d}-\frac1{N_d}\sum_r k_r
\left\langle2a_r u_r+a_r^2\right\rangle_Q.} \tag{R19}
$$

This does **not** require weak weight correlations, although it is not autonomous in $R$ alone. For an equivariant constant-budget rule define

$$
K_0=\langle2a_1z_1+a_1^2\rangle_0,\qquad
K_1=\langle(2a_1z_1+a_1^2)\mathcal T\rangle_0.
$$

Then its weak-correlation closure is

$$
\boxed{\dot R=\frac{\eta^2}{N_d}-K_0-
\left(K_0+\frac{K_1}{N_d-1}\right)(R-1)+O(\epsilon^2).} \tag{R20}
$$

For **fixed-amplitude hard $n$-WTA**, $a_r=(\eta/n)\mathbf1\{r\in\operatorname{Top}_n(\mathbf u)\}$, let $Z_{(1)}\geq\cdots\geq Z_{(N_d)}$ be descending order statistics of independent $\mathcal N(0,1)$ variables. The initial slope is

$$
\boxed{\left.\dot R\right|_{Q=I}
=\frac1{N_d}\left[\eta^2\left(1-\frac1n\right)
-\frac{2\eta}{n}\mathbb E\sum_{j=1}^n Z_{(j)}\right].} \tag{R21}
$$

For $n=1$ and $N_d>1$, this slope is strictly negative. This corollary concerns **fixed-amplitude $n$-WTA**, not the manuscript's original fixed-*target* $n$-WTA: the latter is handled by (R15)–(R18) without imposing a constant budget.

## 6. Practical interpretation and limits

| Quantity | Physical meaning | How to obtain it |
|---|---|---|
| $D=\langle a_1^2\rangle_0$ | Local interference / single-dendrite diffusion | Sample independent Gaussian activations and apply the rule |
| $C_o=\langle a_1a_2\rangle_0$ | Co-plasticity of two dendrites | Same Gaussian samples |
| $F_o=\langle a_1z_2\rangle_0$ | Coupling between another branch's activation and local learning | Same Gaussian samples |
| $B=C_o+2F_o$ | Initial weight-correlation drift per shared-axon fraction | Combine the preceding moments |
| $\Lambda$ | Linear feedback of prior weak correlations on collective alignment | Gaussian expectation involving $\mathcal T$ |
| $\alpha_{rs}$ | Fraction of axons shared by a pair of dendrites | Synaptic connectivity masks |
| $R$ | Squared collective synaptic-weight magnitude per dendrite | Weight vectors or Gram matrix |
| $T_{srk}$ | Weight overlap restricted to a third dendrite's connectivity | Masks and weight vectors |

The hierarchy of claims is deliberate: **(R4), (R6)–(R9), (R12), (R17), and (R19)** follow from the large-$N_s$ spherical dynamics under their stated input assumptions; **(R10), (R13)–(R16), (R20)** are controlled first-order weak-correlation expansions at fixed $N_d$; **(R18)** additionally assumes random-mask mixing. A subsequent $N_d\to\infty$ limit needs uniform bounds on the expansion errors and should not be inferred from these fixed-$N_d$ results without verification.

These calculations characterize learning-induced correlations and forgetting. **They do not, by themselves, determine familiarity-detection capacity**: capacity also depends on the initial post-learning response distribution, dendritic nonlinearity, and the joint (not merely marginal) distribution of old-pattern overlaps.
