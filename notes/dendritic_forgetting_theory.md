# Single-Dendrite Memory Forgetting Under Interacting and Non-Interacting Plasticity

**Theoretical note — large synapse number, finite dendrite number**

This note derives the evolution of the overlap of a stored input with an individual dendrite under two angular-plasticity rules and two input-correlation regimes:

1. Non-interacting plasticity, independent dendritic inputs.
2. Interacting plasticity, independent dendritic inputs.
3. Non-interacting plasticity, maximally shared dendritic inputs.
4. Interacting plasticity, maximally shared dendritic inputs.

**Scope.** We take the limit $N_s\to\infty$ at fixed, arbitrary $N_d$, $\eta$, and $\beta$. We **do not** assume that $N_d$ is large. All weight vectors remain exactly on the unit sphere. The large-$N_s$ approximation is used only to obtain the limiting diffusion dynamics. In the maximally correlated interacting case, a particularly simple two-mode description additionally assumes an exchangeable weight-overlap matrix.

---

# Part I. Results

## 1. Model, notation, and assumptions

Let

$$
d=N_s,\qquad M=N_d,\qquad \tau=\frac{t}{d},
$$

where $t$ is the number of patterns learned after the stored pattern. Every dendritic weight has unit norm:

$$
\|\mathbf w_r\|=1,\qquad r=1,\ldots,M.
$$

For a **newly presented** input, set

$$
v_r=\mathbf w_r\cdot\mathbf x_r.
$$

The angular weight update is a rotation toward the input by

$$
\Delta\theta_r=\frac{a_r}{\sqrt d}.
$$

We consider the amplitudes

$$
\boxed{\begin{aligned}
a_r^{\mathrm I}(\mathbf v)
&=\eta\,\frac{e^{\beta v_r}}{\sum_{s=1}^{M}e^{\beta v_s}},\\[4pt]
a_r^{\mathrm N}(v_r)
&=\frac{\eta}{M}e^{\beta v_r-\beta^2/2}.
\end{aligned}}
\tag{1}
$$

The superscripts $\mathrm I$ and $\mathrm N$ indicate **interacting** and **non-interacting** learning. The second denominator is the expectation of the first denominator when each activation is marginally standard Gaussian, regardless of correlations among dendrites:

$$
\mathbb E\!\left[\sum_s e^{\beta v_s}\right]=M e^{\beta^2/2}.
$$

The two input models are:

- **Uncorrelated inputs (U):** $\mathbf x_1,\ldots,\mathbf x_M$ are independent $\mathcal N(\mathbf0,I_d)$ vectors for each pattern.
- **Maximally correlated inputs (C):** $\mathbf x_1=\cdots=\mathbf x_M=\mathbf x$, where $\mathbf x\sim\mathcal N(\mathbf0,I_d)$.

Let $\mathbf x_r^0$ be the previously stored pattern at dendrite $r$. Its overlap at memory age $t$ is

$$
u_r(t)=\mathbf w_r(t)\cdot\mathbf x_r^0.
$$

We seek $P(u_r(t)=u\mid u_r(0)=u_0)$, or, when coupling prevents closure in $u_r$ alone, a conditional law given the complete vector $\mathbf u(0)$.

We write $\mathcal N(u;\mu,V)$ for a normal probability density evaluated at $u$ with mean $\mu$ and variance $V$.

## 2. Four-case summary

| Rule | Input structure | One-dendrite conditional distribution | Essential coefficients |
|---|---|---|---|
| Non-interacting | Uncorrelated | Ornstein–Uhlenbeck Gaussian | $g_{\mathrm N}=\eta^2e^{\beta^2}/(2M^2)$ |
| Interacting | Uncorrelated | Ornstein–Uhlenbeck Gaussian | $g_{\mathrm{I,U}}=\eta^2 A_0/2$ |
| Non-interacting | Maximally correlated | **Same marginal** as uncorrelated case; joint noises correlated | $g_{\mathrm N}$ and evolving $Q$ |
| Interacting | Maximally correlated | Joint linear Gaussian process; two modes for exchangeable $Q$ | $H(Q)$, $C(Q)$, and $Q(\tau)$ |

### 2.1 Non-interacting, uncorrelated inputs

The old-pattern overlap follows

$$
\boxed{du_r=-g_{\mathrm N}u_r\,d\tau+\sqrt{2g_{\mathrm N}}\,dB_r,\qquad
 g_{\mathrm N}=\frac{\eta^2e^{\beta^2}}{2M^2}.}
\tag{2}
$$

Therefore, for any **finite** $M$,

$$
\boxed{
P_{\mathrm{N,U}}(u,\tau\mid u_0)
=\mathcal N\!\left(u;\,u_0e^{-g_{\mathrm N}\tau},\,1-e^{-2g_{\mathrm N}\tau}\right).
}
\tag{3}
$$

In particular,

$$
\mathbb E[u(\tau)\mid u_0]=u_0e^{-g_{\mathrm N}\tau},\qquad
\operatorname{Var}(u(\tau)\mid u_0)=1-e^{-2g_{\mathrm N}\tau}.
$$

### 2.2 Interacting, uncorrelated inputs

For independent standard normal variables $z_s$, define the finite-$M$ Gaussian expectation

$$
\boxed{
A_0(\beta,M)=
\mathbb E_{\mathbf z\sim\mathcal N(0,I_M)}
\left[
\frac{e^{2\beta z_1}}{\big(\sum_{s=1}^{M}e^{\beta z_s}\big)^2}
\right].
}
\tag{4}
$$

The forgetting rate is

$$
\boxed{g_{\mathrm{I,U}}=\frac{\eta^2}{2}A_0(\beta,M).}
\tag{5}
$$

The conditional distribution is

$$
\boxed{
P_{\mathrm{I,U}}(u,\tau\mid u_0)
=\mathcal N\!\left(u;\,u_0e^{-g_{\mathrm{I,U}}\tau},\,1-e^{-2g_{\mathrm{I,U}}\tau}\right).
}
\tag{6}
$$

The finite-$M$ limits are

$$
g_{\mathrm{I,U}}(\beta=0)=\frac{\eta^2}{2M^2},\qquad
\lim_{\beta\to\infty}g_{\mathrm{I,U}}=\frac{\eta^2}{2M}.
\tag{7}
$$

No approximation such as $A_0\simeq e^{\beta^2}/M^2$ is needed (and it would be inaccurate in the strong-selection regime at finite $M$).

### 2.3 Non-interacting, maximally correlated inputs

The **marginal** dynamics of any individual dendrite are unchanged:

$$
\boxed{
P_{\mathrm{N,C}}(u,\tau\mid u_0)
=\mathcal N\!\left(u;\,u_0e^{-g_{\mathrm N}\tau},\,1-e^{-2g_{\mathrm N}\tau}\right).
}
\tag{8}
$$

To describe correlations among dendrites, define their weight Gram matrix

$$
Q_{rs}(\tau)=\mathbf w_r(\tau)\cdot\mathbf w_s(\tau),\qquad Q_{rr}=1.
$$

For $r\ne s$,

$$
\boxed{
\frac{dQ_{rs}}{d\tau}=
\frac{\eta^2}{M^2}\left(e^{\beta^2Q_{rs}}-Q_{rs}e^{\beta^2}\right).
}
\tag{9}
$$

The joint overlap distribution, conditional on the complete initial vector $\mathbf u_0$ and $Q(0)$, is

$$
\boxed{
\mathbf u(\tau)\mid \mathbf u_0,Q(0)
\sim \mathcal N\!\left(
  e^{-g_{\mathrm N}\tau}\mathbf u_0,
  Q(\tau)-e^{-2g_{\mathrm N}\tau}Q(0)
\right).
}
\tag{10}
$$

Thus the single-dendrite law is independent of input sharing, but joint forgetting and the somatic response distribution are not.

For initially uncorrelated weights, $Q_{rs}(0)=0$, one obtains

$$
\left.\frac{dQ_{rs}}{d\tau}\right|_{Q_{rs}=0}=\frac{\eta^2}{M^2}>0.
\tag{11}
$$

### 2.4 Interacting, maximally correlated inputs: general finite-$M$ result

In this case $\mathbf v\sim\mathcal N(0,Q)$. Define

$$
p_r(\mathbf v)=\frac{e^{\beta v_r}}{\sum_s e^{\beta v_s}},\qquad
\boxed{C_{rs}(Q)=\mathbb E_{\mathbf v\sim\mathcal N(0,Q)}[p_rp_s].}
\tag{12}
$$

The matrix $C(Q)$ is evaluated as a finite-dimensional Gaussian expectation, **without** replacing $M$ by a large-$M$ limit.

Define the drift matrix

$$
\boxed{
H_{rs}(Q)=
\begin{cases}
-\dfrac{\eta^2}{2}C_{rr}
+\eta\beta\displaystyle\sum_{k\ne r}Q_{rk}C_{rk},&r=s,\\[8pt]
-\eta\beta C_{rs},&r\ne s.
\end{cases}
}
\tag{13}
$$

The closed weight-overlap dynamics are

$$
\boxed{\frac{dQ}{d\tau}=H(Q)Q+QH(Q)^\top+\eta^2C(Q).}
\tag{14}
$$

For the vector of old-pattern overlaps $\mathbf u=(u_1,\ldots,u_M)^\top$,

$$
\boxed{
d\mathbf u=H(Q)\mathbf u\,d\tau+\eta C(Q)^{1/2}d\mathbf B_\tau.
}
\tag{15}
$$

Let $\Phi(\tau)$ solve

$$
\dot\Phi(\tau)=H(Q(\tau))\Phi(\tau),\qquad \Phi(0)=I_M.
$$

Then the complete **conditional** transition law is

$$
\boxed{
\mathbf u(\tau)\mid\mathbf u_0,Q(0)
\sim\mathcal N\!\left(
\Phi(\tau)\mathbf u_0,\;
Q(\tau)-\Phi(\tau)Q(0)\Phi(\tau)^\top
\right).
}
\tag{16}
$$

Equation (16) holds for arbitrary finite $M$ and a general initial Gram matrix $Q(0)$ in the large-$N_s$ limit. The matrix ODE, not a large-$M$ approximation, determines its coefficients.

### 2.5 Exchangeable maximally correlated case: two forgetting modes

Suppose $M\ge2$ and the state has exchangeable overlaps,

$$
Q_{rr}=1,\qquad Q_{r\ne s}=q.
$$

The physically allowed interval is $-1/(M-1)\le q\le1$.

Set

$$
A(q)=\mathbb E[p_r^2],\qquad B(q)=\mathbb E[p_rp_s]\quad(r\ne s).
\tag{17}
$$

These expectations require only $M$ independent standard normals: the softmax allocation has the same distribution as

$$
\boxed{
p_r\overset{d}{=}
\frac{e^{\beta\sqrt{1-q}\,z_r}}{\sum_s e^{\beta\sqrt{1-q}\,z_s}},
\qquad z_s\overset{\mathrm{iid}}{\sim}\mathcal N(0,1).
}
\tag{18}
$$

The identity holds for positive **and negative** admissible $q$, since softmax is invariant under common shifts and depends only on activation differences. Also,

$$
A(q)+(M-1)B(q)=\frac1M.
$$

The entire weight-correlation dynamics reduce to

$$
\boxed{
\dot q
=\eta^2[B(q)-qA(q)]
-2\eta\beta(1-q)[1+(M-1)q]B(q).
}
\tag{19}
$$

Define the drift eigenvalues of the collective ($\parallel$) and relative ($\perp$) modes:

$$
\boxed{\begin{aligned}
h_\parallel(q)
&=-\frac{\eta^2A(q)}2-\eta\beta(M-1)(1-q)B(q),\\[4pt]
h_\perp(q)
&=-\frac{\eta^2A(q)}2+\eta\beta[1+(M-1)q]B(q).
\end{aligned}}
\tag{20}
$$

Define their propagators

$$
E_j(\tau)=\exp\!\left[\int_0^\tau h_j(q(s))\,ds\right],\qquad j\in\{\parallel,\perp\},
$$

and the initial mean across dendrites

$$
\bar u_0=\frac1M\sum_{s=1}^M u_s(0).
$$

Conditioned on the **full** initial overlap vector, the single-dendrite density is

$$
\boxed{
P_{\mathrm{I,C}}(u,\tau\mid\mathbf u_0,Q(0))
=\mathcal N\big(u;\mu_r(\tau),V_r(\tau)\big),
}
\tag{21}
$$

where

$$
\boxed{
\mu_r(\tau)=E_\perp(\tau)u_{r,0}+
[E_\parallel(\tau)-E_\perp(\tau)]\bar u_0,
}
\tag{22}
$$

and

$$
\boxed{\begin{aligned}
V_r(\tau)={}&\frac1M\Big\{1+(M-1)q(\tau)
-E_\parallel(\tau)^2[1+(M-1)q(0)]\Big\}\\
&+\frac{M-1}{M}\Big\{1-q(\tau)
-E_\perp(\tau)^2[1-q(0)]\Big\}.
\end{aligned}}
\tag{23}
$$

If the weights have reached a nondegenerate stationary correlation $q_*$, take $q(\tau)=q_*$ and define $g_j=-h_j(q_*)$. Then $E_j=e^{-g_j\tau}$, giving

$$
\boxed{
\mu_r(\tau)=\bar u_0e^{-g_\parallel\tau}
+(u_{r,0}-\bar u_0)e^{-g_\perp\tau},
}
\tag{24}
$$

$$
\boxed{\begin{aligned}
V_r(\tau)={}&\frac{1+(M-1)q_*}{M}(1-e^{-2g_\parallel\tau})\\
&+\frac{(M-1)(1-q_*)}{M}(1-e^{-2g_\perp\tau}).
\end{aligned}}
\tag{25}
$$

Stationary covariance balance further gives, for strictly positive covariance eigenvalues,

$$
\boxed{\begin{aligned}
g_\parallel&=\frac{\eta^2}{2M[1+(M-1)q_*]},\\
g_\perp&=\frac{\eta^2[A(q_*)-B(q_*)]}{2(1-q_*)}.
\end{aligned}}
\tag{26}
$$

The corresponding memory-age scales (in number of newly learned patterns) are $T_j=d/g_j$.

**Conditioning only on one initial overlap.** Unlike the other cases, $u_{r,0}=u_0$ is generally insufficient to fix the subsequent law: the mean also depends on $\bar u_0$. If $\pi(\bar u_0\mid u_0)$ denotes the distribution induced by the learning event, then

$$
\boxed{\begin{aligned}
P_{\mathrm{I,C}}(u,\tau\mid u_0)
={}&\int d\bar u_0\;\pi(\bar u_0\mid u_0)\\
&\times \mathcal N\!\left(
 u;\;E_\perp u_0+(E_\parallel-E_\perp)\bar u_0,\;V_r(\tau)
\right).
\end{aligned}}
\tag{27}
$$

In general this is a **Gaussian mixture**. For a pattern just stored by the interacting rule in the large-$d$ limit, its initial overlaps have the distribution

$$
\boxed{
\mathbf z\sim\mathcal N(0,Q(0)),\qquad
u_{r,0}=z_r+\eta p_r(\mathbf z),\qquad
\bar u_0=\bar z+\frac{\eta}{M}.
}
\tag{28}
$$

These equations specify $\pi(\bar u_0\mid u_0)$ without additional free parameters.

### 2.6 Initial weight-decorrelation diagnostic

If the weights are initially uncorrelated, $q(0)=0$, then

$$
\boxed{
\dot q_{\mathrm N}(0)=\frac{\eta^2}{M^2}>0,
\qquad
\dot q_{\mathrm I}(0)=\eta(\eta-2\beta)B(0).
}
\tag{29}
$$

Therefore, for finite $M\ge2$,

$$
\boxed{\beta>\eta/2\quad\Longrightarrow\quad\dot q_{\mathrm I}(0)<0,}
$$

whereas the non-interacting rule initially increases weight correlations. This is a property of the $N_s\to\infty$ dynamics and does not require $M\to\infty$.

---

# Part II. Derivations

## 3. Exact rotation and its large-$N_s$ expansion

For each new input, decompose

$$
\mathbf x_r=v_r\mathbf w_r+\mathbf x_{r,\perp},
\qquad \mathbf w_r\cdot\mathbf x_{r,\perp}=0.
$$

Let $s_r=\|\mathbf x_{r,\perp}\|$. The exactly norm-preserving update is

$$
\boxed{
\mathbf w_r'=
\mathbf w_r\cos\left(\frac{a_r}{\sqrt d}\right)
+\frac{\mathbf x_{r,\perp}}{s_r}
\sin\left(\frac{a_r}{\sqrt d}\right).
}
\tag{30}
$$

If $\mathbf x_r^0$ is the old pattern, put

$$
m_r=\mathbf w_r\cdot\mathbf x_r^0,
\qquad y_r=\mathbf x_r\cdot\mathbf x_r^0.
$$

The **exact** scalar update is

$$
\boxed{
m_r'=m_r\cos\left(\frac{a_r}{\sqrt d}\right)
+\frac{y_r-v_rm_r}{s_r}\sin\left(\frac{a_r}{\sqrt d}\right).
}
\tag{31}
$$

For fixed $M,\eta,\beta$ and Gaussian inputs, $s_r/\sqrt d\to1$ in probability. Expansion of Eq. (31) then gives the leading terms

$$
\boxed{
\Delta m_r=
\frac{a_r}{d}(y_r-v_rm_r)
-\frac{a_r^2}{2d}m_r
+\text{higher-order terms}.
}
\tag{32}
$$

The first term supplies random fluctuations and, for shared inputs, a coupling-dependent drift. The negative $a_r^2$ term represents geometric forgetting due to rotation on the sphere. Throughout, the old pattern has $\|\mathbf x_r^0\|^2/d\to1$, as it does for a Gaussian input.

## 4. Independent input directions: both learning rules

For uncorrelated dendritic inputs, the new pattern's perpendicular component is independent of the full vector of local activations $\mathbf v$ and is isotropically distributed. Consequently,

$$
\mathbb E[y_r-v_rm_r\mid\mathbf v]=0.
$$

The drift from Eq. (32) is therefore

$$
\boxed{
\mathbb E[\Delta m_r\mid m_r]=
-\frac{m_r}{2d}\mathbb E[a_r^2]+o(d^{-1}).
}
\tag{33}
$$

The perpendicular inner product has variance

$$
\mathbb E[(y_r-v_rm_r)^2]
=\|\mathbf x_r^0\|^2-m_r^2=d+o(d),
$$

so

$$
\boxed{
\mathbb E[(\Delta m_r)^2\mid m_r]
=\frac{\mathbb E[a_r^2]}{d}+o(d^{-1}).
}
\tag{34}
$$

For $r\ne s$, the leading conditional noise cross-moment vanishes: although interacting amplitudes depend on all activations, the perpendicular input directions are independent. Set

$$
2g=\mathbb E[a_r^2].
$$

Under the time scaling $\tau=t/d$, Eqs. (33)–(34) yield

$$
dm_r=-gm_r\,d\tau+\sqrt{2g}\,dB_r.
\tag{35}
$$

The Ornstein–Uhlenbeck solution is

$$
m_r(\tau)=e^{-g\tau}m_r(0)
+\sqrt{2g}\int_0^\tau e^{-g(\tau-s)}dB_r(s),
$$

so

$$
m_r(\tau)\mid m_r(0)=u_0
\sim\mathcal N(u_0e^{-g\tau},1-e^{-2g\tau}).
$$

For the non-interacting rule,

$$
\mathbb E[a_r^2]=
\frac{\eta^2}{M^2}
\mathbb E[e^{2\beta v_r-\beta^2}]
=\frac{\eta^2e^{\beta^2}}{M^2}.
$$

For the interacting rule,

$$
\mathbb E[a_r^2]
=\eta^2\mathbb E\left[
\frac{e^{2\beta v_r}}{(\sum_s e^{\beta v_s})^2}
\right]=\eta^2A_0.
$$

This proves Eqs. (2)–(7).

## 5. Shared inputs: general drift and diffusion

Now all dendrites receive $\mathbf x_r=\mathbf x$ and the stored pattern is also common, $\mathbf x_r^0=\mathbf x^0$.

Set

$$
Q_{rs}=\mathbf w_r\cdot\mathbf w_s,
\qquad m_r=\mathbf w_r\cdot\mathbf x^0,
\qquad y=\mathbf x\cdot\mathbf x^0.
$$

The local activation vector is exactly Gaussian conditional on the weights:

$$
\mathbf v\sim\mathcal N(0,Q).
\tag{36}
$$

For its joint Gaussian distribution with $y$,

$$
\mathbb E[v_rv_s]=Q_{rs},\qquad
\mathbb E[yv_s]=m_s.
$$

The overlap increment becomes

$$
\Delta m_r
=\frac{a_r}{d}(y-v_rm_r)
-\frac{a_r^2}{2d}m_r+\cdots.
\tag{37}
$$

For differentiable $a_r(\mathbf v)$, Gaussian integration by parts gives

$$
\mathbb E[a_ry]=
\sum_s m_s\mathbb E\left[\frac{\partial a_r}{\partial v_s}\right],
$$

$$
\mathbb E[a_rv_r]=
\sum_s Q_{rs}\mathbb E\left[\frac{\partial a_r}{\partial v_s}\right].
$$

Therefore the conditional drift is

$$
\boxed{
\lim_{d\to\infty}d\,\mathbb E[\Delta m_r]
=\sum_s(m_s-Q_{rs}m_r)
\mathbb E\left[\frac{\partial a_r}{\partial v_s}\right]
-\frac{m_r}{2}\mathbb E[a_r^2].
}
\tag{38}
$$

The leading random fluctuations arise from the common high-dimensional input component orthogonal to the span of the current weights. Its variance is asymptotically $d$, yielding the diffusion matrix

$$
\boxed{
D_{rs}(Q)=\lim_{d\to\infty}d\,\mathbb E[\Delta m_r\Delta m_s]
=\mathbb E[a_ra_s].
}
\tag{39}
$$

The limit is for fixed $M$ and a typical stored pattern with $\|\mathbf x^0\|^2/d\to1$. The conditional mean increments are order $d^{-1}$, so subtracting their products does not change the leading diffusion coefficient.

### 5.1 A general weight-overlap evolution identity

Expand the updated Gram-matrix element $Q'_{rs}=\mathbf w'_r\cdot\mathbf w'_s$ using Eq. (30). For $r\ne s$, the large-$d$ drift is

$$
\boxed{\begin{aligned}
\dot Q_{rs}={}&
\mathbb E[a_r(v_s-Q_{rs}v_r)]
+\mathbb E[a_s(v_r-Q_{rs}v_s)]\\
&+\mathbb E[a_ra_s]
-\frac{Q_{rs}}2\mathbb E[a_r^2+a_s^2].
\end{aligned}}
\tag{40}
$$

This equation is the starting point for both shared-input cases.

## 6. Shared inputs, non-interacting learning

Here

$$
a_r=\frac\eta M e^{\beta v_r-\beta^2/2},\qquad
\frac{\partial a_r}{\partial v_s}=\beta a_r\delta_{rs}.
$$

In Eq. (38), the derivative term is proportional to $m_r-Q_{rr}m_r=0$. Hence the drift is simply $-g_{\mathrm N}m_r$. The diffusion moments follow from joint Gaussian exponential averages:

$$
\begin{aligned}
D_{rs}
&=\frac{\eta^2}{M^2}
\mathbb E[e^{\beta(v_r+v_s)-\beta^2}]\\
&=\boxed{\frac{\eta^2}{M^2}e^{\beta^2Q_{rs}}}.
\end{aligned}
\tag{41}
$$

On the diagonal $D_{rr}=2g_{\mathrm N}$. The drift and diagonal diffusion thus coincide with the uncorrelated single-dendrite model.

In Eq. (40), the first two terms vanish by Gaussian integration by parts, leaving

$$
\dot Q_{rs}=D_{rs}-2g_{\mathrm N}Q_{rs},
$$

which proves Eq. (9).

For the multivariate process,

$$
d\mathbf u=-g_{\mathrm N}\mathbf u\,d\tau+D(Q)^{1/2}d\mathbf B_\tau.
$$

Its mean is $e^{-g_{\mathrm N}\tau}\mathbf u_0$. The covariance $V$ obeys

$$
\dot V=-2g_{\mathrm N}V+D(Q),\qquad V(0)=0.
$$

Since $Q$ solves the same inhomogeneous ODE, subtracting the two equations gives

$$
V(\tau)=Q(\tau)-e^{-2g_{\mathrm N}\tau}Q(0),
$$

proving Eq. (10).

## 7. Shared inputs, interacting learning

With $a_r=\eta p_r$,

$$
\frac{\partial a_r}{\partial v_s}
=\eta\beta p_r(\delta_{rs}-p_s).
$$

Let $C_{rs}=\mathbb E[p_rp_s]$. Since the $s=r$ term in $(m_s-Q_{rs}m_r)$ vanishes, Eq. (38) simplifies to

$$
\boxed{
\frac{dm_r}{d\tau}\Big|_{\mathrm{drift}}
=-\frac{\eta^2}{2}C_{rr}m_r
-\eta\beta\sum_s C_{rs}(m_s-Q_{rs}m_r).
}
\tag{42}
$$

Reading off the coefficients of $m_s$ yields the matrix $H(Q)$ in Eq. (13). From Eq. (39),

$$
D_{rs}(Q)=\eta^2 C_{rs}(Q).
$$

Therefore the overlap vector satisfies the multivariate diffusion equation (15).

For $Q$, apply Gaussian integration by parts to the first term of Eq. (40):

$$
\mathbb E[a_r(v_s-Q_{rs}v_r)]
=\sum_k(Q_{sk}-Q_{rs}Q_{rk})
\mathbb E\left[\frac{\partial a_r}{\partial v_k}\right].
$$

The $\delta_{rk}$ contribution cancels and the remaining term is

$$
-\eta\beta\sum_k(Q_{sk}-Q_{rs}Q_{rk})C_{rk}.
$$

Exchanging $r$ and $s$ handles the second term. Thus

$$
\boxed{\begin{aligned}
\dot Q_{rs}={}&-\eta\beta\sum_k(Q_{sk}-Q_{rs}Q_{rk})C_{rk}\\
&-\eta\beta\sum_k(Q_{rk}-Q_{rs}Q_{sk})C_{sk}\\
&+\eta^2\left[C_{rs}-\frac{Q_{rs}}2(C_{rr}+C_{ss})\right].
\end{aligned}}
\tag{43}
$$

This is equivalent to $\dot Q=HQ+QH^\top+\eta^2C$, proving Eq. (14).

### 7.1 Full Gaussian transition law

Once $Q(\tau)$ is determined, the coefficients of Eq. (15) are deterministic in the $d\to\infty$ limit. The mean satisfies

$$
\dot{\boldsymbol\mu}=H(Q)\boldsymbol\mu,
\qquad \boldsymbol\mu(0)=\mathbf u_0,
$$

so $\boldsymbol\mu=\Phi\mathbf u_0$. The covariance satisfies

$$
\dot V=HV+VH^\top+\eta^2C,\qquad V(0)=0.
$$

Because $Q$ obeys the identical inhomogeneous equation, the difference $Q-V$ evolves as

$$
\frac{d}{d\tau}(Q-V)=H(Q-V)+(Q-V)H^\top.
$$

With $Q(0)-V(0)=Q(0)$, its solution is

$$
Q(\tau)-V(\tau)=\Phi(\tau)Q(0)\Phi(\tau)^\top.
$$

This proves Eq. (16).

## 8. Exchangeable reduction and its two modes

For exchangeable $Q$ and $C$,

$$
Q=(1-q)I+q\mathbf1\mathbf1^\top,
\qquad
C=(A-B)I+B\mathbf1\mathbf1^\top.
$$

The drift matrix has diagonal element

$$
H_{rr}=-\frac{\eta^2 A}{2}+\eta\beta(M-1)qB
$$

and off-diagonal element $H_{r\ne s}=-\eta\beta B$.

The two invariant subspaces are generated by the projectors

$$
P_\parallel=\frac{\mathbf1\mathbf1^\top}{M},
\qquad P_\perp=I-P_\parallel.
$$

Their drift eigenvalues are

$$
h_\parallel=H_{rr}+(M-1)H_{rs},
\qquad
h_\perp=H_{rr}-H_{rs},
$$

which immediately yield Eq. (20).

The Gram matrix has eigenvalues

$$
Q_\parallel=1+(M-1)q,
\qquad Q_\perp=1-q,
$$

and the diffusion covariance has eigenvalues

$$
D_\parallel=\eta^2[A+(M-1)B]=\frac{\eta^2}{M},
\qquad
D_\perp=\eta^2(A-B).
$$

The propagator is

$$
\Phi(\tau)=E_\parallel(\tau)P_\parallel+E_\perp(\tau)P_\perp.
$$

The mean of dendrite $r$ is therefore

$$
\mu_r=E_\parallel\bar u_0+E_\perp(u_{r,0}-\bar u_0),
$$

which gives Eq. (22).

The covariance identity $V=Q-\Phi Q(0)\Phi^\top$ acts separately on the two eigenspaces. The variance of one dendrite is the diagonal element

$$
V_r=\frac{V_\parallel}{M}+\frac{M-1}{M}V_\perp,
$$

where

$$
V_j=Q_j(\tau)-E_j(\tau)^2Q_j(0),
\qquad j\in\{\parallel,\perp\}.
$$

This proves Eq. (23).

At a stationary, nondegenerate $q_*$, stationarity in each covariance eigenmode gives

$$
0=2h_jQ_j+D_j,
$$

so $g_j=-h_j=D_j/(2Q_j)$, proving Eq. (26).

### 8.1 Scalar overlap equation

Taking the off-diagonal component of the general Gram-matrix equation (43) and substituting exchangeable $Q,C$ gives

$$
\dot q=\eta^2(B-qA)
-2\eta\beta(1-q)[1+(M-1)q]B.
$$

At $q=0$ this becomes $\dot q=\eta(\eta-2\beta)B(0)$, while the non-interacting equation yields $\dot q=\eta^2/M^2$; together these prove Eq. (29).

---

# Part III. Interpretation and limitations

## 9. What the four cases establish

**Individual forgetting versus collective interference.** With independent dendritic inputs, the old-pattern overlap of each dendrite is an Ornstein–Uhlenbeck process. Interacting normalization changes the second moment of the angular update, hence the forgetting rate, but does not cause leading-order coupling between individual overlap noises.

With fully shared inputs, the non-interacting rule leaves **each marginal distribution** unchanged, while creating correlated weight vectors and correlated forgetting noise. The interacting rule adds deterministic coupling between different old-pattern overlaps, leading to collective and relative modes.

**Finite number of dendrites.** All Gaussian expectations, matrix equations, and two-mode eigenvalues above are valid at fixed finite $M$. In particular, the finite-$M$ softmax moments $A_0$, $A(q)$, and $B(q)$ should be evaluated directly; they should not automatically be replaced by large-$M$ asymptotics.

**Stationarity versus transient training.** Equations (14)–(16) handle evolving weight correlations. Equations (24)–(26) assume a stationary nondegenerate exchangeable state. The latter are appropriate for a memory stored after the neuron has undergone prolonged training.

**Conditioning on $u_0$ alone.** For interacting learning under fully shared inputs, the future overlap of one dendrite generally depends on the initial population-average overlap $\bar u_0$. The genuinely single-overlap conditional distribution is the mixture in Eq. (27), with an initial-state distribution generated by Eq. (28).

**Relationship to memory capacity.** The single-dendrite laws do not alone determine the familiarity-detection capacity, because the somatic output

$$
Y(t)=\sum_{r=1}^M\max(u_r(t)-b,0)
$$

depends on the *joint* distribution of overlaps. Equations (10) and (16), together with the immediate post-learning distribution, provide the necessary joint theory for computing response statistics and eventual capacity.

## 10. Technical qualifications

- All diffusion statements are leading-order limits for $d=N_s\to\infty$ at fixed $M=N_d$, $\eta$, and $\beta$. They are not exact finite-$N_s$ Gaussian laws.
- Gaussian patterns and exactly unit-normalized dendritic weights are assumed. The norm concentration $\|\mathbf x^0\|^2/d\to1$ is used for the old stored pattern.
- The exchangeable reduction assumes that the Gram matrix is actually of exchangeable form; average pairwise correlation alone does not generally ensure exact closure to a scalar $q$. The full matrix theory applies to nonexchangeable states.
- For the non-interacting exponential rule, the angular amplitude is unbounded. Its moments are finite for fixed $\beta$, and the diffusion limit is taken at fixed $\beta$; the approximation is not uniform as $\beta$ increases with $N_s$.
- The formulas involving stationary rate ratios $D_j/(2Q_j)$ require nonzero covariance eigenvalues. Use the original drift formulas at degenerate boundaries such as perfectly synchronized weights.
- When specifying only $u_r(0)=u_0$ rather than the entire initial state, one must say which initial learning ensemble is being conditioned on. This is especially important for the interacting shared-input model.

---

## 11. Compact equation index

| Case | Rate or state equation | Conditional distribution |
|---|---|---|
| Non-interacting / uncorrelated | $g=\eta^2e^{\beta^2}/(2M^2)$ | Eq. (3) |
| Interacting / uncorrelated | $g=\eta^2A_0/2$ | Eq. (6) |
| Non-interacting / maximally correlated | $\dot Q_{rs}=\eta^2M^{-2}(e^{\beta^2Q_{rs}}-e^{\beta^2}Q_{rs})$ | Eq. (8) marginal; Eq. (10) joint |
| Interacting / maximally correlated | $\dot Q=HQ+QH^\top+\eta^2C$ | Eq. (16) joint; Eq. (27) conditioned only on one $u_0$ |

**Central conclusion:** At large $N_s$ but arbitrary finite $N_d$, the non-interacting rule admits a universal single-dendrite forgetting rate independent of input sharing. Interacting learning with shared inputs has a closed matrix-valued theory and, under exchangeability, two distinct collective forgetting modes. This distinction supplies a route from plasticity normalization to correlated forgetting and ultimately to neuronal familiarity-detection capacity.
