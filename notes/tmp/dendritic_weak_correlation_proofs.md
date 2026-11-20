# Proofs and derivations for the weak-correlation dendritic-learning theory

**Companion:** [`dendritic_weak_correlation_results.md`](dendritic_weak_correlation_results.md).  
This document proves results **(R0)–(R21)** from that note. We use **$N_s$** for synapses per dendrite, **$N_d$** for the number of dendrites, and **$u_r$** for the local activation of a newly presented pattern. An old-pattern overlap is denoted by $m_r$, so the two are never confused.

## A. Assumptions and precision of the statements

1. New patterns consist of fresh, temporally independent Gaussian axonal vectors $\mathbf x\sim\mathcal N(0,I_{N_a})$. Each dendrite $r$ has a fixed orthogonal binary projector $P_r$ with $\operatorname{tr}P_r=N_s$. Weight vectors satisfy $P_r\mathbf w_r=\mathbf w_r$ and $\|\mathbf w_r\|=1$.
2. Unless a separate $N_d\to\infty$ limit is stated, take $N_s\to\infty$ with **$N_d$ fixed** and learning amplitudes $a_r(\mathbf u)$ having finite Gaussian moments sufficient for the expansions below. For target-based and hard-selection rules, discontinuities are allowed. We assume the angular step $a_r/\sqrt{N_s}$ is asymptotically small in probability.
3. Weight dynamics on $\tau=t/N_s$ converge at leading order to deterministic Gram-matrix dynamics and, for a fixed old pattern of norm $\|\mathbf x^{(0)}\|^2/N_s\to1$, to a diffusion process for the old-pattern overlaps. The letter $\dot{}$ means $d/d\tau$ throughout.
4. The **weak-correlation** expansions are made at $Q=I_{N_d}+E$, $E_{rr}=0$, $\epsilon=\max_{r\ne s}|E_{rs}|\ll1$, for fixed $N_d$. Their $O(\epsilon^2)$ remainders need not be uniform when $N_d$ also grows.
5. Statements requiring isotropic random initialization or an additional *random-mask mixing closure* are marked explicitly. Weak ordinary weight overlaps do not imply that connectivity-restricted overlaps are small or homogeneous.

## B. Exact sphere geometry and its large-$N_s$ expansion — (R0)–(R2)

Set $u_r=\mathbf w_r^\top P_r\mathbf x=\mathbf w_r^\top\mathbf x$ and define

$$s_r=\sqrt{\|P_r\mathbf x\|^2-u_r^2},\qquad
\mathbf e_r=\frac{P_r\mathbf x-u_r\mathbf w_r}{s_r}.$$

Since $P_r\mathbf w_r=\mathbf w_r$ and $P_r=P_r^\top=P_r^2$,

$$\mathbf w_r^\top\mathbf e_r=0,\quad
\|\mathbf e_r\|=1,\quad
P_r\mathbf e_r=\mathbf e_r.$$

Therefore the **exact** rotation

$$
\boxed{\mathbf w_r'=\mathbf w_r\cos\theta_r+\mathbf e_r\sin\theta_r,
\quad \theta_r=\frac{a_r(\mathbf u)}{\sqrt{N_s}}}
\tag{P1}
$$

preserves $\|\mathbf w_r'\|=1$ and $P_r\mathbf w_r'=\mathbf w_r'$, even when the ambient dimension $N_a$ is much larger than $N_s$. The updated activation for the *new* pattern is exactly

$$u_r'=\mathbf w_r'^\top P_r\mathbf x=u_r\cos\theta_r+s_r\sin\theta_r.$$

For unit-norm weights and Gaussian inputs, $u_r=O_p(1)$ and $s_r/\sqrt{N_s}\to1$ in probability. Consequently, for $a_r=O_p(1)$,

$$\boxed{u_r'-u_r=a_r(\mathbf u)+o_p(1).} \tag{P2}$$

This justifies mapping a desired local activation increase to an angular amplitude. For example, the target-based learning rule $u_r'\approx b+\kappa$ uses $a_r=[b+\kappa-u_r]_+$ on selected branches; the other amplitudes in (R2) follow identically. These updates match the *intended target activations* in the large-$N_s$ regime, not the exact finite-$N_s$ weights of the original manuscript.

In scalar products with a fixed unit weight or with the old-pattern vector, the rotation has the generator-level expansion

$$
\boxed{\Delta\mathbf w_r
=\frac{a_r}{N_s}(P_r\mathbf x-u_r\mathbf w_r)
-\frac{a_r^2}{2N_s}\mathbf w_r
+\text{terms negligible in the limiting drift/diffusion}.} \tag{P3}
$$

The first term is a tangent update; the second is the spherical norm-preservation correction. The latter is essential for the forgetting-rate and correlation-damping terms. We use (P3) only in moments to their leading $N_s^{-1}$ order, not as a uniform vector-norm error estimate.

## C. Single-dendrite forgetting with disjoint inputs — (R3)–(R4)

Let the stored pattern at dendrite $r$ be $\mathbf x_r^{(0)}$, define $m_r=\mathbf w_r^\top\mathbf x_r^{(0)}$, and set $y_r=\mathbf x_r^\top\mathbf x_r^{(0)}$. From (P3),

$$
\Delta m_r=
\frac{a_r}{N_s}(y_r-u_rm_r)-\frac{a_r^2}{2N_s}m_r
+\text{negligible terms}. \tag{P4}
$$

Decompose $\mathbf x_r=u_r\mathbf w_r+\mathbf x_{r,\perp}$. For disjoint inputs across dendrites, the perpendicular Gaussian vector $\mathbf x_{r,\perp}$ is independent of the entire activation vector $\mathbf u$ on that pattern. Hence, even when $a_r$ depends on all $u_s$,

$$
\mathbb E[a_r(y_r-u_rm_r)\mid\mathbf w,\mathbf x^{(0)}]=0.
$$

The conditional mean increment is therefore

$$
N_s\,\mathbb E[\Delta m_r\mid\mathbf w,\mathbf x^{(0)}]
\longrightarrow-\frac12\langle a_r^2\rangle_0m_r
=-\frac{D_r}{2}m_r. \tag{P5}
$$

For the variance, conditional on the current weights and old pattern,

$$
\mathbb E[(y_r-u_rm_r)^2\mid\mathbf w,\mathbf x^{(0)},\mathbf u]
=\|\mathbf x_r^{(0)}\|^2-m_r^2.
$$

If $\|\mathbf x_r^{(0)}\|^2/N_s\to1$ and $m_r=O(1)$, then

$$
N_s\,\mathbb E[(\Delta m_r)^2\mid\mathbf w,\mathbf x^{(0)}]
\longrightarrow D_r. \tag{P6}
$$

For distinct dendrites with disjoint input supports, the residual tangent Gaussian noises are independent, so their leading cross quadratic variations vanish, even if the amplitudes $a_r$ are coupled through $\mathbf u$. The diffusion limit is

$$dm_r=-\frac{D_r}{2}m_r\,d\tau+\sqrt{D_r}\,dB_r.$$

Solving this linear SDE,

$$m_r(\tau)=m_r(0)e^{-D_r\tau/2}
+\sqrt{D_r}\int_0^\tau e^{-D_r(\tau-s)/2}\,dB_r(s),$$

which has variance $D_r\int_0^\tau e^{-D_r(\tau-s)}ds=1-e^{-D_r\tau}$. This proves (R3)–(R4). The initial distribution of $m_r(0)$ need not itself be Gaussian; the statement is conditional on its value.

## D. Maximally shared inputs: the complete finite-$N_d$ closure — (R5)–(R9)

In this section all dendrites receive the **same** new input $\mathbf x\in\mathbb R^{N_s}$. Arrange the weight vectors as columns of $W\in\mathbb R^{N_s\times N_d}$. Then

$$Q=W^\top W,\qquad \mathbf u=W^\top\mathbf x\sim\mathcal N(0,Q).$$

For a singular Gram matrix, Gaussian conditioning uses the Moore–Penrose pseudoinverse:

$$\boxed{\mathbb E[\mathbf x\mid\mathbf u]=WQ^+\mathbf u.} \tag{P7}$$

Indeed, $\operatorname{Cov}(\mathbf x,\mathbf u)=W$ and $\operatorname{Cov}(\mathbf u)=Q$. Let

$$C_{rs}=\langle a_ra_s\rangle_Q,\quad F_{rs}=\langle a_ru_s\rangle_Q,\quad J=FQ^+.$$

Using (P7),

$$\boxed{\mathbb E[a_r(\mathbf u)\mathbf x]=\sum_kJ_{rk}\mathbf w_k.} \tag{P8}$$

Moreover $F_{rr}=\mathbb E[a_ru_r]$, and $JQ=F$ because $\mathbf u$ lies in the range of $Q$ almost surely. Taking expectation in (P3) therefore gives

$$
N_s\,\mathbb E[\Delta\mathbf w_r\mid W]
=\sum_kJ_{rk}\mathbf w_k
-\left(F_{rr}+\frac12C_{rr}\right)\mathbf w_r+o(1).
\tag{P9}
$$

This defines $H=J-\operatorname{diag}(F_{rr}+C_{rr}/2)$, as in (R5).

For the Gram matrix, $\Delta Q_{rs}=\Delta\mathbf w_r\cdot\mathbf w_s+\mathbf w_r\cdot\Delta\mathbf w_s+\Delta\mathbf w_r\cdot\Delta\mathbf w_s$. The first two terms produce $(HQ+QH^\top)_{rs}$ at scale $N_s^{-1}$. The last term is non-negligible because $\|\mathbf x\|^2=O_p(N_s)$; specifically,

$$
N_s\,\mathbb E[\Delta\mathbf w_r\cdot\Delta\mathbf w_s\mid W]
\longrightarrow C_{rs}.
$$

Thus

$$\boxed{\dot Q=HQ+QH^\top+C.} \tag{P10}$$

Expanding the $rs$ entry and using $JQ=F$ yields, for $r\ne s$,

$$
\dot Q_{rs}=F_{rs}+F_{sr}+C_{rs}
-Q_{rs}\left(F_{rr}+F_{ss}+\tfrac12(C_{rr}+C_{ss})\right),
$$

which proves (R6)–(R7). For $r=s$ the expression vanishes, as it must because weights remain normalized.

**Old-pattern process.** Let $\mathbf x^{(0)}$ be a fixed old shared Gaussian pattern and $\mathbf m=W^\top\mathbf x^{(0)}$. Put $y=\mathbf x^{(0)\top}\mathbf x$. Conditionally on $W,\mathbf x^{(0)}$, $(\mathbf u,y)$ is Gaussian with

$$\operatorname{Cov}(\mathbf u)=Q,\quad
\operatorname{Cov}(y,\mathbf u)=\mathbf m^\top,\quad
\operatorname{Var}(y)=\|\mathbf x^{(0)}\|^2.$$

Consequently, $\mathbb E[y\mid\mathbf u]=\mathbf m^\top Q^+\mathbf u$, so

$$\mathbb E[a_r y]=\sum_kJ_{rk}m_k.$$

Subtracting $m_r\mathbb E[a_ru_r]$ and the spherical correction in (P4) gives the drift $H\mathbf m$. The conditional residual variance of $y$ given $\mathbf u$ is

$$\|\mathbf x^{(0)}\|^2-\mathbf m^\top Q^+\mathbf m=N_s[1+o(1)]$$

in the fixed-$N_d$, bounded-overlap regime; therefore the leading quadratic variation of $\mathbf m$ is $C\,d\tau$. This proves (R8).

As $Q(\tau)$ obeys a deterministic ODE at leading order, the old-overlap SDE is linear with deterministic time-dependent coefficients. Its mean is $\Phi(\tau)\mathbf m_0$, where $\dot\Phi=H\Phi$, $\Phi(0)=I$. Its conditional covariance $V$ satisfies

$$\dot V=HV+VH^\top+C,\qquad V(0)=0.$$

But $Q$ satisfies the same inhomogeneous equation. Hence $V=Q-\Phi Q(0)\Phi^\top$ by uniqueness, proving (R9). Non-Gaussianity can re-enter when averaging over an unspecified initial vector $\mathbf m_0$ or initial Gram matrix $Q(0)$.

## E. Weak-correlation Gaussian density expansion — (R10)

Let $Q=I_{N_d}+E$, with $E_{rr}=0$ and $\epsilon=\max_{r\ne s}|E_{rs}|\ll1$. For fixed $N_d$,

$$Q^{-1}=I-E+O(\epsilon^2),\qquad
\det(Q)^{-1/2}=1+O(\epsilon^2),$$

because $\operatorname{tr}E=0$. If $\varphi_Q$ is the Gaussian density of $\mathcal N(0,Q)$, then

$$
\begin{aligned}
\frac{\varphi_Q(\mathbf z)}{\varphi_I(\mathbf z)}
&=\det(Q)^{-1/2}
\exp\!\left[-\frac12\mathbf z^\top(Q^{-1}-I)\mathbf z\right]\\
&=1+\frac12\mathbf z^\top E\mathbf z+O(\epsilon^2\,\text{polynomially weighted terms})\\
&=1+\sum_{a<b}E_{ab}z_az_b+O(\epsilon^2\,\text{weighted terms}).
\end{aligned}
$$

Integrate against $G(\mathbf z)$, assuming sufficient Gaussian-weighted integrability to exchange expansion and integration. This gives

$$\boxed{\langle G\rangle_{I+E}=\langle G\rangle_0+
\sum_{a<b}E_{ab}\langle Gz_az_b\rangle_0+O(\epsilon^2).}$$

No derivatives of $G$ occur. Consequently the expansion remains valid for thresholded and rank-based update functions with sufficient moments.

## F. Non-interacting learning with arbitrary axonal sharing — (R11)–(R14)

Let $a_r=f(u_r)$, with $P_r$ as in section B. Write $q=Q_{rs}$ for $r\ne s$, $\alpha_{rs}=\operatorname{tr}(P_rP_s)/N_s$, and $K(q)=\mathbb E[f(Z_1)f(Z_2)]$ for correlated standard normals with covariance $q$.

The conditional expectation of the projected input against another weight is

$$
\mathbb E[\mathbf w_s^\top P_r\mathbf x\mid u_r]
=\operatorname{Cov}(\mathbf w_s^\top P_r\mathbf x,u_r)\,u_r
=(\mathbf w_s^\top P_r\mathbf w_r)u_r=q u_r.
$$

This identity uses $P_r\mathbf w_r=\mathbf w_r$ and holds even if $f$ is discontinuous. Therefore

$$\mathbb E[f(u_r)(\mathbf w_s^\top P_r\mathbf x-qu_r)]=0.$$

The analogous term for dendrite $s$ vanishes. The quadratic product of the two tangent updates has leading mean $\alpha_{rs}\mathbb E[f(u_r)f(u_s)]/N_s$: the only extensive contribution is the shared input norm

$$\mathbb E\left[\mathbf x^\top P_rP_s\mathbf x\right]=\operatorname{tr}(P_rP_s)=N_s\alpha_{rs}.$$

The spherical correction gives $-\frac12q(\mathbb E[f(u_r)^2]+\mathbb E[f(u_s)^2])=-Dq$ because each activation is standard normal. Hence

$$\boxed{\dot Q_{rs}=\alpha_{rs}K(Q_{rs})-DQ_{rs}.} \tag{P11}$$

This proves (R12), and notably needs no equality of the off-diagonal overlaps.

For the weak-correlation expansion, expand $f$ in probabilists' Hermite polynomials $H_k$:

$$f(z)=\sum_{k\geq0}c_kH_k(z),\qquad
c_k=\frac{\mathbb E[f(Z)H_k(Z)]}{k!}.$$

For correlated standard normals with covariance $q$,

$$\mathbb E[H_j(Z_1)H_k(Z_2)]=\delta_{jk}\,k!\,q^k.$$

Therefore,

$$K(q)=\sum_{k\geq0}k!c_k^2q^k=\mu^2+\chi^2q+O(q^2),$$

since $c_0=\mu$, $c_1=\chi$. Inserting into (P11) proves (R13). Parseval's identity gives $D=\mu^2+\chi^2+\sum_{k\geq2}k!c_k^2\geq\mu^2+\chi^2$, so $D-\alpha_{rs}\chi^2\geq0$ for $0\leq\alpha_{rs}\leq1$.

For homogeneous anatomical sharing, sum (R13) over ordered pairs $r\ne s$ and divide by $N_d$. Use

$$R-1=\frac1{N_d}\sum_{r\ne s}Q_{rs}.$$

The source term becomes $(N_d-1)\alpha\mu^2$, proving (R14). Its linearized fixed point is valid only when $\alpha\mu^2/(D-\alpha\chi^2)$ is small and the denominator is positive.

## G. Interacting weak-correlation theory with fully shared inputs — (R15)–(R16)

Assume the rule $a_r(\mathbf u)$ is permutation-equivariant and square-integrable, but not necessarily smooth. Define at independent standard normal activations

$$
C_o=\langle a_1a_2\rangle_0,\quad F_o=\langle a_1z_2\rangle_0,\quad
F_d=\langle a_1z_1\rangle_0,\quad D=\langle a_1^2\rangle_0.
$$

For a pair $r\ne s$, define

$$G_{rs}(\mathbf z)=a_ra_s+a_rz_s+a_sz_r,$$

where $a_r=a_r(\mathbf z)$. From the full-sharing equation (R7),

$$
\dot Q_{rs}=\langle G_{rs}\rangle_Q-Q_{rs}
\left(\langle a_ru_r\rangle_Q+\langle a_su_s\rangle_Q+
\frac{\langle a_r^2\rangle_Q+\langle a_s^2\rangle_Q}{2}\right).
$$

Apply the Gaussian expansion (R10). In the second term, the factor $Q_{rs}=O(\epsilon)$ means that only its zeroth-order expectation contributes at first order. In the first term use the full first-order expansion. Thus

$$
\boxed{\dot Q_{rs}=B-(2F_d+D)Q_{rs}
+\sum_{a<b}Q_{ab}\langle G_{rs}z_az_b\rangle_0+O(\epsilon^2),}
\tag{P12}
$$

where $B=C_o+2F_o$. This is a **matrix-level** weak-correlation theory; it generally couples different $Q_{rs}$.

To prove that its collective sum closes, let

$$\mathcal T(\mathbf z)=\sum_{a<b}z_az_b.$$

For every fixed pair $a<b$, permutation equivariance gives the identity

$$
\sum_{r<s}\langle G_{rs}(\mathbf z)z_az_b\rangle_0
=\langle G_{12}(\mathbf z)\mathcal T(\mathbf z)\rangle_0.
\tag{P13}
$$

One way to see this is to classify the pairs $(r,s)$ by whether they share two, one, or no indices with $(a,b)$; the same classes and multiplicities occur on the right. This uses symmetry of the *rule and independent Gaussian measure*, not equality of the *actual* $Q_{rs}$.

Now sum (P12) over the unordered pairs $r<s$. Since

$$\sum_{r<s}Q_{rs}=\frac{N_d}{2}(R-1),\qquad
\dot R=\frac2{N_d}\sum_{r<s}\dot Q_{rs},$$

we obtain

$$
\dot R=(N_d-1)B+
\left[\langle G_{12}\mathcal T\rangle_0-2F_d-D\right](R-1)+O(\epsilon^2).
$$

This proves (R15)–(R16) **without assuming $Q_{rs}=q$**. The proof requires $N_d\geq2$; the $N_d=1$ collective correlation question is trivial.

As an internal check, consider the constant-amplitude rule $a_r=\eta/N_d$. It has $B=D=\eta^2/N_d^2$, $F_d=F_o=0$, and $\langle G_{12}\mathcal T\rangle_0=0$. Thus (R16) becomes

$$\dot R=(N_d-1)\frac{\eta^2}{N_d^2}-\frac{\eta^2}{N_d^2}(R-1),$$

which agrees with direct spherical random-walk geometry.

## H. Partially shared inputs: exact restricted-overlap equation — (R17) and the limits of closure

Set $T_{srk}=\mathbf w_s^\top P_r\mathbf w_k$. In particular,

$$T_{srr}=Q_{sr},\qquad
T_{srs}=\|P_r\mathbf w_s\|^2.
$$

The latter is **not** generally determined by the ordinary Gram matrix. Define $C$ and $F$ by the same Gaussian activation expectations as in section D, since the activation vector still has covariance $Q$. Write $J=FQ^+$.

Taking scalar products in (P3), expanding $\Delta Q_{rs}$, and using Gaussian conditioning as in (P7) gives the exact leading-$N_s$ **fixed-mask** relation, for $r\ne s$,

$$
\boxed{\begin{aligned}
\dot Q_{rs}={}&
\sum_k(T_{srk}-Q_{rs}Q_{rk})J_{rk}
+\sum_k(T_{rsk}-Q_{rs}Q_{sk})J_{sk}\\
&+\alpha_{rs}C_{rs}
-\frac{Q_{rs}}{2}(C_{rr}+C_{ss}).
\end{aligned}} \tag{P14}
$$

The projected-input covariance is $\mathbb E[(\mathbf w_s^\top P_r\mathbf x)u_k]=T_{srk}$, explaining the first sum. The second is its $r\leftrightarrow s$ counterpart. The quadratic update term is $\alpha_{rs}C_{rs}$ because the shared support has dimension $N_s\alpha_{rs}$. Formula (P14) reduces to (R7) when all $P_r=I$, since then $T_{srk}=Q_{sk}$.

**Initial condition.** Suppose weights are initialized independently and isotropically on their individual support spheres, with the support-overlap fraction $\alpha_{rs}$ converging to a deterministic value as $N_s\to\infty$. Then, for a fixed pair $r\ne s$,

$$Q_{rs}(0)\to0,\qquad
T_{srs}(0)\to\alpha_{rs},\qquad
T_{srk}(0)\to0\quad(k\notin\{r,s\}).$$

At $Q=I$, $J=F$, and the $k=r$ contribution to the first sum cancels exactly because $T_{srr}=Q_{rs}$. Only $k=s$ remains, giving $\alpha_{rs}F_o$. Similarly the second sum gives $\alpha_{rs}F_o$, and $\alpha_{rs}C_{rs}=\alpha_{rs}C_o$. The spherical correction vanishes at $Q_{rs}=0$. Hence

$$\boxed{\left.\dot Q_{rs}\right|_{Q=I}=\alpha_{rs}(C_o+2F_o)=\alpha_{rs}B.}$$

This proves (R17), including the stated homogeneous-sharing formula for $R$.

**Why weak $Q$ is insufficient later.** The terms $T_{srs}$ and $T_{srk}$ describe where the weights are supported on axonal subsets. Two synaptic configurations can have the same small $Q_{rs}$ but different $T_{srs}$ or $T_{srk}$; (P14) then gives different drifts. Thus a weak-$Q$ approximation alone cannot prove a partial-sharing scalar equation for $R$.

## I. An explicitly conditional random-mask mixing closure — (R18)

For homogeneous support sharing $\alpha_{rs}=\alpha$ and weak $Q=I+E$, *additionally postulate* that restricted overlaps are delocalized to first order:

$$
\begin{cases}
T_{srr}=Q_{sr}&\text{(exact)},\\
T_{srs}=\alpha+O(\epsilon^2)+\delta_{srs},\\
T_{srk}=\alpha Q_{sk}+O(\epsilon^2)+\delta_{srk},&r,s,k\ \text{all distinct}.
\end{cases} \tag{P15}
$$

The $\delta$ terms represent errors of the mask-mixing approximation. Their contributions may be non-negligible for structured plasticity or clustered connectivity; they are not controlled by small $Q$ alone.

Set $q=Q_{rs}$. In the first sum of (P14), the $k=r$ term cancels exactly. For $k\ne r$, replace $T_{srk}$ by $\alpha Q_{sk}$ at first order and note $qQ_{rk}=O(\epsilon^2)$ (because $k\ne r$). Thus the first sum becomes

$$
\alpha\sum_{k\ne r}Q_{sk}J_{rk}
=\alpha(F_{rs}-qJ_{rr})+O(\epsilon^2)+\text{mask error}.
$$

Likewise the second becomes $\alpha(F_{sr}-qJ_{ss})+O(\epsilon^2)+\text{mask error}$. Since $J_{rr}=F_d+O(\epsilon)$, $J_{ss}=F_d+O(\epsilon)$, and $C_{rr}=C_{ss}=D+O(\epsilon)$, formula (P14) becomes

$$
\dot Q_{rs}=\alpha\langle G_{rs}\rangle_Q
-q(2\alpha F_d+D)+O(\epsilon^2)+\text{mask error}.
\tag{P16}
$$

Expand $\langle G_{rs}\rangle_Q$ with (R10), then sum over pairs using (P13). This yields

$$
\dot R=(N_d-1)\alpha B+
\bigl[\alpha\langle G_{12}\mathcal T\rangle_0-2\alpha F_d-D\bigr](R-1)
+O(\epsilon^2)+\text{mask error}.
$$

Because $\Lambda=\langle G_{12}\mathcal T\rangle_0-2F_d-D$, the coefficient is $\alpha(\Lambda+D)-D$. This proves (R18) **conditional on (P15)**. It is not claimed as an unconditional consequence of (R17).

Checks: for $\alpha=1$, (P16) recovers the full-sharing weak-correlation equation. For a non-interacting local rule $a_r=f(u_r)$, $F_d=\chi$, $F_o=0$, and the expansion gives $\langle G_{12}\mathcal T\rangle_0=2\chi+\chi^2$, so $\Lambda=\chi^2-D$. Then $\Lambda_\alpha=\alpha\chi^2-D$, matching the **exact** non-interacting weak-correlation equation (R14).

## J. Constant total plasticity budget — (R19)–(R21)

This section applies only when $\sum_r a_r(\mathbf u)=\eta$ *for every pattern*. It should not be used for the original fixed-target rules without a separate demonstration of that condition.

For full sharing define $\mathbf W=\sum_r\mathbf w_r$ and $k_r=\mathbf W^\top\mathbf w_r=\sum_sQ_{rs}$. Summing (P3) gives

$$
\Delta\mathbf W=\frac1{N_s}\left[
\eta\mathbf x-\sum_r a_ru_r\mathbf w_r
-\frac12\sum_r a_r^2\mathbf w_r
\right]
+\text{negligible terms}. \tag{P17}
$$

Because $\eta$ is deterministic and $\mathbf x$ is centered,

$$\mathbb E[\mathbf W^\top(\eta\mathbf x)]=0.$$

Also, the extensive contribution to $\|\Delta\mathbf W\|^2$ is $\eta^2\|\mathbf x\|^2/N_s^2$, so $N_s\mathbb E[\|\Delta\mathbf W\|^2]\to\eta^2$. With $R=\|\mathbf W\|^2/N_d$, use $\Delta\|\mathbf W\|^2=2\mathbf W\cdot\Delta\mathbf W+\|\Delta\mathbf W\|^2$ to obtain

$$
\boxed{\dot R=\frac{\eta^2}{N_d}-\frac1{N_d}\sum_r k_r\langle2a_ru_r+a_r^2\rangle_Q.}
$$

This proves (R19), without requiring weak $Q$.

For the weak expansion, set $h_r(\mathbf z)=2a_r(\mathbf z)z_r+a_r(\mathbf z)^2$. Define

$$K_0=\langle h_1\rangle_0,\qquad
K_1=\langle h_1\mathcal T\rangle_0.$$

Use $k_r=1+\sum_{s\ne r}Q_{rs}$. Apply (R10) to $\langle h_r\rangle_Q$, and retain terms linear in the off-diagonal overlaps. The $k_r$ expansion contributes $-K_0(R-1)$. For the distribution correction, permutation symmetry gives, for each pair $a<b$,

$$
\sum_r\langle h_rz_az_b\rangle_0
=\frac{2}{N_d-1}\langle h_1\mathcal T\rangle_0
=\frac{2K_1}{N_d-1}.
$$

After summing over pairs, the correction is $-K_1(R-1)/(N_d-1)$. Hence

$$
\boxed{\dot R=\frac{\eta^2}{N_d}-K_0-
\left(K_0+\frac{K_1}{N_d-1}\right)(R-1)+O(\epsilon^2),}
$$

proving (R20). It is algebraically equivalent to (R16) under the constant-budget constraint; either expression can be used to estimate the coefficients.

**Fixed-amplitude $n$-WTA.** Let $\mathcal S_n(\mathbf z)$ denote the indices of the largest $n$ components and define

$$a_r=\frac{\eta}{n}\mathbf1\{r\in\mathcal S_n(\mathbf z)\}.$$

Exactly $n$ dendrites are updated, so

$$\sum_r a_r^2=\frac{\eta^2}{n},\qquad
\sum_r a_rz_r=\frac{\eta}{n}\sum_{j=1}^n Z_{(j)}.$$

At $Q=I$ we have $k_r=1$ for all $r$. Insert these identities into (R19):

$$
\left.\dot R\right|_{Q=I}=\frac1{N_d}\left[
\eta^2\left(1-\frac1n\right)
-\frac{2\eta}{n}\mathbb E\sum_{j=1}^n Z_{(j)}\right].
$$

For $n=1$ and $N_d>1$, $\mathbb E[Z_{(1)}]>0$, so the slope is negative. This proves (R21) and shows explicitly why hard rank-based competition can initially decorrelate weights.

## K. Computing coefficients for the original rules

All required coefficients are expectations over **$N_d$ independent standard Gaussian numbers**, not over the full synaptic matrix. For the manuscript's independent threshold rule, let $L=b-\lambda$ and $T=b+\kappa$, with $L<T$, and define $f(u)=\mathbf1\{u>L\}[T-u]_+$. Then, with $\varphi$ the standard normal density,

$$
\boxed{\begin{aligned}
\mu&=\int_L^T(T-u)\varphi(u)\,du,\\
D&=\int_L^T(T-u)^2\varphi(u)\,du,\\
\chi&=\int_L^T u(T-u)\varphi(u)\,du.
\end{aligned}} \tag{P18}
$$

For $n$-WTA, let $\Phi$ be the standard normal CDF and $\bar\Phi(u)=1-\Phi(u)$. Given that dendrite 1 has activation $u$, it belongs to the largest $n$ exactly when at most $n-1$ of the other $N_d-1$ activations exceed $u$. Thus

$$
\boxed{
D_{\mathrm{WTA}}=
\int_{-\infty}^{b+\kappa}
(b+\kappa-u)^2\varphi(u)
\sum_{j=0}^{n-1}\binom{N_d-1}{j}
\bar\Phi(u)^j\Phi(u)^{N_d-1-j}\,du.
} \tag{P19}
$$

This gives the **uncorrelated-input** forgetting coefficient for the original target-based n-WTA rule directly by one-dimensional quadrature. The remaining $C_o$, $F_o$, and $\Lambda$ values for n-WTA or variable-target learning are ordinary finite-$N_d$ Gaussian expectations and can be obtained by Monte Carlo or appropriate low-dimensional order-statistic reductions. Their discontinuities pose no obstacle to (R10), which differentiates the **Gaussian density**, not the rule itself.

## L. Consistency checks and limitations

- **No shared axons:** $\alpha_{rs}=0$ and geometrically disjoint supports imply $Q_{rs}=0$ for those pairs. (R12) respects this invariant subspace. For interacting rules, the general partial-sharing equation (P14) likewise gives no direct overlap on disjoint supports.
- **Full sharing:** $P_r=I_{N_s}$, $\alpha_{rs}=1$, $T_{srk}=Q_{sk}$; (P14) exactly reduces to the fully shared formula (R7).
- **Constant angular amplitude:** $a_r=\eta/N_d$ gives $\dot Q_{rs}=\alpha_{rs}\eta^2/N_d^2-Q_{rs}\eta^2/N_d^2$, confirming (R12) and (R18) in their common domain.
- **Local amplitude as an interacting special case:** For $a_r=f(u_r)$, $F_o=0$, $\Lambda=\chi^2-D$, and the general full-sharing linearization (R16) reproduces the closed non-interacting result (R14) at $\alpha=1$.
- **Initial hard one-winner competition:** (R21) is negative for $n=1$ and $N_d>1$, consistent with selection of the most aligned branch and with the universal $B$ formula.
- **Finite $N_s$:** The angular rotation is exact, but the drift/diffusion and closure equations omit corrections that vanish in the $N_s\to\infty$ limit. They should be tested against simulations at the manuscript's actual $N_s$.
- **Weak correlations, partial inputs:** No theorem identifies $T_{srk}$ from $Q$ alone. The mask-mixing assumption in (P15) is deliberately separate and could fail under learned synaptic clustering.
- **Memory capacity:** $D$, $B$, $\Lambda$, and $R$ characterize interference and weight-correlation dynamics, but neuronal capacity additionally needs the post-learning overlap distribution and the joint old-pattern response after the dendritic nonlinearity.

**Bottom line.** The robust results are the spherical diffusion law, the full-sharing finite-$N_d$ Gaussian-moment closure, the non-interacting arbitrary-sharing overlap equation, and the general weak-correlation collective equation. The only step that needs an independent anatomical/mixing hypothesis is propagation of the interacting collective equation beyond initial time with partially overlapping synaptic supports.
