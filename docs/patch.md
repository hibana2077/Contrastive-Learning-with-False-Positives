Below is a **drop-in “repair patch”** (English, Markdown) that makes (1) $ \alpha^{\mathrm{BBP}}_c / \alpha^{\mathrm{fail}}*c$ **explicit** by committing to a concrete spiked model, and (2) upgrades **Lemma 6** from informal to a **fully citable statement** with explicit $g*\pm$, a precise $\mathrm{Spec}*K$, and an explicit bound on $\Delta*{\mathrm{red}}$.

> Context: this patch matches the draft’s existing definitions of the effective strength $ \alpha(p) $  and the “InfoNCE $\to$ second-order sandwich” idea around Lemma 6 , including the weighted cross-covariance $C(W)$ and weight $\phi_{\tau,N}$ .

---

## PATCH 1 — Make $ \alpha^{\mathrm{BBP}}_c / \alpha^{\mathrm{fail}}_c$ concrete (commit to a spiked model)

### Insert after Definition 1 (effective signal strength) 

#### Concrete reduced spiked model (rectangular additive spike)

Define the (population-whitened) **weighted cross-view matrix estimator** (a sample analogue of the weighted cross-covariance in Section 3.2 ):
$$
\widehat{Y}(W)
:= \frac{1}{\sqrt{M}}\sum_{i=1}^M \phi_{\tau,N}^{(i)} , \big(W\tilde{x}^{(1)}_i\big)\big(\tilde{x}^{(2)}*i\big)^\top
\in \mathbb{R}^{m\times d},
$$
where $\tilde{x} = \Sigma^{-1/2}x$ and $\phi*{\tau,N}\in(0,1)$ is the InfoNCE softmax weight already defined in the draft .

In the high-dimensional regime $d,m\to\infty$ with $d/m\to\gamma\in(0,\infty)$ and fixed $K$, we **model** the reduced matrix (after centering and negligible remainder terms) by the **rank-$K$ additive spiked rectangular model**
$$
Y = \theta(p), U V^\top + \frac{1}{\sqrt{m}}G,
\qquad
G_{ij}\stackrel{i.i.d.}{\sim}\mathcal{N}(0,1),
$$
where $U\in\mathbb{R}^{m\times K}$ and $V\in\mathbb{R}^{d\times K}$ have orthonormal columns, and $\mathrm{col}(V)=S$ (the semantic subspace). The spike strength is parameterized as
$$
\theta(p) := \lambda_{\tau,N},\sqrt{\alpha(p)} ,
\qquad
\alpha(p)=(1-2p)^2\cdot\frac{\rho}{\sigma^2},
$$
with $\alpha(p)$ exactly as in Definition 1 . Here $\lambda_{\tau,N}>0$ is a deterministic reduction constant induced by the weighting/normalization (equal to $1$ in the “tight reduction” scaling limit; otherwise it can be carried explicitly as below).

> Practical note (allowed by COLT style): you can either (i) **set $\lambda_{\tau,N}=1$** by defining the reduced object with a matching normalization, or (ii) keep $\lambda_{\tau,N}$ explicit as a known scalar depending on $(N,\tau)$.

---

### Proposition (explicit BBP threshold for this spiked model)

For the model $Y=\theta U V^\top + \frac{1}{\sqrt{m}}G$ with $d/m\to\gamma$ and fixed rank $K$, an **information-carrying outlier** (and non-trivial overlap of the top-$K$ right singular space with $S$) appears **iff**
$$
\theta(p)>\gamma^{1/4}.
$$
Equivalently,
$$
\alpha(p) > \alpha^{\mathrm{BBP}}_c(\gamma,N,\tau,K)
\quad\text{where}\quad
\alpha^{\mathrm{BBP}}*c(\gamma,N,\tau,K):=\frac{\gamma^{1/2}}{\lambda*{\tau,N}^2}.
$$
Conversely, in the subcritical regime
$$
\theta(p)<\gamma^{1/4}
\quad\Longleftrightarrow\quad
\alpha(p)<\alpha^{\mathrm{fail}}_c(\gamma,N,\tau,K),
\qquad
\alpha^{\mathrm{fail}}*c(\gamma,N,\tau,K):=\frac{\gamma^{1/2}}{\lambda*{\tau,N}^2},
$$
the top-$K$ right singular space is asymptotically uninformative about $S$.

Moreover, when $\theta>\gamma^{1/4}$, the top singular value converges to the explicit outlier location
$$
\sigma_{\mathrm{out}}(\theta)
=============================

\sqrt{\frac{(\theta^2+1)(\theta^2+\gamma)}{\theta^2}},
$$
while the bulk edge stays at $1+\sqrt{\gamma}$.

---

### Replace Definition 2 (phase boundaries) with the explicit version

Using the explicit thresholds above,
$$
p^- = \frac{1}{2}\left(1-\sqrt{\alpha^{\mathrm{BBP}}_c(\gamma,N,\tau,K)\cdot \frac{\sigma^2}{\rho}}\right),
\qquad
p^+ = \frac{1}{2}\left(1-\sqrt{\alpha^{\mathrm{fail}}_c(\gamma,N,\tau,K)\cdot \frac{\sigma^2}{\rho}}\right),
$$
clipped to $[0,\tfrac12]$ as already stated in the draft .

> If you adopt the “tight reduction normalization” $\lambda_{\tau,N}=1$, this becomes simply $\alpha_c=\sqrt{\gamma}$ and thus $p^- = p^+$ (asymptotically sharp).

---

## PATCH 2 — Make Lemma 6 formal, with explicit $g_\pm$, $\mathrm{Spec}*K$, and $\Delta*{\mathrm{red}}$

### Replace the current Lemma 6 (informal)  with the following

#### Lemma 6 (InfoNCE-to-second-order sandwich, formal)

Let $W\in\mathcal{W}*{m,d}$, and let $\phi*{\tau,N}$ and the weighted whitened cross-covariance
$$
C(W) := \mathbb{E}\big[\phi_{\tau,N}(r,r^+,r^-_{1:N}),\tilde{x}^{(1)}(\tilde{x}^{(2)})^\top\big]
$$
be as defined in Section 3.2 . Define the spectral summary
$$
\mathrm{Spec}*K(C) := \sum*{i=1}^K \sigma_i(C),
$$
i.e., the **Ky–Fan $K$-norm** (sum of the top-$K$ singular values).

Assume the following “reduction regularity” conditions hold:

1. **Negative concentration:** for $b_j:=\langle r,r^-*j\rangle$,
   $$
   \mathbb{E}\Big[\max*{1\le j\le N} |b_j|\Big];\le; c_0\sqrt{\frac{\log N}{m}}.
   $$
2. **Norm/normalization concentration:** there is $c_1>0$ such that
   $$
   \mathbb{E}\Big[\big||Wx|_2^2-\mathbb{E}|Wx|_2^2\big|\Big];\le; c_1\sqrt{m},
   $$
   implying an $O(m^{-1/2})$ control on the error induced by output normalization.

Then there exists a universal constant $C>0$ such that, for all $W\in\mathcal{W}*{m,d}$,
$$
g*-\big(\mathrm{Spec}*K(C(W))\big)-\Delta*{\mathrm{red}}
;\le;
L(W)
;\le;
g_+\big(\mathrm{Spec}*K(C(W))\big)+\Delta*{\mathrm{red}},
$$
where the two monotone functions $g_\pm:\mathbb{R}*{\ge 0}\to\mathbb{R}$ are explicitly
$$
g*-(s) := \log(N+1) - \frac{s}{\tau},
\qquad
g_+(s) := \log(N+1) - \frac{s}{\tau} + \frac{2}{\tau^2},
$$
and the reduction gap is bounded by
$$
\Delta_{\mathrm{red}}(N,\tau,d,m)
;\le;
C\left(
\frac{1}{\tau^2}
+
\sqrt{\frac{\log N}{m}}
+
\frac{1}{\sqrt{m}}
\right).
$$

**Interpretation.** In regimes where $\tau\to\infty$ and $m\to\infty$ with $\log N=o(m)$, we have $\Delta_{\mathrm{red}}=o(1)$, so **near-minimizers of InfoNCE correspond to near-maximizers of $\mathrm{Spec}_K(C(W))$**, making the reduction operational and citable. This matches the draft’s intent that $\Delta_{\mathrm{red}}$ vanishes in “tight reduction” regimes .

---

### (Optional, but helps reviewers) One-line definition of what “CCA-like” means here

Right after $\mathrm{Spec}_K(C)$, add:

> Equivalently, $\mathrm{Spec}_K(C)$ is the sum of the top-$K$ canonical correlations between the two whitened views when restricted to $K$-dimensional subspaces, since singular values of a whitened cross-covariance are canonical correlations.