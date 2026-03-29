# Architecture for the Strong-Coupling Free Energy

## Target

Reproduce the first-order strong-coupling expansion of the binary symmetric HMM:

$$\log P(x_{1:T}) = \text{const} + \underbrace{\log\cosh(h\,m_T)}_{\text{frozen sector}} + \underbrace{y\sum_{\tau=1}^{T-1}\frac{\cosh\!\big(h(2m_\tau - m_T)\big)}{\cosh(h\,m_T)}}_{\text{single-wall correction}} + O(y^2),$$

where $m_\tau = \sum_{t=1}^\tau (1 - 2x_t)$, $h = \frac{1}{2}\log\frac{a}{1-a}$, $y = \frac{\varepsilon}{1-\varepsilon}$.

We ask: can this be realized in the framework of the LLM Embeddings note (Section 10)?

$$\text{embed} \to \text{affine map} \to \text{bilinear form (with quadratic head)}$$

---

## Recap: the Gaussian architecture (Section 10 of LLM Embeddings note)

For a permutation-invariant Gaussian law on counts, the exact architecture is:

$$x_t = e_{w_t}, \qquad s = \sum_t x_t, \qquad h = Ws + b, \qquad \log p = c - \frac{1}{2}\|h\|^2.$$

This is a Deep Sets model: embed $\to$ sum pool $\to$ affine $\to$ squared norm. The NN perturbation from the Proposal extends this to a bilinear form:

$$\log p = c - \frac{1}{2}\mathbf{z}^\top W \mathbf{z}, \qquad W = \mathbf{1} + \lambda \Lambda_{+1},$$

where $\Lambda_{+1}$ is the shift-by-one (nearest-neighbor) matrix.

---

## Term 1: $\log\cosh(h\,m_T)$ — the frozen sector

This depends only on $s = m_T = \sum_t \phi_t$ (the sum-pooled embedding), so it is permutation-invariant. But it is **not** a squared norm:

$$\log\cosh(x) = \begin{cases} \frac{1}{2}x^2 + O(x^4) & |x| \ll 1, \\ |x| - \log 2 & |x| \gg 1. \end{cases}$$

The squared-norm head $\frac{1}{2}\|hs\|^2 = \frac{h^2}{2}m_T^2$ is only the small-argument (Gaussian) limit. In the strong-coupling regime where $hm_T \sim O(T)$, the quadratic approximation fails.

However, $\log\cosh$ can be written as a **LogSumExp**:

$$\log\cosh(hm_T) = \log\!\Big(\frac{e^{+hm_T} + e^{-hm_T}}{2}\Big) = \text{LSE}(+hm_T,\; -hm_T) - \log 2.$$

This is the log-partition function of a two-class softmax with scores $\pm hm_T$. It reflects the mixture structure of the frozen sector: $P \propto \pi_A \prod e_A(x_t) + \pi_B \prod e_B(x_t)$.

**Realization**: replace the squared-norm head with a LogSumExp of two linear heads applied to the pooled sum:

$$\text{embed} \to \text{sum pool} \to (\pm h) \to \text{LogSumExp}.$$

This is a minimal modification: one extra linear head and a soft-max instead of a square.

---

## Term 2: $y\sum_\tau \frac{\cosh(h(2m_\tau - m_T))}{\cosh(h\,m_T)}$ — the wall correction

Using the sigmoid decomposition (Section 06):

$$f_1 = y\Big[\sigma(-2hm_T)\sum_\tau e^{2hm_\tau} + \sigma(2hm_T)\sum_\tau e^{-2hm_\tau}\Big],$$

where $\sigma$ is the logistic sigmoid.

Each $e^{2hm_\tau}$ is a prefix product:

$$e^{2hm_\tau} = \prod_{t=1}^{\tau} e^{2h\phi_t} = \prod_{t=1}^{\tau} r_t, \qquad r_t = \begin{cases} e^{2h} & x_t = 0, \\ e^{-2h} & x_t = 1. \end{cases}$$

The sum $\sum_\tau e^{2hm_\tau}$ is a **sum of prefix products** — a cumulative product pooling.

**Can this be a bilinear form?** No. Here is why:

A bilinear form $\mathbf{z}^\top W \mathbf{z} = \sum_{s,t} W_{st}\, z_s z_t$ is a degree-2 polynomial in the embeddings. But $e^{2hm_\tau} = e^{2h(\phi_1 + \cdots + \phi_\tau)}$ is an **exponential** of a sum — it contains all powers of the $\phi_t$'s. No choice of context matrix $W$ can reproduce this.

**Realization**: compute $e^{2hm_\tau}$ via a **multiplicative scan** (equivalently, a linear RNN in log-space):

$$\log R_\tau = \log R_{\tau-1} + 2h\phi_t, \qquad R_\tau = e^{2hm_\tau}.$$

Then aggregate: $\Sigma_\pm = \sum_\tau R_\tau^{\pm 1}$.

---

## The architecture

```text
Input:  x = (x_1, ..., x_T) ∈ {0,1}^T

1. EMBED:         φ_t = 1 - 2x_t                                (scalar, ±1)

2. SUM POOL:      s = Σ_t φ_t = m_T                             (total magnetization)

3. FROZEN HEAD:   f_0 = LogSumExp(+hs, -hs) - log 2             (= log cosh(hs))

4. SCAN:          R_τ = exp(2h · Σ_{t≤τ} φ_t)                   (prefix product, via
                  = R_{τ-1} · exp(2h · φ_τ)                      multiplicative RNN)

5. AGGREGATE:     Σ_+ = Σ_τ R_τ,    Σ_- = Σ_τ R_τ^{-1}

6. GATED READOUT: f_1 = y · [σ(-2hs) · Σ_+ + σ(+2hs) · Σ_-]

Output: log P = const + f_0 + f_1
```

**Parameters**: $h$ (emission strength), $y$ (wall fugacity) — 2 learnable scalars.

---

## Bottlenecks: what prevents a pure bilinear realization

### Obstruction 1: LogSumExp vs squared norm

The frozen-sector term $\log\cosh(hm_T)$ is the log-partition function of a 2-state mixture, not a Gaussian. It matches the squared norm only when $hm_T \ll 1$ (Gaussian regime). In the strong-coupling regime, the saturation $\log\cosh(x) \to |x|$ at large $|x|$ is essential — it reflects that one hidden state completely dominates.

**Severity**: moderate. LogSumExp is a standard operation. The architectural change is minimal: replace $\frac{1}{2}\|\cdot\|^2$ with LSE.

### Obstruction 2: prefix products cannot be bilinear

The wall correction involves $e^{2hm_\tau}$, which is multiplicative in the token embeddings. A bilinear form $\mathbf{z}^\top W \mathbf{z}$ is additive (degree 2). The exponential generates all degrees simultaneously.

**Severity**: fundamental. This is the core reason the bilinear framework breaks down in the strong-coupling regime. The multiplicative structure is intrinsic to the domain-wall picture: each token multiplicatively updates the likelihood ratio between the two hidden states.

---

## Weak-coupling recovery

When $h \ll 1$, both obstructions disappear:

$$\log\cosh(hm_T) \approx \frac{h^2}{2}m_T^2 \quad \text{(squared norm)},$$

and for the wall correction, expand $\frac{\cosh(h\delta_\tau)}{\cosh(hm_T)} \approx 1 + \frac{h^2}{2}(\delta_\tau^2 - m_T^2) + \cdots$ where $\delta_\tau = 2m_\tau - m_T$. This gives:

$$f_1 \approx y(T-1) + 2yh^2 \sum_{\tau=1}^{T-1} m_\tau(m_\tau - m_T) + \cdots$$

Using $m_\tau = \sum_{s \leq \tau} \phi_s$ and $m_\tau - m_T = -\sum_{t > \tau} \phi_t$:

$$\sum_\tau m_\tau(m_\tau - m_T) = -\sum_{1 \leq s < t \leq T} (t - s)\,\phi_s\,\phi_t.$$

This **is** a bilinear form, but with a linearly growing kernel $K(s,t) = |t - s|$, not a pure nearest-neighbor term. The context matrix is:

$$W_{st} \propto |t - s| \quad \text{for } s \neq t.$$

The nearest-neighbor ($|t-s| = 1$) piece matches the Proposal's $\lambda \sum z_t z_{t+1}$. But there are also next-nearest-neighbor terms with weight 2, weight 3, etc. So:

- The **Proposal's NN architecture** captures the **shortest-range piece** of the weak-coupling expansion.
- The full weak-coupling correction has **algebraically decaying** (in fact, linearly growing) long-range bilinear interactions.
- In the strong-coupling regime, even the bilinear framework is insufficient — the exponential structure takes over.

---

## Summary

| Regime | Architecture | Head |
| --- | --- | --- |
| Gaussian ($hm_T \ll 1$, many switches) | embed $\to$ sum pool $\to$ affine | $\frac{1}{2}\|Ws+b\|^2$ (squared norm) |
| Weak NN correction | embed $\to$ bilinear $\mathbf{z}^\top(\mathbf{1}+\lambda\Lambda_{+1})\mathbf{z}$ | squared norm + NN bilinear |
| Strong coupling ($\varepsilon \ll 1$, rare switches) | embed $\to$ multiplicative scan $\to$ gated aggregation | LogSumExp + sigmoid-gated exp-pooling |

The strong-coupling regime demands:

1. A **LogSumExp** readout (reflecting the mixture structure) instead of a squared norm
2. A **multiplicative scan** (prefix products) instead of sum pooling — this is the hidden-state likelihood ratio, and it is intrinsically sequential
3. **Sigmoid gating** by the total evidence — selecting which hidden state dominates globally

These are exactly the operations that a minimal RNN or state-space model would perform, lending support to the view that sequential architectures (RNNs, SSMs) are naturally matched to the sticky/metastable regime, while bag-of-words / Deep Sets models are matched to the Gaussian regime.
