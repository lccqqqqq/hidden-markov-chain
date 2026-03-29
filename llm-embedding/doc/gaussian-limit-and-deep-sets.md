# Token Embeddings from Long-Window Co-occurrence: Gaussian Limit and Deep Sets Architecture

*Reconstructed from a ChatGPT conversation (research-note.txt). The conversation developed the idea in three rounds: initial draft, corrected version with reviewer feedback, and a final collaborator-facing note with a LaTeX appendix.*

---

## Roadmap

Consider a large context window and forget token order, keeping only the count vector $N$. Since $N$ is a sum of many weakly dependent one-hot indicators, short-range dependence implies a multivariate CLT: the centered fluctuations $(N - Lp)/\sqrt{L}$ become Gaussian. The leading-order statistics of large windows are therefore encoded by a single matrix, the long-run covariance $\Sigma$. A factorization $\Sigma = EE^\top$ gives token vectors whose inner products reproduce these fluctuations. Low-dimensional embeddings appear when $\Sigma$ is well-approximated by its top eigenspaces; ordering and higher-order interactions enter only beyond this Gaussian level.

---

## 1. Setup

- Vocabulary $\mathcal{V} = \{1, \ldots, V\}$, stationary token process $(w_t)_{t \in \mathbb{Z}}$.
- Window of length $L$: count vector $N_i = \#\{t \in \{1,\ldots,L\} : w_t = i\}$, with $\sum_i N_i = L$.
- One-hot process: $Y_t \in \mathbb{R}^V$, $(Y_t)_i = \mathbf{1}\{w_t = i\}$, so $N = \sum_{t=1}^L Y_t$.
- Marginal: $p = \mathbb{E}[Y_t]$, $p_i = P(w_t = i)$.

**Goal:** Identify the natural notion of token embedding implied by the probability law of large context windows.

## 2. Long-window Gaussian limit

**Assumption:** The process has short-range dependence (multivariate CLT holds, covariance series absolutely summable).

Define lag-$\tau$ covariance matrices:
$$\Gamma(\tau) = \text{Cov}(Y_0, Y_\tau)$$

and the long-run covariance:
$$\Sigma = \sum_{\tau \in \mathbb{Z}} \Gamma(\tau)$$

Centered, rescaled count fluctuations:
$$X_L = \frac{N - Lp}{\sqrt{L}} \Rightarrow \mathcal{N}_H(0, \Sigma)$$

where $\mathcal{N}_H$ denotes a Gaussian supported on the hyperplane $H = \{x \in \mathbb{R}^V : \mathbf{1}^\top x = 0\}$.

The support restriction is forced by $\mathbf{1}^\top N = L$, hence $\mathbf{1}^\top X_L = 0$. Accordingly, $\Sigma \mathbf{1} = 0$.

## 3. Relation to co-occurrence statistics

$$\text{Cov}(N) = \sum_{s,t=1}^L \text{Cov}(Y_s, Y_t) = \sum_{\tau=-(L-1)}^{L-1} (L - |\tau|)\Gamma(\tau) = L\Sigma + o(L)$$

So $\mathbb{E}[NN^\top] = L^2 pp^\top + L\Sigma + o(L)$, splitting into:
- **Trivial rank-one mean structure:** $L^2 pp^\top$
- **Nontrivial fluctuation structure:** $L\Sigma$

### Important decomposition

$$\Sigma = \Gamma(0) + \sum_{\tau \neq 0} \Gamma(\tau)$$

where $\Gamma(0) = \text{diag}(p) - pp^\top$ is the multinomial baseline (present even for i.i.d. tokens). Genuine temporal dependence enters through the $\tau \neq 0$ terms.

## 4. Embeddings from covariance factorization

Eigendecompose: $\Sigma = U_r \Lambda_r U_r^\top$, with $\Lambda_r = \text{diag}(\lambda_1, \ldots, \lambda_r)$, $\lambda_a > 0$, $r = \text{rank}(\Sigma) \leq V-1$.

Define $E = U_r \Lambda_r^{1/2}$. Then the $i$-th row $e_i^\top$ satisfies:

$$\boxed{\Sigma_{ij} = e_i \cdot e_j}$$

**Key points:**
- This embedding is canonical at the level of second moments, up to orthogonal freedom.
- **Low-dimensional embeddings require truncation** — they arise only when $\Sigma$ is approximately low-rank. This is not automatic.
- Two dual viewpoints: *covariance embeddings* $e_i$ (encode pairwise geometry) and *whitened additive features* $\psi(i)$ (whose window sums have standard Gaussian statistics).

## 5. Gaussian effective theory

The Gaussian density on $H$:
$$P(X) \propto \exp\left(-\frac{1}{2} X^\top \Sigma^+ X\right), \quad X \in H$$

where $\Sigma^+$ is the Moore–Penrose pseudoinverse.

In whitened principal coordinates $q_a = u_a^\top X / \sqrt{\lambda_a}$:
$$P(q) \propto \exp\left(-\frac{1}{2}\sum_{a=1}^r q_a^2\right)$$

Each $q_a$ is an additive sum of centered token features:
$$q_a = \frac{1}{\sqrt{L}} \sum_{t=1}^L \psi_a(w_t), \quad \psi_a(i) = \frac{u_a(i) - \sum_j p_j u_a(j)}{\sqrt{\lambda_a}}$$

**The centering term is essential.** The Gaussian theory depends on the window only through additive sums of centered, whitened token features.

## 6. Relation to PCA, SVD, and PMI

- PCA of centered window-count vectors diagonalizes the empirical $\Sigma$. The construction is a principled large-window justification for PCA/SVD-type embeddings.
- **This is NOT the same as factorizing PMI.** For an i.i.d. process, $\Sigma$ is nonzero (due to $\Gamma(0)$), but nonzero-lag PMI vanishes. So diagonalizing $\Sigma$ $\neq$ factorizing PMI in general.
- Conceptually closer to **PCA/LSA** on large context windows than to SGNS/GloVe shifted-PMI factorizations.

## 7. Infinite-range form vs source of Gaussianity

The Gaussian for counts can be written as:
$$P(N) \approx \exp\left[-\frac{1}{2L}(N-Lp)^\top \Sigma^+ (N-Lp)\right]$$

Substituting $N_i = \sum_t \mathbf{1}\{w_t = i\}$ yields an all-to-all interaction $-\frac{1}{2L}\sum_{s,t} K(w_s, w_t)$ with mean-field scaling $1/L$.

**But Gaussianity comes from the CLT, not from the all-to-all form.** Counts are sums of weakly dependent contributions; short-range dependence ensures only the second cumulant survives at leading order.

## 8. What lies beyond embeddings

The Gaussian description omits:
- Ordering and positional information
- Higher-order cumulants: the $m$-th cumulant of $X_L$ scales as $O(L^{1-m/2})$
  - Third-order: $O(L^{-1/2})$
  - Fourth-order: $O(L^{-1})$
- Nonlinear multi-token interactions
- Large-deviation tails

**If the process has long-range dependence, heavy tails, or critical behavior, this scaling may fail.** The construction is a short-range, large-window theory.

## 9. Conclusion

> In the large-window limit, the centered token-count vector satisfies a multivariate CLT governed by the long-run covariance $\Sigma$. Factorizing $\Sigma = EE^\top$ gives a canonical second-order embedding. In whitened coordinates, the Gaussian theory depends only on additive sums of centered token features — making embeddings the natural first representation. What remains beyond this level are the non-Gaussian corrections: ordering, higher cumulants, and structured dependencies.

---

## Part II: Exact Architecture for the Gaussian Count Model

### The question

Given a permutation-invariant distribution that is exactly Gaussian (quadratic) in token counts, what is the minimal architecture that learns it?

### Answer: Deep Sets with quadratic head

The exact architecture is:

```
token IDs → embedding lookup → sum pooling → affine whitening → squared norm head → log-probability
```

Concretely:

1. **Embedding lookup:** $x_t = e_{w_t} \in \mathbb{R}^r$
2. **Sum pooling:** $s = \sum_{t=1}^L x_t = E^\top N$
3. **Affine whitening:** $h = L^{-1/2} \Lambda_r^{-1}(s - L\bar{s})$, where $\bar{s} = E^\top p$
4. **Quadratic energy:** $\log p(w_{1:L}) = c - \frac{1}{2}\|h\|^2$

This is exactly a **Deep Sets** model with a quadratic readout.

### Why this is exact

Since $\Sigma^+ = U\Lambda^{-1}U^\top$, the Gaussian energy rewrites as:
$$\frac{1}{2L}(N - Lp)^\top \Sigma^+(N-Lp) = \frac{1}{2L}(s - L\bar{s})^\top \Lambda^{-2}(s - L\bar{s}) = \frac{1}{2}\|h\|^2$$

The pooled sum $s = E^\top N$ is a **sufficient statistic** for the Gaussian model. When $r = V-1$ (no truncation), $N \mapsto s$ is injective on the count hyperplane.

### What's unnecessary

For the exact Gaussian setting: attention, recurrence, positional encoding are all unnecessary. Those become relevant only beyond the permutation-invariant quadratic regime.

### Gauge freedom

If embeddings are learned jointly with the head, there is a natural gauge freedom: $E \to ER$, $W \to R^{-1}W$ for any invertible $R$. Learned coordinates are identifiable only up to such transformations.

### Count-law vs sequence-law

The architecture above models the permutation-invariant *sequence* law. If one instead targets the count-vector law $p_{\text{count}}(N)$, the induced sequence probability includes a multinomial coefficient:
$$p_{\text{seq}}(w_{1:L}) = \frac{p_{\text{count}}(N)}{L! / \prod_i N_i!}$$

---

## Key corrections from the review

1. **Section 5 centering:** The whitened features $\psi_a(i)$ must be centered by subtracting $\sum_j p_j u_a(j)/\sqrt{\lambda_a}$, otherwise $Q_a$ has a nonzero mean of order $\sqrt{L}$.
2. **Low-dimensionality is not automatic:** Full embedding has dimension $r = V-1$. Low-dimensional embeddings require spectral concentration.
3. **PCA ≠ PMI factorization:** Diagonalizing $\Sigma$ is PCA/LSA on centered counts. It is *not* the same as the shifted-PMI factorization in SGNS/GloVe.
4. **$\Gamma(0)$ separation:** The multinomial baseline $\Gamma(0) = \text{diag}(p) - pp^\top$ is present even for i.i.d. tokens. Genuine dependence enters through $\tau \neq 0$ terms.
5. **Gaussianity from CLT, not max-entropy:** The Gaussian appears because of the CLT, not because one solves a maximum-entropy problem (though it is true that the Gaussian maximizes entropy given fixed covariance).
