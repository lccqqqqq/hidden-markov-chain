# Analysis: Cagnetta et al., "Deriving Neural Scaling Laws from the Statistics of Natural Language"

**Paper:** arXiv:2602.07488 (Feb 2026), Cagnetta, Raventós, Ganguli, Wyart

## What the paper does

First quantitative theory predicting data-limited scaling law exponents for LLMs from measurable language statistics alone, with no free parameters. Two dataset-level exponents suffice:

- **$\gamma$**: decay of next-token conditional entropy with context length, $H_n - H_\infty \sim n^{-\gamma}$
- **$\beta$**: decay of pairwise token-token correlation strength with lag, $\|C(n)\|_{\text{op}} \sim n^{-\beta}$

The predicted scaling exponent is $\alpha_D = \gamma / (2\beta)$.

| Dataset | $\gamma$ | $\beta$ | Predicted $\alpha_D$ | Observed |
|---|---|---|---|---|
| TinyStories | 0.34 | 0.88 | 0.19 | matches |
| WikiText | 0.27 | 0.94 | 0.14 | matches |

## Key mechanism: data-dependent prediction time horizon

Given $P$ training tokens, the model can leverage context up to horizon $n^*(P)$, defined by when pairwise correlations become detectable above noise:

$$\|C(n)\|_{\text{op}} \sim n^{-\beta} = O(1/\sqrt{P}) \implies n^*(P) \sim P^{1/(2\beta)}$$

This is a **pairwise** (two-point) threshold — it requires detecting the top singular value of a $V \times V$ covariance matrix, not estimating $n$-gram frequencies.

## The scaling collapse

All $n$-gram loss curves $L_n(P)$ collapse onto a single universal curve under the rescaling:

$$L_n(P) = n^{-\gamma} \cdot \ell(P / n^{2\beta})$$

- Vertical scale: $H_n \sim n^{-\gamma}$ (conditional entropy at horizon $n$)
- Horizontal scale: $P_n^* \sim n^{2\beta}$ (data needed to resolve lag-$n$ correlations)

## The data sparsity question

### The puzzle

The $n$-gram loss $L_n = -\mathbb{E}[\log \hat{p}_\theta(x_{n+1} \mid x_{1:n})]$ is the cross-entropy of the trained model's prediction conditioned on $n$ tokens of context. For this to be meaningful, the model must learn a good conditional distribution $\hat{p}_\theta(x_{n+1} \mid x_{1:n})$.

But for large $n$ (say $n = 128$), the number of possible $n$-grams is $V^{128} \sim 10^{640}$, while the training set has at most $P \sim 10^8$ tokens. The model sees essentially every 128-gram **at most once**. Raw $n$-gram frequency estimation is impossible.

### How the paper handles this

The paper **does not** estimate conditional entropies from $n$-gram tables. They explicitly acknowledge this is infeasible:

> "Direct estimation of the conditional entropies $H_n$ from raw counts is computationally infeasible at the vocabulary sizes and horizons of interest."

Instead, they use trained autoregressive models as upper bounds: $L_n(P, M) \geq H_n$, with equality in the infinite-data limit. They observe convergence of $L_n$ with increasing $P$ for small $n$, and extrapolate.

### What actually resolves the sparsity

The data sparsity problem is **sidestepped by the architecture's inductive bias**, not solved by having enough data:

1. **Pairwise detection, not $n$-gram estimation.** To "use" context at lag $n$, the model only needs to detect that $C(n) \neq 0$. This requires $P \sim n^{2\beta}$ samples — for $n = 128$ with $\beta = 0.88$, that's $P_{128}^* \sim 128^{1.76} \sim 6000$ tokens, which is trivially small. The threshold is about the **top singular value** of a $V \times V$ matrix, not the full $n$-gram distribution over $V^n$ outcomes.

2. **Compositional generalization.** The transformer decomposes long contexts into local features (via attention layers) and composes them. Two 128-grams that share local structure produce similar predictions even if they are globally distinct. The model generalizes across contexts via learned representations, never needing to see the same $n$-gram twice.

3. **The "fast learning within the horizon" assumption.** The paper assumes that for $n \ll n^*(P)$, the model learns an effective predictor quickly. This is the critical assumption — it is **not derived** but empirically verified for small $n$ (Fig. 6: $n$-gram losses for $n \leq 12$ decay faster than the autoregressive scaling $P^{-\gamma/(2\beta)}$). For large $n$, it is taken on faith.

### The honest summary

- The $P/n^{2\beta}$ collapse tells you when **pairwise signals** become detectable, not when $n$-gram tables converge.
- The model doesn't need enough data for $n$-gram statistics. It needs enough data to detect pairwise correlations at lag $n$, plus an architecture that can compose local features into long-range predictions.
- The "fast learning" assumption does heavy lifting and is architecture-dependent. The authors explicitly note it would fail for kernel methods, shallow networks, and classical $n$-gram models.

### What remains unclear

- The theory is tested at $n^* \approx$ tens of tokens. Whether the horizon-limited regime holds at $n^* \sim 10^5$ (current SOTA) is an open question.
- The fast-learning assumption for large $n$ has no theoretical justification — it's an empirical observation at small $n$ extrapolated to large $n$.
- The paper hints at a possible superior universality class (e.g., CNNs with translational equivariance outperform transformers on hierarchical toy data) that could have larger scaling exponents, but this is speculative for natural language.

## Connection to the embedding theory

| Embedding theory concept | Cagnetta et al. equivalent |
|---|---|
| Covariance $\Sigma_{ij} = e_i \cdot e_j$ | Covariance matrix $C_{\mu,\nu}(n)$ at lag $n$ |
| Precision matrix $G = \Sigma^{-1}$ | Inverse covariance (not used directly) |
| Transfer matrix $T_{ab} = e^{-\lambda \Sigma_{ab}}$ | Top singular value $\|C(n)\|_{\text{op}}$ governs learnability |
| BoW captures co-occurrence | $n = 1$: unigram statistics |
| NN perturbation captures bigrams | $n = 2$: bigram correlations, first correction |
| Dressed embedding $\delta e_i = \sum_j P_{ij}^{(b)} \tilde{e}_j$ | Model learns to use lag-$n$ correlations compositionally |
| Language is critical (gapless transfer matrix) | $\beta < 1$: power-law decay, not exponential |

The key insight connecting the two frameworks: the data requirement for learning lag-$n$ structure is set by the **two-point function** $C(n)$ (equivalently, $\Sigma_{ij}$ in our notation), not by $n$-gram frequencies. The embedding theory provides the local structure (BoW + NN perturbation); the Cagnetta et al. theory provides the scaling of how quickly models can extend their effective horizon as data grows.
