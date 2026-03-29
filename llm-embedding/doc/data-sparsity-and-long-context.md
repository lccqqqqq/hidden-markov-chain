# Data Sparsity and Long-Context Learning

## The puzzle

A training corpus of $N$ tokens yields $\sim N/k$ independent samples of $k$-grams. For a 1M-token context window ($k = 10^6$), even a trillion-token corpus gives only $\sim 10^6$ independent samples — far too few to estimate the joint distribution over a vocabulary of size $V \sim 10^5$. Yet models trained on such corpora demonstrably use long-range context. Why?

## Key finding: language is critical, not gapped

The naive expectation from 1D statistical mechanics — that a transfer matrix with a spectral gap $\Delta$ produces exponentially decaying correlations $\sim e^{-t/\xi}$, so that $O(\xi)$ tokens of context suffice — turns out to be **wrong for natural language**.

Lin & Tegmark (2017) showed that:

- A **probabilistic regular grammar** (= HMM) gives exponential MI decay, like a 1D Ising chain off-criticality.
- A **probabilistic context-free grammar** (PCFG) gives **power-law** MI decay, like a system at criticality.
- Empirical measurements of natural language consistently show **power laws**: $I(d) \sim d^{-\alpha}$ with $\alpha \approx 0.12$.

The recursive tree structure of syntax provides a "hidden dimension" that puts language at criticality — analogous to how a 2D Ising model can be critical while a 1D chain cannot.

| Property | 1D Ising (finite $\xi$) | Natural Language |
|---|---|---|
| MI decay | Exponential: $\sim e^{-t/\xi}$ | Power law: $\sim t^{-0.12}$ |
| Correlation length | Finite | Divergent |
| Screening by finite window | Complete | Incomplete (sublinear leakage) |
| Transfer matrix gap | Gapped | Gapless (critical) |
| Generative model | HMM / finite Markov | PCFG / recursive |
| Stat-mech analogue | 1D off-critical | Quasi-2D at criticality |

## The Hilberg conjecture

The total mutual information between two adjacent blocks of text of length $L$ scales as:

$$I(L) \sim L^\beta, \quad \beta \in (0, 1)$$

Empirical estimates: $\beta \approx 0.5$–$0.95$. This is **sublinear** (each doubling of context adds diminishing marginal information) but **unbounded** (there is always more to gain from longer context).

Validated empirically by the L2M paper (arXiv:2503.04725) using LLaMA 3.1 405B.

## Why small data works (four layers)

### 1. ~80% of structure is local

Shannon entropy rate is ~1 bit/char vs ~5 bits/char for random text. Local patterns (bigrams, syntax) capture most predictive structure. The BoW + nearest-neighbor model captures this regime: unigram/co-occurrence statistics give $e_i \cdot e_j = \Sigma_{ij}$, and bigram corrections give the dressed embedding $\delta e_i = (\lambda/2)\sum_j P_{ij}^{(b)} \tilde{e}_j$.

### 2. The long-range tail is real but thin

MI decays as $t^{-0.12}$ — incredibly slow, but each individual long-range correlation contributes very little per token. The cumulative $L^\beta$ grows, but sublinearly.

### 3. Low intrinsic dimensionality

Aghajanyan et al. (2021) showed pre-trained models have intrinsic dimension ~200: 200 parameters suffice for 90% of fine-tuning performance. The embedding manifold is vastly lower-dimensional than the combinatorial space of contexts.

### 4. Optimal context scales with data

For fixed dataset size, there is an optimal context length beyond which performance degrades (arXiv:2502.01481). The approximation loss from trying to learn long-range structure outweighs the Bayes risk reduction.

## Connection to the embedding theory

The BoW + NN framework gives a clean local picture:

- **BoW** ($\lambda = 0$): captures all co-occurrence structure. $O(N^2)$ parameters learnable from $O(N^2)$ token-pair observations.
- **NN perturbation** ($\lambda \neq 0$): captures bigram corrections. Dressed embedding absorbs bigram statistics: $e_i' = e_i + (\lambda/2)\sum_j P_{ij}^{(b)}(e_j - \bar{e})$.
- **Transfer matrix** $T_{ab} = e^{-\lambda \Sigma_{ab}}$: determines all $n$-gram correlations via $T^n$. If $T$ were gapped, correlations would decay exponentially. But language being critical means $T$ is gapless.

The effective number of independent degrees of freedom in a context of length $L$ is roughly $L^\beta \approx (10^6)^{0.5} \approx 1000$ — not $10^6$. That's why it's learnable.

## Architectural implications

- **Transformers** satisfy the L2M condition: KV-cache grows linearly with $L$, which exceeds $L^\beta$ for $\beta < 1$. They can access the full power-law tail.
- **Fixed-state models** (RNNs, SSMs with constant state size) fundamentally cannot capture perigraphic processes (Debowski). They need state size scaling with input length.
- **Retrieval-augmented generation** with a 4K context can match fine-tuned 16K context models on many tasks, confirming that most information is local.

## What long context actually captures at different scales

- **Positions 1–100**: co-occurrence + bigram structure. Captured by the BoW + NN model.
- **Positions 100–10,000**: power-law correlations from recursive syntax. Captured by transformer attention.
- **Positions 10,000–1M**: very weak correlations (topical coherence, narrative arc). Only $O(L^{0.5})$ additional bits. Useful for retrieval but marginal for next-token prediction.

## Shannon entropy: still decreasing

Modern LLM-based estimates (arXiv:2512.24969) show conditional entropy keeps decreasing up to at least 10,000 characters with no sign of plateauing — contradicting Shannon's 1948 hypothesis of plateau at ~100 characters. The true entropy rate may be below 1 bit/char.

## Open questions

1. Can the Hilberg exponent $\beta$ be estimated precisely enough to predict architectural requirements?
2. Are current transformers actually capturing the full power-law tail, or just the first few orders?
3. How to make length generalization robust (currently fragile, works to ~2.5x training length)?
4. Can the "tautological interaction" program (NN perturbation of Gaussian theory) be extended to capture power-law correlations, e.g., via a hierarchical/recursive generalization?

## Key references

- Lin & Tegmark, "Critical Behavior from Deep Dynamics" (2017) — [arXiv:1606.06737](https://arxiv.org/abs/1606.06737)
- L2M: Mutual Information Scaling Law (2025) — [arXiv:2503.04725](https://arxiv.org/abs/2503.04725)
- Context Length Scaling and Bounds (2025) — [arXiv:2502.01481](https://arxiv.org/abs/2502.01481)
- Aghajanyan et al., Intrinsic Dimensionality (2021) — [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- LLMs and the Entropy of English (2025) — [arXiv:2512.24969](https://arxiv.org/abs/2512.24969)
- Lin, Tegmark, Rolnick, "Why Does Deep and Cheap Learning Work?" (2017) — [arXiv:1608.08225](https://arxiv.org/abs/1608.08225)
- Mehta & Schwab, RG = Deep Learning (2014) — [arXiv:1410.3831](https://arxiv.org/abs/1410.3831)
- Debowski, "Is Natural Language a Perigraphic Process?" — [PMC:7512648](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512648/)
- Echeverria et al., "Language Design as Information Renormalization" (2021) — [Springer](https://link.springer.com/article/10.1007/s42979-021-01002-y)
