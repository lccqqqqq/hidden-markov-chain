# Time-Reversal Symmetry Experiment

## Theoretical Background

### Entropy Rate Invariance

For a stationary stochastic process $\{X_t\}$, the entropy rate is defined as:

$$h = \lim_{n \to \infty} \frac{1}{n} H(X_1, \ldots, X_n)$$

Since joint entropy $H(X_1, \ldots, X_n)$ is permutation-invariant (it depends only on the joint distribution, not the ordering), the entropy rate of the forward process $(\ldots, X_{t-1}, X_t, X_{t+1}, \ldots)$ is identical to that of the reversed process $(\ldots, X_{t+1}, X_t, X_{t-1}, \ldots)$.

**Consequence:** Any difference in transformer loss between forward and backward prediction cannot be explained by an inherent information-theoretic difference. The Bayes-optimal predictor with infinite context achieves identical loss in both directions.

### Reverse HMM Construction

Given a forward HMM with emission matrices $E[j, i, k] = P(x_t = j, s_{t+1} = k \mid s_t = i)$ and stationary distribution $\pi$, the reverse-time HMM has emission matrices:

$$E_{\text{rev}}[j, k, i] = \frac{\pi(i) \cdot E[j, i, k]}{\pi(k)}$$

This follows from Bayes' rule applied to the time-reversed chain. The reverse HMM satisfies:
- Same stationary distribution $\pi$
- Same entropy rate $h$
- Same observation marginals (by stationarity)

### Belief State Convergence and Finite-Context Effects

The key quantity that *can* differ is how quickly the belief state $P(s_t \mid x_{t-k}, \ldots, x_{t-1})$ converges as a function of context length $k$. This convergence rate depends on the spectral properties of the transition matrices, which are generally different for forward and reverse chains.

For the CylinderGraph HMM specifically:
- **Forward:** Within-ring transitions go clockwise (+1, +2 mod n), inter-layer transitions go forward (l → l+1 mod depth). The directed structure creates a mixing pattern where context quickly narrows down the current state.
- **Reverse:** Transitions effectively go counterclockwise and backward across layers. The mixing pattern may be different.

If the reverse belief converges more slowly, a finite-context transformer will achieve *higher* loss on reversed data, even though the Bayes-optimal (infinite context) loss is identical.

## Hypotheses

1. **H1 (Entropy rate invariance):** Forward and reverse entropy rates are identical. *(Verified theoretically and numerically — see analysis results.)*

2. **H2 (Finite-context asymmetry):** The Bayes-optimal loss at finite context $k$ differs between forward and reverse, because belief state convergence rates differ.

3. **H3 (Transformer learnability):** Transformers trained on reversed data will converge to a *higher* final loss than those trained on forward data (same architecture, same hyperparameters), reflecting the harder finite-context prediction problem.

4. **H4 (Architecture dependence):** The forward-reverse gap may depend on model capacity (depth, width) — larger models may close the gap by implementing longer effective context.

## Experimental Setup

### Forward sweep (existing)
- **Dataset:** `data/datasets/cylinder_graph_hmm/` (10M tokens, seq_length=16)
- **HMM:** CylinderGraph(n=6, depth=3, tokens_per_cluster=16, alpha=0.3, p=0.1)
- **Sweep:** 36-run grid over n_layer∈{1,2,4}, n_embd∈{4,8,16}, attn_only∈{T,F}, normalization∈{none,LN}
- **Config:** `config/base_config.yaml`, sweep: `config/sweeps/sweep_config.yaml`

### Reverse sweep (new)
- **Dataset:** `data/datasets/cylinder_graph_hmm_reversed/` (same data, time-flipped)
- **Same 36-run grid** as forward
- **Config:** `config/base_config_reversed.yaml`, sweep: `config/sweeps/sweep_config_reversed.yaml`
- **Submission:** `experiment/submit_sweep_reversed.sh`
- **W&B tags:** `["reversed", "time-reversal"]`

### N-gram baseline
Run `src/ngram_baseline.py` on both forward and reversed datasets for comparison:
```bash
python src/ngram_baseline.py --config config/base_config.yaml
python src/ngram_baseline.py --config config/base_config_reversed.yaml
```
(Note: the n-gram baseline reads `data_generator.save_dir` for the data path.)

## Theoretical Predictions

Run full analysis:
```bash
python src/reverse_hmm_analysis.py --num_samples 50000 --max_context 16
```

Preliminary results (500 samples, first 8 positions) show:
- **Entropy rates match** to machine precision (|diff| ~ 2e-15)
- **Belief convergence is slower for the reverse process** — KL from final belief is ~2-3x larger at each context position
- **Bayes-optimal loss** shows some finite-context asymmetry (noisy at n=500, needs more samples)

## Wandb Sweep Links

*(To be filled in after sweep creation)*

- Forward sweep: `TODO`
- Reverse sweep: https://wandb.ai/chuqiao-lin-university-of-oxford/hidden-markov-chain-src/sweeps/eib9duyf

## To-Do List (Post-Sweep Analysis)

- [ ] Run `src/reverse_hmm_analysis.py` with full 50k samples and save results
- [ ] Run n-gram baseline on reversed data and compare to forward
- [ ] After sweeps complete, download final val_loss for all 36×2 runs
- [ ] Plot forward vs reverse val_loss for each architecture configuration
- [ ] Compute gap Δ = loss_reverse - loss_forward for each config
- [ ] Test H3: Is Δ > 0 consistently across architectures?
- [ ] Test H4: Does Δ depend on model capacity (depth/width)?
- [ ] Compare transformer gap to Bayes-optimal gap at k=16
- [ ] Investigate whether specific architectural features (attention-only, LayerNorm) affect the gap
- [ ] Plot belief convergence curves (forward vs reverse) alongside transformer loss curves
- [ ] Write final analysis section with plots
