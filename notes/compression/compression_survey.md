# Transformer compression for the cylinder-graph HMM model

*Research notes, 2026-08-28. No experiments run yet.*

## 1. Setting and the compression floor

| | |
|---|---|
| Trained models | L4 d32 H2 (~50k params), L4 d64 H4 (~206k params); vocab 48, ctx 16, ReLU, **no LayerNorm**, tied embeddings |
| Process | `cylinder_graph`, n=6 × depth=3 = **18 hidden states**, 48 tokens |
| Loss reference | Bayes-optimal at ctx-16 ≈ 2.47 nats/token; entropy rate hₓ ≈ 2.44 |

The Bayes-optimal predictor is the HMM forward algorithm on the 18-dim belief state. The
joint emission tensor E[j,i,k] has **864 non-zero entries** (18 states × 16 emitted tokens ×
3 next states; the process is non-unifilar, H(S′|X,S) ≈ 0.75 nats), i.e. ~240× fewer
parameters than the d64 model. The benchmark question is therefore: *how close to the minimal HMM can a
transformer's memory footprint get, at what cost in nats?*

Parameter budget of the d64 model, per layer (~49k):

- attention W_Q, W_K, W_V, W_O: 4·64² ≈ 16k (33%)
- MLP W_in, W_out: 2·64·256 ≈ 33k (67%)
- embedding 48·64 ≈ 3k and positional 16·64 ≈ 1k are negligible (~2% of total)

So MLP-targeted methods dominate; LLM-style vocab/embedding tricks are irrelevant here.

## 2. Technique survey, ranked by suitability

### 2.1 Quantization
Trade bits/weight for nats. The only axis that reduces **bytes at fixed parameter count**, so
mandatory for a VRAM-budget benchmark.

- **PTQ** (round-to-nearest, GPTQ, AWQ): cheap; ~lossless at 8 bit, small models degrade
  sharply below 4 bit.
- **QAT** (fake-quantize in the forward pass, straight-through estimator in the backward):
  the right tool here since retraining costs seconds. Reaches 4/3/2-bit and ternary.
- Practical points for this model: weight-only quantization (no LayerNorm ⇒ activation
  ranges are less controlled); per-output-channel scales, not per-tensor; mixed precision
  (embeddings/final layer at 8 bit, MLPs low-bit).

### 2.2 Low-rank / structured factorization
Best for parameter-count budgets.

- Plain SVD truncation of weights is known to be poor; use **activation-aware SVD**
  (weight by input covariance; ASVD, BALF) or Fisher-weighted SVD (§3).
- Physically motivated: the residual stream must carry a 17-dof belief simplex, so the
  effective rank of W_V, W_O, W_out is plausibly ≪ 64. **Diagnostic to do first**: singular
  value spectra of the trained weights.
- Attention: only the products W_Q W_Kᵀ and W_O W_V matter (rank ≤ d_head = 16 already);
  factorize the products, not the factors.

### 2.3 Structured pruning
- Unstructured sparsity does not reduce VRAM at this scale (index overhead). Prune
  **MLP neurons**, **heads**, **layers**.
- Second-order saliencies (OBS, Wanda-style |w|·‖x‖) are affordable exactly here.
- Layer dropping is a natural experiment: the HMM update is one recurrent step, so depth
  may be largely redundant.
- Always fine-tune after pruning.

### 2.4 Knowledge distillation
- Train small student on the teacher's full 48-way soft distributions.
- **Unusually strong here**: the exact Bayes-optimal teacher is available analytically
  (mixed-state presentation gives P(x_{t+1}|x_{≤t})). Distillation from the HMM removes
  sampling noise entirely and gives the cleanest size-vs-loss curve. Use as a control
  against (a) scratch-trained small models (the 135-config sweep) and (b) compressed big
  models.

### 2.5 Weight sharing / recurrence
- Share one block across the 4 layers (ALBERT / Universal Transformer). Interpretable: the
  optimal algorithm *is* a repeated belief update. 4× reduction in block parameters.

### 2.6 Not relevant here
KV-cache compression, MQA/GQA, speculative decoding, FlashAttention (throughput, not
memory-at-ctx-16); vocabulary pruning (embeddings ~2%); hashing / product quantization of
weights (only pays off at LLM scale).

## 3. How loss-landscape (Hessian) information is used  ← direction to push

### 3.1 The core expansion
Every method perturbs weights w → w + δw. At a trained minimum (∇L ≈ 0):

    δL ≈ ½ δwᵀ H δw,   H = ∂²L/∂w²

Flat directions (small curvature) can be pruned / rounded / truncated cheaply; stiff
directions must be preserved. Every "smart" compression method is a strategy for
estimating H, or an approximation of it, and exploiting it.

### 3.2 Which matrices each method actually uses

| Method | Landscape info | Matrix used |
|---|---|---|
| Magnitude pruning, plain SVD, RTN quantization | none | H ∝ I (implicit) |
| OBD (LeCun 1990), Wanda, AWQ, ASVD | diagonal curvature | diag(H), or diag(XXᵀ) as proxy |
| OBS (Hassibi & Stork 1993) | full curvature + compensation | H, H⁻¹ (global loss) |
| GPTQ / OBQ / SparseGPT | full curvature + compensation, **layerwise proxy** | H_layer = 2XXᵀ (d_in × d_in) |
| HAWQ (mixed-precision bit allocation) | curvature spectrum | top eigenvalues of H per layer |
| Fisher-weighted SVD, WoodFisher, Fisher pruning | curvature via Fisher | F = 𝔼[∇log p ∇log pᵀ], often K-FAC F ≈ A ⊗ G |
| QAT, distillation, post-prune fine-tuning | searches the landscape | none explicit |
| SAM-style training for compressibility | reshapes the landscape (flat minima) | reduces λ_max(H) |

Details:

- **OBD**: saliency sᵢ = ½ Hᵢᵢ wᵢ². Prune lowest.
- **OBS**: removing weight q with an optimal compensating update to the others:

      δw = −(w_q / [H⁻¹]_qq) H⁻¹ e_q,      δL = ½ w_q² / [H⁻¹]_qq

  Loss increase is governed by the *inverse* Hessian; remaining weights move along H⁻¹.
- **GPTQ**: OBS applied to the *layer reconstruction* objective ‖WX − ŴX‖², whose Hessian is
  exact and cheap, H = 2XXᵀ = input activation covariance (64×64 for us). Quantize column
  by column, propagate error to the unquantized columns through H⁻¹. Ignores error
  propagation through later layers and nonlinearities — a second-order treatment of the
  wrong function.
- **Wanda / AWQ / ASVD**: same principle with H ≈ diag(XXᵀ); |wᵢⱼ|·‖xⱼ‖ is OBD's saliency
  with that diagonal. No inverse needed.
- **Fisher methods**: F is PSD and equals the Gauss–Newton part of H; for cross-entropy at
  the true distribution, F = H. K-FAC makes F⁻¹ cheap per layer. HAWQ assigns more bits to
  layers with larger top-Hessian eigenvalues.
- **Fisher-weighted SVD** (Hsu et al. 2022): minimize ‖√F ⊙ (W − W_r)‖ instead of
  ‖W − W_r‖ — truncate where the landscape is flat.
- **QAT**: gradient descent constrained to the quantization lattice via the straight-through
  estimator; the model migrates to a region where lattice points have low loss.
  Quantization-robust minima are flat minima ⇒ sharpness-aware training improves
  compressibility (landscape *shaping* rather than *measuring*).

### 3.3 Why this model is a good testbed for a new landscape-aware method
- 2×10⁵ parameters: exact Hessian-vector products and full per-layer Fisher are cheap; the
  approximations in the table are *optional*, so their cost can be measured directly.
- The data-generating process is known, so at the optimum F = H in expectation, and the
  exact Bayes-optimal predictor is available as a teacher — the Hessian of the *true*
  cross-entropy (not a sampled proxy) can be estimated.
- Open questions to exploit:
  1. How much does GPTQ's layerwise XXᵀ lose relative to the true global H?
  2. Hessian spectrum at the minimum: how many stiff directions? If ≈ the ~1k dof of the
     minimal HMM, that is a direct landscape measurement of the compression floor.
  3. Can H⁻¹-compensation (OBS) be done *globally* across layers here, where it is
     intractable at LLM scale?
  4. Quantization as projection onto a lattice under the H-metric (‖δw‖_H) rather than the
     Euclidean metric — a "metric-aware" rounding scheme.

## 4. Benchmark design
Axes: parameter count **and** bytes. Metric: validation cross-entropy (nats/token) with
horizontal lines at 2.47 (Bayes-optimal ctx-16) and 2.44 (hₓ). Curves:

1. Scratch-trained models at each size (existing sweep)
2. Big → structured pruning + fine-tune
3. Big → activation/Fisher-aware low-rank + fine-tune
4. Big → QAT at 8/4/3/2 bit, weight-only, per-channel (only curve on the bytes axis that
   separates from 1)
5. Distillation from the exact HMM teacher
6. Combinations (low-rank + 4-bit is usually the Pareto winner in the literature)

Caveat: published results are on ≥10⁸-param models and do not transfer cleanly to 10⁵;
expect PTQ to fail earlier and QAT / fine-tuning to be mandatory.

## References
- Surveys: ACM CSUR 10.1145/3728636; TACL 10.1162/tacl_a_00704; HuangOwen/Awesome-LLM-Compression
- Small models: "Revisiting Pruning vs Quantization for Small LMs", EMNLP Findings 2025
- Li et al. 2020, "Train Large, Then Compress", arXiv:2002.11794
- BALF, arXiv:2509.25136; low-bit survey arXiv:2409.16694
- LeCun et al. 1990 (OBD); Hassibi & Stork 1993 (OBS); Frantar et al. 2022 (GPTQ, OBQ);
  Sun et al. 2023 (Wanda); Dong et al. 2019 (HAWQ); Hsu et al. 2022 (Fisher-weighted SVD);
  Singh & Alistarh 2020 (WoodFisher)
- Belief-state geometry: arXiv:2405.15943, arXiv:2502.01954, arXiv:2506.01919

## 5. Named quantization methods — reference definitions

**Honesty principle.** Every "known method" we compare against must be implemented exactly
as defined in its paper, with the paper's defaults frozen in the constructor. Any deviation
(grid choice, ordering, granularity, calibration size, fine-tuning) is either (a) a named,
logged constructor argument, or (b) a *new method* living under a separate `experimental/`
namespace. Never improve a reference implementation in place: "GPTQ with a few sensible
tweaks" is no longer GPTQ, and results labelled with a paper's name must reproduce that
paper's algorithm. Shared protocol choices (biases kept fp32, embeddings 8-bit, calibration
drawn from the train split, test split never used for calibration) are recorded once and
applied identically to every method.

| Method | Reference | Defining algorithm | Landscape info | Canonical settings |
|---|---|---|---|---|
| **RTN** | none (baseline) | ŵ = s·(round(w/s + z) − z); s, z from min/max (asym) or max\|w\| (sym) | none | per-channel asym, or g128; state which |
| **OBQ** | Frantar & Alistarh, NeurIPS 2022 | exact OBS per row: greedy order by minimal δL, H⁻¹ compensation | layerwise H = 2XXᵀ, full inverse | — (superseded by GPTQ) |
| **GPTQ** | Frantar et al., ICLR 2023, arXiv:2210.17323 | layerwise ‖WX − ŴX‖²; columns in *fixed* order; OBS compensation via Cholesky of H⁻¹; grid = RTN min/max per row | layerwise H = 2XXᵀ | damp = 0.01·mean(diag H); blocksize 128; 128 calib seqs; `act_order` (descending diag H) is a documented option |
| **AWQ** | Lin et al., MLSys 2024, arXiv:2306.00978 | per-input-channel scaling W·diag(s), s = ‖x‖^α, α grid-searched per layer on recon. error, then RTN | diag(XXᵀ) only | weight-only, g128, α ∈ [0,1] in 20 steps |
| **STE-QAT** | Bengio et al. 2013 (STE) | forward uses ŵ, backward passes grad to w (`w + (ŵ−w).detach()`); scale fixed from min/max | searches landscape | fine-tune from checkpoint, low LR |
| **LSQ** | Esser et al., ICLR 2020 | STE-QAT with learned scale s; grad scaled by 1/√(N·Q_max) | searches landscape | learned s per channel |
| **BitNet b1.58** | Ma et al. 2024 | ternary weights, s = mean\|W\| (absmean), 8-bit activations, **trained from scratch** | searches landscape | ternary by fine-tuning is "ternary STE-QAT", *not* BitNet |
| **HAWQ / HAWQ-V2** | Dong et al., ICCV 2019 / NeurIPS 2020 | per-layer bit allocation from top Hessian eigenvalue (V1) or Hutchinson trace (V2) + ILP; rounding is RTN, then QAT | Hessian spectrum / trace | an allocation *policy* wrapping another quantizer |
| **SqueezeLLM** | Kim et al., 2023 | non-uniform codebook per row by *Fisher-weighted* k-means; outliers kept sparse | diag Fisher | closest prior art to "quantization under the H-metric" — read before designing anything new |
| SmoothQuant, OmniQuant, QuIP/QuIP#, AQLM, SpQR | various 2023–24 | activation smoothing / learned clipping / incoherence processing / vector quant. / outlier sparsity | mixed | define the 2-bit SOTA frontier; not implemented, cited if claiming 2-bit results |

Granularity caveat: all these papers report at ≥10⁸ params with g128 grouping. On 64-wide
matrices g128 is meaningless; **per-channel is the only comparable granularity** here and
must be stated in any writeup.

### 5.1 Code structure
```
src/compress/
  base.py          Quantizer(ABC): quantize(model, calib) -> model; bytes(model); name; citation
  rtn.py           RTN(bits, granularity, symmetric)
  gptq.py          GPTQ(bits, granularity, n_calib=128, damp=0.01, act_order=False, blocksize=128)
  awq.py           AWQ(bits, granularity, alpha_grid=20)
  qat.py           STEQAT(bits, granularity, epochs, lr);  LSQ(bits, epochs, lr)
  hawq.py          HAWQv2(bit_budget, base=...)
  experimental/    new methods only — compared against, never merged into, the above
src/evaluate.py    nats/token (total and per position), params, bytes; Bayes-optimal and hₓ lines
```
Rules: (1) reference constructors' defaults equal the paper's; deviations are named arguments
that appear in the results CSV; (2) where practical, validate each reference implementation
against a published number or reference repo (e.g. `auto-gptq`) before trusting it.

### 5.2 Implementation order
1. `evaluate.py` + **RTN** — no calibration, no training, ~50 lines; establishes the baseline
   and the bytes accounting every other method reuses.
2. **STE-QAT** — reuses the existing training loop; the second-simplest and the one that
   makes low-bit results possible at all at this model size.
3. **GPTQ** — needs activation hooks (XXᵀ) and the Cholesky compensation loop; the hooks are
   also the infrastructure for the Hessian experiments in §3.3.
4. AWQ, LSQ, HAWQ-V2 as needed for the comparison figure.

Results are kept separately in `experiment_log.md`.
