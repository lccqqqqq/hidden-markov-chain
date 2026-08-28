# Compression experiment log

Running log of experiments. Design / method definitions live in `compression_survey.md`;
this file is results only, newest entry at the bottom. Each entry records: date, model,
method + exact settings, data, command, numbers, and what was learned.

Conventions: losses in nats/token on the 6 250-sequence test split of
`data/datasets/cylinder_graph_hmm` (ctx 16, vocab 48). Reference lines for that split:

| reference | value |
|---|---|
| Bayes-optimal predictor at ctx 16 (exact, `evaluate.bayes_optimal`) | 2.4724 |
| entropy rate hₓ (`evaluate.entropy_rate`, 2×10⁵ tokens) | 2.4397 |
| ln 48 (uniform prediction) | 3.871 |
| exact predictor size (non-zeros of E) | 864 dof = 3 456 bytes fp32 |

Note on the 864: the joint emission tensor E[j,i,k] has 864 non-zeros = 18 states × 16
emitted tokens × 3 reachable next states. The process is **not** unifilar (H(S′|X,S) ≈ 0.75
nats, see the `entropy_rate_theory_estimate` docstring in `src/hmm.py`), so the exact
predictor must carry a genuine 18-dim belief state; 864 is the count of free entries of
the generator, i.e. the parameter floor for an exact forward-algorithm predictor.
(An earlier draft of this log wrongly called the process unifilar from the coincidence
48·18 = 864.)

Models:

| tag | run dir | params | fp32 bytes | test loss |
|---|---|---|---|---|
| d64 | `models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN` | 206 128 | 824 512 | 2.4761 |
| d32 | `models/cylinder_graph/20260828_140456_L4_d32_H2_full_noLN` | ~50k | — | (not yet evaluated) |

(`tie_word_embeddings: True` in the configs is ignored by TransformerLens; both runs have an
untied `unembed.W_U`.)

---

## 2026-08-28 — RTN post-training quantization, d64

**Method.** RTN (`compress/rtn.py`), weight-only, fake-quantized, all of
W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U; biases fp32. Full grid over
bits ∈ {16, 8, 6, 5, 4, 3, 2, 1} × {per_channel, per_tensor} × {asymmetric, symmetric}.
No calibration data, no fine-tuning.

**Command.**
```
uv run python src/compress/run_quant_sweep.py models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN \
    --bits 16 8 6 5 4 3 2 1 --out results/compression/L4_d64_rtn.csv
```
Full 30-row grid: `results/compression/L4_d64_rtn.csv` (+ `.refs.json`).

**Best setting per width** (per-channel, asymmetric won at every width):

| bits | bytes | ×HMM bytes | loss | Δ vs fp32 | Δ vs Bayes | last-position loss |
|---|---|---|---|---|---|---|
| fp32 | 824 512 | 239 | 2.4761 | — | +0.0037 | 2.4670 |
| 16 | 424 576 | 123 | 2.4761 | 0.0000 | +0.0037 | 2.4670 |
| 8 | 220 800 | 64 | 2.4762 | +0.0001 | +0.0038 | 2.4671 |
| 6 | 169 856 | 49 | 2.4767 | +0.0005 | +0.0043 | 2.4680 |
| 5 | 144 384 | 42 | 2.4786 | +0.0025 | +0.0062 | 2.4702 |
| 4 | 118 912 | 34 | 2.4862 | +0.0101 | +0.0138 | 2.4790 |
| 3 | 93 440 | 27 | 2.5401 | +0.0640 | +0.0677 | 2.5337 |
| 2 | 67 968 | 20 | 2.8976 | +0.4215 | +0.4252 | 2.8759 |
| 1 | 42 496 | 12 | 3.7781 | +1.3020 | +1.3057 | 3.7745 |

**Granularity / symmetry comparison** (Δ vs fp32, mnats):

| bits | ch-asym | ch-sym | tensor-asym | tensor-sym |
|---|---|---|---|---|
| 5 | 2.5 | 3.8 | 8.4 | 8.5 |
| 4 | 10.1 | 19.1 | 25.7 | 41.0 |
| 3 | 64 | 95 | 146 | 277 |
| 2 | 422 | 1310 | 955 | 1305 |

**Findings.**
1. The fp32 model sits 3.7 mnats above Bayes-optimal, so compression costs are cleanly
   resolvable down to ~1 mnat.
2. 8-bit is free; 6-bit costs 0.5 mnats; 5-bit costs 2.5 mnats (less than the model's own
   gap to Bayes); 4-bit costs 10 mnats. The knee is between 4 and 3 bits — where the
   small-model PTQ literature says it should be.
3. Per-channel > per-tensor and asymmetric > symmetric at every width; the weights are not
   zero-centred (ReLU, no LayerNorm), so a symmetric grid wastes half its levels.
4. 1-bit ≈ uniform prediction (3.78 vs ln 48 = 3.87): the model is destroyed, not degraded.
5. Bytes floor: even the 4-bit model is 34× the exact predictor. Quantization at fixed
   parameter count cannot approach the 864-dof floor; the per-channel scale/zero overhead
   (4 bytes per output column) is already 15% of the 2-bit model. Getting near the floor
   needs parameter-count reduction (low-rank, pruning, distillation into a smaller
   architecture) with quantization stacked on top.

**Next.** STE-QAT from the d64 checkpoint (needs the `train.py` inner loop factored into a
reusable function) to see how far the knee moves toward 2–3 bits; then GPTQ; then run the
same sweep on d32 to start the params-vs-bytes Pareto plot.
