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

## Running comparison table (update with every entry; last updated 2026-08-31, reference set complete)

Test cross-entropy (nats/token) vs bytes, d64 model. Best known per budget in **bold**.

**Bits axis** (206 128 params fixed):

| bytes | ×HMM | RTN | GPTQ | +act_order | AWQ | HAWQ(GPTQ) | HAWQ wide | STE-QAT | LSQ | HAWQ+QAT |
|---|---|---|---|---|---|---|---|---|---|---|
| 824 512 (fp32) | 239 | — | — | — | — | — | — | — | — | **2.4761** |
| 223 104 (8b) | 65 | 2.4762 | **2.4761** | | 2.4762 | | | | | |
| 172 160 (6b) | 50 | 2.4767 | **2.4761** | | | | | | | |
| 146 688 (5b) | 42 | 2.4786 | **2.4765** | | | | | | | |
| 121 216 (4b) | 35 | 2.4860 | 2.4782 | | 2.4846 | **2.4769** | | 2.4799 | 2.4791 | |
| 95 744 (3b) | 28 | 2.5415 | 2.4850 | 2.4832 | 2.5440 | 2.4822 | **2.4792** | 2.4859 | 2.4817 | 2.4795 |
| 82 944 | 24 | | | | | | 2.4876 | | | **2.4816** |
| 70 272 (2b) | 20 | 2.8589 | 2.5478 | 2.5341 | 2.8460 | | | 2.5175 | **2.4942** | |
| 44 800 (1b) | 13 | 3.7777 | | | | | | | | |

**Parameter axis** (fp32 weights; FWSVD and pruning + 1-epoch FT; superseded at most sizes by the R3.2 global-field points and E9 tying — see those entries):

| params | bytes | loss | method |
|---|---|---|---|
| 206 128 | 824 512 | 2.4761 | fp32 reference |
| 173 104 | 692 416 | **2.4772** | wanda-prune f=0.25 |
| 156 592 | 626 368 | **2.4778** | FWSVD ρ=0.5 |
| 140 080 | 560 320 | **2.4782** | prune f=0.5 (both criteria) |
| 107 056 | 428 224 | **2.4810** | wanda-prune f=0.75 |
| 81 520 | 326 080 | **2.4851** | FWSVD ρ=0.25 (dominates wanda f=0.875: 2.4862 @ 90 544) |
| 43 984 | 175 936 | **2.5156** | FWSVD ρ=0.125 |
| 25 216 | 100 864 | **2.6472** | FWSVD ρ=0.0625 |
| 864 | 3 456 | 2.4724 | HMM minimum |

**Combined bytes frontier** (both axes; the low-bytes end is all composed or learned-scale):

| bytes | loss | method |
|---|---|---|
| 146 688 | 2.4765 | GPTQ 5-bit |
| 95 744 | **2.4766** | measured-alloc + 1000 soft QAT steps (E4/R1.4 recipe) |
| 82 944 | **2.4786** | measured-alloc + 1000 soft QAT |
| 70 239 | **2.4836** | measured-alloc (incl. ternary) + 1000 soft QAT |
| 59 891 | **2.4953** | measured-alloc (incl. delete) + 1000 soft QAT |
| 49 987 | **2.5110** | measured-alloc + 1000 soft QAT |
| 44 908 | **2.5171** | measured-alloc + 1000 soft QAT |
| 39 953 | **2.5449** | measured-alloc + 1000 soft QAT |
| 34 877 | **2.5509** | measured-alloc + 1000 soft QAT (12 tensors deleted) |
| 3 456 | 2.4724 | HMM minimum (864 dof) |

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

---

## 2026-08-31 — Protocol amendment: true per-output-channel grids

Preparing GPTQ exposed an inconsistency: `_channel_view` grouped W_Q/W_K/W_V
(shape (head, d_model, d_head)) by d_head index *across* heads (16 groups) instead of per
(head, d_head) output channel (64 groups). GPTQ's per-row grid would have used the correct
grouping, making the comparison unfair. Fixed in `compress/base.py`; per-channel now means
per output channel for every matrix. RTN rerun under the corrected view
(`results/compression/L4_d64_rtn.csv` overwritten):

- ≥3 bits: changes ≤ 0.2 mnats (2.5401 → 2.5415 at 3-bit ch-asym); conclusions unchanged.
- 2-bit ch-asym improved 2.8976 → 2.8589; bytes up slightly (more scales, e.g. 95 744 vs
  93 440 at 3-bit).

## 2026-08-31 — GPTQ, d64

**Method.** GPTQ (`compress/gptq.py`), paper defaults: layerwise H = 2XXᵀ, sequential
propagation (quantized blocks generate calibration activations for later stages), damp =
0.01·mean(diag H), fixed column order, grid from original weights per output channel,
n_calib = 128 train sequences (seed 0). W_E/W_pos have one-hot inputs ⇒ H diagonal ⇒ GPTQ
≡ RTN there (applied as such). Lazy batching omitted — mathematically identical at
d_in ≤ 256. `act_order=True` (descending diag H) run as the documented variant at 2–3 bits.

**Command.**
```
uv run python src/compress/run_quant_sweep.py models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN \
    --methods gptq --bits 8 6 5 4 3 2 --granularity per_channel --out results/compression/L4_d64_gptq.csv
```

**Results** (per-channel asymmetric; RTN = corrected rerun; Δ in mnats vs fp32 2.4761):

| bits | bytes | ×HMM bytes | RTN Δ | GPTQ Δ | GPTQ act_order Δ |
|---|---|---|---|---|---|
| 8 | 223 104 | 65 | +0.1 | +0.0 | — |
| 6 | 172 160 | 50 | +0.6 | +0.0 | — |
| 5 | 146 688 | 42 | +2.5 | +0.4 | — |
| 4 | 121 216 | 35 | +9.9 | +2.1 | — |
| 3 | 95 744 | 28 | +65.4 | +8.8 | +7.1 |
| 2 | 70 272 | 20 | +382.8 | +71.6 | +58.0 |

**Findings.**
1. GPTQ shifts the whole curve ≈ 1 bit left: GPTQ-3 (2.4850) ≈ RTN-4 (2.4860); GPTQ-4
   (+2.1 mnats) is cheaper than RTN-5. First direct evidence on this benchmark that
   landscape information (here the layerwise proxy H = 2XXᵀ) buys compression.
2. At 2 bits GPTQ still degrades (+72 mnats) but stays a working predictor where RTN is
   halfway to uniform (+383). act_order recovers a further ~19% at 2 bits.
3. Symmetric grids remain strictly worse, catastrophically at 2 bits (3.26): same
   non-zero-centred-weights story as RTN.
4. Working-predictor frontier now 70 272 bytes = 20× the exact-predictor floor, at
   +72 mnats. Next candidates for the ×HMM column: STE-QAT at 2–3 bits, and the
   experimental global-Hessian variant vs this layerwise-proxy reference (survey §3.3).

**Next.** HAWQ-V2 (true-loss Hessian trace → mixed precision) using GPTQ/RTN as base, then
STE-QAT.

---

## 2026-08-31 — HAWQ-V2 mixed precision, d64

**Method.** HAWQ-V2 (`compress/hawq.py`): Hutchinson block traces of the **true**
cross-entropy Hessian (exact HVPs via double backward, 64 Rademacher probes, 128 calib
seqs, seed 0), sensitivity Ω_i(b) = tr(H_i)/n_i · ‖Q_b(W_i)−W_i‖², exact knapsack (DP,
8-byte resolution) over bits ∈ {2,4,8} per tensor, then RTN or GPTQ applied with the
resulting bit map (`bits_map` plumbing added to base/RTN/GPTQ; per-matrix algorithms
unchanged). Logged deviations: no QAT fine-tune after allocation (that is the separate
STE-QAT experiment); negative trace estimates clamped to 0. Variant with the wider menu
{2,3,4,5,6,8} logged as a named deviation. CSVs: `results/compression/L4_d64_hawq*.csv`.

**Command.**
```
uv run python src/compress/run_hawq.py models/cylinder_graph/20260828_142001_L4_d64_H4_full_noLN \
    --budgets 121216 95744 70272 --bases rtn gptq --out results/compression/L4_d64_hawq.csv
```

**Hessian traces** (loss landscape of the trained model): unembed.W_U dominates (140);
then blocks.3.mlp W_out (10.7) / W_in (5.3), pos_embed (3.2), W_E (2.7), early W_O/W_V and
MLPs (0.5–2.2). **W_Q and W_K traces are ≈ 0 in every layer** (|tr| ≤ 0.05, 2/27 slightly
negative): the loss is locally flat in every query/key direction — where the model's
attention pattern sits appears to be almost irrelevant at this minimum. The knapsack
therefore sends W_Q/W_K to 2 bits first, protects W_U/late-MLP last.

**Results** (per-channel asymmetric; Δ mnats vs fp32 2.4761; uniform rows from earlier
entries at the same byte budget):

| bytes | ×HMM | uniform RTN | HAWQ(RTN) | uniform GPTQ | HAWQ(GPTQ) | HAWQ(GPTQ) wide menu |
|---|---|---|---|---|---|---|
| 121 216 | 35 | +9.9 (4b) | +2.8 | +2.1 (4b) | +0.8 | — |
| 95 744 | 28 | +65.4 (3b) | +19.9 | +8.8 (3b) | +6.1 | +3.1 |
| 82 944 | 24 | — | — | — | — | +11.5 |
| 76 544 | 22 | — | — | — | — | +18.1 |
| 70 272 | 20 | +382.8 (2b) | =uniform | +71.6 (2b) | =uniform | =uniform |

(at 70 272 bytes the all-2-bit assignment is the only feasible point, so HAWQ ≡ uniform by
construction; headroom for allocation only exists above the smallest-menu floor.)

**Findings.**
1. Mixed precision from true-Hessian traces beats uniform at every budget with headroom:
   at 3-bit parity, HAWQ(GPTQ, wide menu) costs +3.1 mnats vs +8.8 uniform GPTQ and +65.4
   uniform RTN; at 4-bit parity +0.8 vs +2.1. The two landscape signals compose: global
   traces decide *where* the bits go, layerwise XXᵀ decides *how* to round.
2. New Pareto frontier (this benchmark so far): +3.1 mnats @ 95 744 B, +11.5 @ 82 944 B,
   +18.1 @ 76 544 B, +71.6 @ 70 272 B — the region between 20× and 28× the exact-predictor
   floor is now populated.
3. Physics-flavoured observation: stiffness is concentrated in the read-out (W_U, final
   MLP) and the flat directions are the entire QK circuit. Consistent with the belief-
   geometry picture: many attention patterns can implement the same belief update, but the
   belief → next-token map is unique. Suggests a targeted experiment: how few bits can the
   QK circuit take alone (attention-pattern robustness)?
4. Caveat: traces are curvature at the minimum; Ω assumes independent quadratic responses
   (no cross-tensor Hessian terms). The all-2-bit point (+71.6) is far outside the
   quadratic regime, which is why allocation can't help there — that regime belongs to QAT.

**Next.** STE-QAT (fine-tune at 2–3 bits, uniform and on top of the HAWQ allocation), which
the paper itself prescribes after allocation. Then Fisher-weighted SVD (parameter-count axis).

---

## 2026-08-31 — STE-QAT, d64

**Method.** STE-QAT (`compress/qat.py`): straight-through estimator via torch
parametrizations (forward sees w + (fake_quant(w) − w).detach(); grid recomputed from the
current weights each step under no_grad — the fixed-grid variant, distinct from LSQ).
Fine-tune from the fp32 checkpoint on the full train split (618 749 seqs), 1 epoch, Adam
lr 1e-4, batch 128, seed 0, no scheduler/weight-decay; then weights baked to the lattice.
Supports `bits_map`, so HAWQ-V2 allocations were fine-tuned too — completing the paper's
allocation → QAT pipeline (removes the `no_finetune` deviation for those rows). Runs:
uniform 4/3/2-bit + HAWQ wide-menu allocations (RTN(8) error metric, choices {2,3,4,5,6,8})
at 95 744 and 82 944 B. CSV: `results/compression/L4_d64_qat.csv`.
Design note: the QAT loop is self-contained rather than refactoring train.py's inner loop
(wandb/scheduler/checkpoints unwanted; train.py is in active use).

**Results** (Δ mnats vs fp32 2.4761; best PTQ = best post-training row from earlier entries
at the same bytes):

| bytes | ×HMM | best PTQ | STE-QAT | note |
|---|---|---|---|---|
| 121 216 | 35 | +0.8 HAWQ(GPTQ) | +3.8 uniform | QAT *loses* at 4 bits |
| 95 744 | 28 | +3.1 HAWQ(GPTQ, wide) | +9.8 uniform / **+3.4 HAWQ+QAT** | tie with PTQ frontier |
| 82 944 | 24 | +11.5 HAWQ(GPTQ, wide) | **+5.5 HAWQ+QAT** | new frontier point |
| 70 272 | 20 | +58.0 GPTQ act_order | **+41.4 uniform** | new frontier point |

**Findings.**
1. QAT helps exactly where the quadratic approximation fails: at 2 bits it takes the
   frontier (+41 vs +58), at 82 944 B it nearly halves the PTQ cost (+5.5 vs +11.5); at
   3-bit parity it ties, and at 4 bits one-epoch QAT is *worse* than Hessian-aware PTQ
   (+3.8 vs +0.8) — retraining adds optimization noise where OBS compensation is already
   near-lossless.
2. Allocation and retraining compose: HAWQ bit map + QAT beats uniform QAT at every shared
   budget (95 744: +3.4 vs +9.8).
3. Frozen defaults (1 epoch, lr 1e-4) are deliberately un-tuned; a longer schedule would
   likely improve the 2-bit point further — any such tuning must be logged as named args.
4. 200-step smoke run at 2 bits already beat all PTQ (2.5273), i.e. most of the QAT gain
   arrives almost immediately.

**Frontier after this entry** (best known, Δ mnats @ bytes): +0.8 @ 121 216 (HAWQ·GPTQ),
+3.1 @ 95 744 (HAWQ·GPTQ wide), +5.5 @ 82 944 (HAWQ+QAT), +41.4 @ 70 272 (uniform 2-bit QAT).

**Next.** Fisher-weighted SVD (parameter-count axis — the only route toward the 3 456-byte
floor); AWQ and structured pruning as controls.

---

## 2026-08-31 — Fisher-weighted SVD (final)

Sweep complete (20/20 configs). CSV: `results/compression/L4_d64_fwsvd.csv`.

**Method.** FWSVD (`compress/fwsvd.py`), Hsu et al. ICLR 2022: empirical Fisher
Î_ij = mean over batches of (∂L/∂W_ij)² (4 096 train seqs, batch 32, seed 0);
row-aggregated D = diag(√Σⱼ Îᵢⱼ) over output rows; truncated SVD of D·W; factors
A = D⁻¹UᵣSᵣ, B = Vᵣᵀ. Plain truncated SVD run as the paper's own baseline ("SVD").
Factorized: W_Q/K/V/O (flattened heads), W_in, W_out, W_U; W_E/W_pos dense (paper does not
factorize embeddings). Uniform rank ratio ρ, r = max(1, round(ρ·min(d_out, d_in))).
Fine-tuning (paper's pipeline): factors A, B trained directly via a low-rank
parametrization, 1 epoch, Adam lr 1e-4, batch 128 — same loop settings as STE-QAT;
ft=0 rows are the paper's "w/o fine-tune" variant. **First parameter-count method** —
params = Σ r(d_out+d_in) over factorized matrices + dense rest; bytes = 4·params (fp32,
no quantization yet).

**Results** (loss in nats; fp32 baseline 2.4761 @ 206 128 params):

| ρ | params | bytes | FWSVD | SVD | FWSVD+FT | SVD+FT |
|---|---|---|---|---|---|---|
| 0.75 | 231 664 | 926 656 | 2.4800 | 2.4819 | — | — |
| 0.5 | 156 592 | 626 368 | 2.5086 | 2.5169 | 2.4778 | 2.4780 |
| 0.375 | 119 056 | 476 224 | 2.5566 | 2.5507 | — | — |
| 0.25 | 81 520 | 326 080 | 2.7171 | 2.7849 | 2.4851 | 2.4857 |
| 0.125 | 43 984 | 175 936 | 3.2196 | 3.5229 | 2.5156 | 2.5185 |
| 0.0625 | 25 216 | 100 864 | 3.7543 | 3.6673 | 2.6472 | 2.6754 |

**Findings.**
1. ρ = 0.75 *increases* params (231k > 206k): rank 48 of a 64×64 matrix costs 48·128 >
   64² — low-rank only pays below ρ = 0.5 for square matrices. Kept as a sanity row.
2. Fine-tuning dominates the outcome: without it the factorization collapses quickly
   (ρ = 0.25 at +241 mnats); with 1 epoch, ρ = 0.25 → +9.0 mnats at **2.5× fewer
   parameters** (81 520), and ρ = 0.125 → +39.5 mnats at 4.7× fewer (43 984). These are
   the first points on the parameter axis below the trained-from-scratch d64 count.
3. Fisher weighting helps clearly without fine-tuning at mid ranks (ρ = 0.25: 2.717 vs
   2.785; ρ = 0.125: 3.22 vs 3.52) but the advantage nearly vanishes after fine-tuning
   (≤ 0.7 mnats) — consistent with the paper, where FWSVD's edge is largest pre-FT. At
   the extremes (ρ = 0.75, 0.0625) the weighting is within noise or slightly worse.
4. Landscape reading: Fisher weighting picks a better *starting point* in the same
   low-rank manifold, but 1 epoch of SGD from either starting point finds comparable
   minima — the landscape information matters most when you cannot retrain (same pattern
   as GPTQ-vs-QAT at high bits). Exception: at ρ = 0.0625 the Fisher edge *survives*
   fine-tuning (2.6472 vs 2.6754, 28 mnats) — at extreme compression the two starting
   points land in different basins and the Fisher one is better.
5. Parameter-axis frontier established (fine-tuned FWSVD): 2.4778 @ 156 592 p,
   2.4851 @ 81 520 p, 2.5156 @ 43 984 p, 2.6472 @ 25 216 p. At 43 984 params the
   factorized d64 model (2.5156) is close to the scratch-trained d32 model's ~50k params
   at its best val loss (~2.53 per the sweep notes) — compress-then-finetune ≈ train-small
   directly here, unlike the LLM regime where compression usually wins.
6. Bytes are fp32 (no quantization of factors yet). Composing FWSVD ρ = 0.125 with 4-bit
   GPTQ on the factors would give ~24 kB — below the 2-bit wall of the fixed-architecture
   track — the obvious next composition experiment.

**Next.** Composition (low-rank + quantized factors); or distillation into smaller
architectures (the shape-changing route toward the 864-dof floor).

---

## 2026-08-31 — Composition: FWSVD factors + quantization (experimental)

**Method.** `compress/experimental/lowrank_quant.py` — a *composition* of frozen reference
methods, labeled FWSVD+RTN / FWSVD+QAT (no reference class modified; FWSVD gained factor
caching only). Pipeline per rank ratio ρ: FWSVD(ρ, Fisher-weighted, 1-epoch FT) → cache
factors A (d_out×r), B (r×d_in) → quantize factors at b bits: PTQ = reference RTN grid
(per-output-channel asym) on A and B; QAT = STE fake-quant of both factors inside a further
1-epoch fine-tune (same loop settings as STE-QAT), then bake W = Q(A)·Q(B). W_E/W_pos stay
dense, RTN at the same width; biases fp32. Bytes = factors·b/8 + scales (4 B/channel) +
embeddings·b/8 + biases·4. CSV: `results/compression/L4_d64_lowrank_quant.csv`.

**Results** (loss nats; base = fine-tuned FWSVD at fp32 from the previous entry):

| ρ | params | base fp32 | bits | bytes | PTQ | QAT |
|---|---|---|---|---|---|---|
| 0.25 | 81 520 | 2.4851 | 8 | 100 144 | 2.4902 | — |
| | | | 4 | 60 560 | 2.5236 | **2.4949** |
| | | | 3 | 50 664 | 2.6401 | **2.5165** |
| | | | 2 | 40 768 | 3.3939 | **2.6099** |
| 0.125 | 43 984 | 2.5156 | 8 | 61 816 | 2.5810 | — |
| | | | 4 | 41 000 | 2.6087 | **2.5325** |
| | | | 3 | 35 796 | 2.7559 | **2.5712** |
| | | | 2 | 30 592 | 3.1546 | **2.7264** |
| 0.0625 | 25 216 | 2.6472 | 8 | 42 652 | 2.8945 | — |
| | | | 4 | 31 220 | 2.8727 | **2.6703** |
| | | | 3 | 28 362 | 3.1431 | **2.7396** |
| | | | 2 | 25 504 | 3.3247 | **2.9018** |

**Findings.**
1. **The 70 kB wall is demolished.** Dense 2-bit QAT (the previous frontier end) was
   2.5175 @ 70 272 B; FWSVD ρ=0.25 + 4-bit QAT reaches 2.4949 @ 60 560 B — better loss at
   fewer bytes — and the frontier now extends to 25 504 B (7.4× the HMM floor) at 2.9018.
2. New frontier (all QAT): 2.4949 @ 60 560 → 2.5165 @ 50 664 → 2.6099 @ 40 768 →
   2.7264 @ 30 592 → 2.7396 @ 28 362 → 2.9018 @ 25 504.
3. Factors are much more quantization-sensitive than dense weights (PTQ at 8-bit already
   costs 5–250 mnats vs ~0 dense): the product Q(A)·Q(B) compounds errors, and low-rank
   factors have no redundancy to absorb rounding. QAT is not optional in this composition.
4. Oddity, unresolved: at ρ=0.0625, 8-bit PTQ (2.8945) is *worse* than 4-bit PTQ (2.8727).
   Differences of this size may be grid-interaction luck; flagged, not chased.
5. Floor structure at the bottom: at 25 504 B, fp32 biases (~9.4 kB) + per-channel scales
   are ~40% of the budget — bit-shaving is exhausted; going lower needs architecture
   change (distillation).

---

## 2026-08-31 — Reference set completed: AWQ, LSQ, structured pruning, d64

**Methods.** (CSV: `results/compression/L4_d64_refset.csv`)
* AWQ (`compress/awq.py`), Lin et al. 2024: per-input-channel scales s = (mean|x|)^α,
  α ∈ [0,1] grid of 20 per layer on recon. error, then reference RTN. **No-LN adaptation
  (documented):** scales only where a fold target exists — W_O (folded into W_V/b_V) and
  W_out (folded through ReLU into W_in/b_in); all else plain RTN, as the paper treats
  unscaled layers. 128 calib seqs.
* LSQ (`compress/qat.py`), Esser et al. 2020: symmetric weight quantizer with **learned**
  per-output-channel step s (init 2·mean|w|/√Qp, grad scale 1/√(N·Qp)), STE round/clamp;
  1-epoch fine-tune, same loop settings as STE-QAT.
* NeuronPrune (`compress/prune.py`), structured MLP-neuron pruning, uniform per layer,
  zero-out ≡ removal, params counted as removed (2·d_model+1 each):
  magnitude s_j = √(‖W_in[:,j]‖² + b_in[j]² + ‖W_out[j,:]‖²);
  wanda s_j = ‖h_j‖₂·‖W_out[j,:]‖₁ (Sun et al. 2023's |W|·‖X‖ per W_out row, 128 calib
  seqs); ± 1-epoch fine-tune with pruned neurons masked at zero.

**Results** (loss nats; fp32 2.4761):

| method | setting | params | bytes | loss |
|---|---|---|---|---|
| AWQ | 8 / 4 / 3 / 2 bit | 206 128 | 223 104 / 121 216 / 95 744 / 70 272 | 2.4762 / 2.4846 / 2.5440 / 2.8460 |
| LSQ | 4 / 3 / 2 bit | 206 128 | 121 216 / 95 744 / 70 272 | 2.4791 / 2.4817 / **2.4942** |
| magnitude-prune | f=.25/.5/.75/.875 noFT | 173k/140k/107k/90.5k | ∝ | 2.5009 / 2.6705 / 2.9589 / 3.2032 |
| magnitude-prune | same, +FT | | | 2.4773 / 2.4782 / 2.4822 / 2.4886 |
| wanda-prune | f=.25/.5/.75/.875 noFT | | | 2.4777 / 2.4879 / 2.5292 / 2.6391 |
| wanda-prune | same, +FT | | | 2.4772 / 2.4782 / **2.4810** / 2.4862 |

**Findings.**
1. **LSQ retakes the 70 272 B point: 2.4942 (+18.1 mnats)**, vs STE-QAT's 2.5175 —
   learning the scales is worth 23 mnats at 2 bits and costs nothing extra in bytes.
   Statistical dead heat with the composed FWSVD ρ=0.25+QAT4 (2.4949 @ 60 560 B).
2. **AWQ ≈ RTN here** (3-bit 2.5440 vs 2.5415; 2-bit 2.8460 vs 2.8589): with no LN, only
   2 of 7 matrix types are scalable, and diag-only scaling adds nearly nothing. Clean
   attribution: GPTQ's advantage on this benchmark is the H⁻¹ compensation, not
   activation scaling.
3. **MLPs are massively overparameterized:** dropping 87.5% of neurons (256→32/layer)
   + FT costs +12.5 mnats (2.4886 @ 90 544 params); wanda at f=0.75 gives +4.9 mnats @
   107 056. Pruning + FT beats FWSVD + FT in the 90k–175k range; FWSVD wins below
   (81 520 @ 2.4851 dominates wanda's 90 544 @ 2.4862).
4. Same PTQ-vs-retrain pattern as everywhere: wanda ≫ magnitude before FT (f=0.875:
   2.64 vs 3.20), nearly equal after — data/landscape info picks the better starting
   point; retraining converges to similar minima.

**Reference set now complete**: RTN, GPTQ(+act_order), AWQ, HAWQ-V2, STE-QAT, LSQ,
FWSVD/SVD, magnitude/wanda pruning — all frozen, all benchmarked on d64.

---

## 2026-08-31 — E1: structured finite-radius profiles (geometry program, Tier 1)

Program: `notes/compression/singular_geometry_evaluation.md` §4. Code:
`compress/experimental/{teacher,geo_probes,run_geo_probes,analyze_geo_probes}.py`.
Outputs: `results/compression/geometry/e1_profiles.json`, `e1_controls.json`.

**Method.** Each structured coordinate is a scalar gate α on one component, chosen so that
gate(α) is *exactly* the parameter-space line w* + (α−1)d (verified to all printed digits):
qk (attention scores of head h × α ≡ (W_Q[h], b_Q[h]) scaled), W_Q / W_K alone, head
(hook_z × α ≡ W_O[h]), attn / mlp sublayer outputs (≡ (W_O,b_O) / (W_out,b_out)), and MLP
neuron groups of 32 ranked by activation variance (≡ W_out rows). α ∈ {0, 0.1, …, 2}.
Loss = exact-teacher KL(P_HMM‖P_model) (E0 protocol; label noise removed) on 4 096 train
seqs; controls on the test split. Quadratic prediction ΔL_quad = (α−1) g·d + ½(α−1)² dᵀHd
from exact gradient/HVP of the same KL. Power law c|α−1|^p fitted on α ∈ [0.5, 0.9].
d64 model; KL(w*) = 3.82 mnats on calib, 3.96 on test (= gap to Bayes).

**Results** (ΔKL in mnats; "quad" = quadratic prediction at α = 0):

| gate | ΔKL(α=0) | quad | ΔKL(α=2) | p (R²) | reading |
|---|---|---|---|---|---|
| qk.L0.h0 / h1 / h2 / h3 | 12.3 / 3.2 / 1.4 / 1.1 | 0.65 / 0.98 / 0.58 / 0.65 | 0.09 / 0.30 / 0.25 / 0.40 | 3.5 / 3.5 / 2.2 / 3.6 (1.00) | **super-quadratic** on the shrink side, flat on the sharpen side |
| qk.L1.h0–h3 | 1.8 / 4.2 / 0.2 / 2.0 | 0.95 / 2.35 / 0.14 / 1.10 | 0.75 / 1.0 / 0.09 / 0.52 | 3.7 / 2.8 / 3.0 / 2.7 (1.00) | same, weaker |
| qk.L2.h0–h3 | 0.6 / 0.9 / 0.7 / 0.4 | ≈ measured | ≤ 0.4 | — | **flat** |
| qk.L3.h0–h3 | 0.7 / 1.1 / 0.8 / 1.8 | 0.3–0.6 | ≤ 0.2 | — | flat |
| head (W_O) all 16 | 0.3–16.6 | within ~20 % except L0.h0 (16.6 vs 8.8) | — | 2.0–3.0 | **quadratic** |
| ngrp (32 neurons) | 18/32 groups < 1 mnat; top group per layer 7 / 6 / 31 / 212 | within 10–25 % for all but the top groups | — | ≈ 2 | quadratic; low-variance groups free |
| attn.L0 / L1 / L2 / L3 (whole sublayer) | 86.6 / 36.5 / 5.5 / 12.1 | 13.0 / 10.2 / 6.0 / 11.6 | 13.5 / 6.8 / 10.7 / 11.6 | 2.9 / 2.8 / 2.6 / 2.0 | L0/L1 far beyond quadratic radius; **L2/L3 quadratic and cheap** |
| mlp.L0 / L1 / L2 / L3 | 59 / 19 / 56 / 453 | 14.6 / 6.8 / 19.4 / 192 | 12.3 / 5.9 / 15.7 / 141 | 2.1–3.0 | quadratic near α=1, 3× under-predicted at α=0 |

W_Q-only and W_K-only profiles coincide with the qk profile to ≤ 0.1 mnat everywhere.

**Controls (test split, Δ vs 2.4761):**

| control | Δ nll (mnats) |
|---|---|
| uniform causal attention, all layers (qk α=0 ∀ heads) | **+189.8** |
| W_Q = W_K = 0, all layers | +189.8 (identical) |
| uniform attention, layer 0 / 1 / 2 / 3 only | +65.7 / +30.5 / **+3.1** / +8.0 |
| context shuffled (all 16 tokens), last position | +2 966 (2.4670 → 5.43) |
| context shuffled except most recent token, last position | +559 |

**Findings.**
1. **The Hessian-flat QK circuit is not free.** Uniform attention costs +190 mnats; the
   model uses token order heavily (shuffling costs 0.6–3 nats). The ≈0 traces of the HAWQ
   entry were true *locally* only: layer-0 head 0 costs 12.3 mnats to remove where the
   quadratic model predicts 0.65 (19× under). This is the docx's "flat ≠ removable"
   warning realised on this model, and a case where one-shot curvature ranking would make
   the wrong *structural* decision (it would delete layer-0 QK first).
2. **Shape of the QK directions.** On the shrink side (α < 1) the KL grows as |α−1|^p with
   p ≈ 2.7–3.8 and R² ≈ 1.00 over [0.5, 0.9] in layers 0–1 — the first direct sighting of
   the docx's higher-order (u⁴-type) directions. On the sharpen side (α > 1) they are flat
   (softmax saturation: sharpening an already-peaked pattern changes nothing). So the
   local geometry is a one-sided plateau, not a symmetric quartic well.
3. **It does not matter for quantization, only for deletion.** At the radii a 2–3-bit grid
   reaches (|α−1| ≲ 0.3) the QK cost is ≤ 0.5 mnat per head — consistent with E4's
   measured 0.02 mnat for 2-bit W_Q. The higher-order structure bites only for
   finite-radius structural edits, which is exactly the regime the docx's compiler
   targets and HAWQ-style allocation does not.
4. **Everything else is quadratic.** Head outputs (W_O), neuron groups, layer-3 attention
   and all mlp gates near α = 1 are predicted by g·d + ½dᵀHd to within ~20 %; the sublayer
   deletions (α=0) exceed the quadratic radius as expected for such large perturbations
   but show no power-law anomaly (p ≈ 2–3). 18 of 32 neuron groups cost < 1 mnat each to
   remove (one at a time); the top-variance group per layer carries most of the MLP
   (layer 3: 212 of 453 mnats).
5. **Compile candidates from the profiles alone:** layer-2 attention (uniform attention
   +3.1 mnats on test; whole sublayer gate +5.5 on calib) and the low-variance 3/4 of
   every MLP. Layer depth ordering of sensitivity: attention matters early (L0 ≫ L1 ≫ L3 >
   L2), MLPs matter late (L3 ≫ L0 ≈ L2 > L1).
6. **E0 note.** All geometry measurements use the exact-teacher KL; the dedicated one-hot
   vs soft-target variance comparison has not been run separately yet.

**Next.** E2 (pairwise gate grids, running), E3 (LLC vs soft rank, running), E4
(measured-sensitivity allocation, running).

---

## 2026-08-31 — E4: measured-sensitivity mixed precision, extended menu (geometry program, Tier 2)

Code: `compress/experimental/{geo_alloc,run_geo_alloc}.py` (ExtGPTQ = reference GPTQ
per-matrix routine + two extra rounding rules; ExtSTEQAT = reference STE loop + extended
widths + optional exact-teacher soft targets; reference classes untouched). Outputs:
`results/compression/geometry/e4_sensitivity.json`, `e4_alloc.csv`.

**Method.** Menu per tensor {del (=0), 1, T (ternary, absmean {−s,0,s} per channel with OBS
compensation), 2, 3, 4, 5, 6, 8}. Two sensitivity signals on the *same* menu and rounding
backend: **measured** S_i(b) = ΔKL (exact teacher, 4 096 train seqs) when tensor i alone is
GPTQ-quantized at width b, all else fp32 (243 GPTQ runs); **trace** Ω_i(b) = tr(H_i)/n_i ·
‖Q_b(W_i) − W_i‖² (HAWQ-V2's formula, reference `hessian_traces`). Exact knapsack (8-byte
resolution) at budgets {95 744, 82 944, 70 272, 60 000, 50 000, 40 000} B; apply ExtGPTQ
(128 calib seqs, paper defaults); then 200 STE-QAT steps from the rounded model, one-hot
and soft targets (Adam 1e-4, batch 128, seed 0). Bytes: del = 0, T = ⌈n·log₂3/8⌉ + 2 B per
channel, else shared accounting.

**Sensitivity table highlights** (ΔKL mnats, tensor alone):

| tensor | del | 1 | T | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| embed.W_E | 1420 | 1153 | 24.9 | 6.1 | 1.2 | 0.2 |
| unembed.W_U | 1321 | 671 | 173 | 29.5 | 4.5 | 1.0 |
| blocks.3.mlp.W_out / W_in | 432 / 425 | 402 / 66 | 69 / 18 | 2.8 / 2.7 | 0.5 / 0.6 | 0.1 / 0.1 |
| blocks.0.attn.W_Q / W_K | 76 / 69 | 0.9 / 2.0 | 1.5 / 2.7 | 0.02 / 0.06 | 0.00 | 0.00 |
| blocks.2.attn.W_Q / W_K / W_V / W_O | 1.9 / 3.3 / 4.9 / 4.7 | 0.5–2.5 | 0.2–0.7 | ≤ 0.2 | ≤ 0.05 | ≤ 0.01 |
| blocks.{0,1,2}.mlp W_in / W_out | 25–65 | 12–35 | 1.4–6.4 | 0.6–1.3 | 0.1–0.2 | 0.03 |

**Allocation results** (Δ test nll, mnats vs 2.4761; frontier column = best previously logged):

| bytes | signal | PTQ | +QAT200 | +QAT200 soft | previous frontier | additive pred. / joint ΔKL (calib) |
|---|---|---|---|---|---|---|
| 95 744 | measured | +3.1 | +2.8 | **+1.8** | +3.1 HAWQ wide | 2.6 / 2.9 |
| 82 944 | measured | +8.1 | +7.3 | **+5.6** | +5.5 HAWQ+QAT (1 epoch) | 6.4 / 7.8 |
| 70 272 | measured | **+23.4** | +20.0 | **+16.1** | +18.1 LSQ (1 epoch); +58 best PTQ | 16.5 / 23.9 |
| 60 000 | measured | +117 | +113 | +98 | +18.8 FWSVD+QAT4 @ 60 560 | 40.5 / 117 |
| 50 000 | measured | +298 | +224 | +206 | +40 FWSVD+QAT3 @ 50 664 | 93 / 299 |
| 40 000 | measured | +420 | +350 | +345 | +134 FWSVD+QAT2 @ 40 768 | 185 / 420 |
| 95 744 | trace | +123 | +83 | +66 | | 164 / 128 |
| 82 944 | trace | +158 | +141 | +108 | | 173 / 161 |
| 70 272 | trace | +240 | — | — | | 245 / 243 |
| 40 000 | trace | +772 | +433 | +418 | | 483 / 774 |

Measured allocations use ternary from 70 272 B down (5 tensors) and start deleting at
60 000 B (2 tensors) → 40 000 B (11 tensors). Trace allocations delete 4 tensors already at
95 744 B — always QK matrices (tr H ≈ 0 ⇒ Ω(del) ≈ 0).

**Findings.**
1. **New bytes-axis frontier points**: +1.8 @ 95 744 B and +16.1 @ 70 272 B (measured
   allocation + 200 soft-target QAT steps); the 70 272 B PTQ point improves from +58 to
   +23.4 with no retraining at all, purely by allocating with measured sensitivities over
   a menu that includes ternary.
2. **Curvature is fine for bit widths, wrong for deletion.** Trace-based allocation on the
   extended menu fails at every budget (+123 mnats at 95 744 B) because it deletes QK
   tensors whose traces are ≈ 0 but whose removal costs 30–76 mnats (E1). This is the
   docx's "flat ≠ removable" caveat producing a real allocation failure, and it means any
   compiler that includes structural edits must use finite-radius (measured) costs.
3. **Additivity holds to 70 kB and breaks at 60 kB.** Σᵢ Sᵢ(bᵢ) predicts the joint ΔKL to
   within 30 % down to 70 272 B (16.5 vs 23.9) and under-predicts 3× at 60 000 B (40 vs
   117) — exactly where deletions and 1-bit entries enter. Cross-terms matter only in the
   extreme regime.
4. **Below 70 kB bit-shaving loses to the parameter axis**: FWSVD ρ=0.25 + QAT-4 (+18.8 @
   60 560 B) beats the best allocation here by 80 mnats at the same bytes. The non-additive
   regime is therefore also the regime where the fixed architecture is the wrong object.
5. **Exact-teacher targets help recovery**: soft beats one-hot by 1–4 mnats at every budget
   in 200 steps (E0 effect on fine-tuning; the separate variance comparison remains to do).

**Next.** Log E2/E3 when complete; E7 (global-Hessian GPTQ), E8–E10 queued; the
relaxation program (`relaxation_program.md`) follows.

---

## 2026-08-31 — E10: gauge canonicalization before quantization (geometry program, Tier 4)

Code: `compress/experimental/{geo_gauge,run_geo_gauge}.py`. CSV: `results/compression/geometry/e10_gauge.csv`.

**Method.** Exact reparameterizations (fp32 logits unchanged to < 1e-4, asserted; measured
Δ = 0.0 µnat for every mode), then reference RTN / GPTQ at 4 and 3 bits per-channel asym:
`relu_equalize` (Nagel et al. 2019 cross-layer equalization through the ReLU scaling
gauge), `qk_balance` / `vo_balance` (diagonal GL(16) per head, column-range balancing),
`qk_rotate` / `vo_rotate` (Cayley-parametrized O(16) per head minimizing the per-channel
range proxy Σ n_rows·(max−min)², Adam 200 steps), `all`.

**Results** (Δ vs no-gauge, mnats; negative = better):

| mode | RTN-4 | RTN-3 | GPTQ-4 | GPTQ-3 |
|---|---|---|---|---|
| relu_equalize | −0.67 | +3.79 | −0.20 | +0.07 |
| qk_balance | 0.00 | 0.00 | 0.00 | −0.03 |
| vo_balance | +0.02 | +0.06 | +0.17 | +0.81 |
| qk_rotate | −0.02 | −0.44 | −0.12 | +0.29 |
| vo_rotate | −0.16 | +0.91 | −0.29 | −0.43 |
| all | −1.11 | +3.05 | −0.53 | +0.58 |

**Findings.** Gauge choice is worth ≤ 1 mnat at 4 bits and can *hurt* at 3 bits (ReLU
equalization +3.8 on RTN-3: balancing ranges moves outliers into W_in, whose per-channel
grid was previously protecting them). qk_balance is an exact no-op under per-channel
grids, as predicted. Docx §10.1 ("quotient-first canonicalization") is real but
negligible here — the per-channel scale already absorbs the scale gauges, and the
rotation gauges are too small a group (16 × 16 per head) to matter. Not pursued further.

---

## 2026-08-31 — E2: pairwise gate interactions (geometry program, Tier 1)

Code as E1 (`run_geo_probes.py --stage e2`). Output: `results/compression/geometry/e2_pairs.json`.

**Method.** For all 120 head-output pairs (hook_z gates ≡ W_O[h] lines) and all 28 sublayer
pairs (attn_l / mlp_l output gates), ΔKL on the 5 × 5 grid (α₁, α₂) ∈ {0, ¼, ½, ¾, 1}²
(exact-teacher KL, 4 096 train seqs). Non-additivity R(α₁,α₂) = ΔKL(α₁,α₂) − ΔKL(α₁,1) −
ΔKL(1,α₂); quadratic prediction of the cross term (α₁−1)(α₂−1)·dᵢᵀHdⱼ from exact HVPs.
Sign convention: R > 0 ⇒ deleting both costs more than the sum ⇒ the two substitute for
each other (redundant partners); R < 0 ⇒ same circuit (deleting either already breaks it).

**Results** (mnats):

| pair class | median \|R(0,0)\| | 90th pct | max | corr(R, quad cross) |
|---|---|---|---|---|
| head–head (120) | 0.40 | — | 20.8 (head.L0.0 \| head.L1.1) | — |
| all 148 | 0.54 | 15.0 | 269 (mlp.L2 \| mlp.L3) | 0.84 |

Largest sublayer interactions (R(0,0); quadratic prediction; unary costs):
mlp.L2|mlp.L3 **+269** (quad 80; 56 + 453); mlp.L0|mlp.L3 +128 (35); mlp.L1|mlp.L2 +118
(7.7; 19 + 56); mlp.L0|mlp.L1 +118 (9.5); mlp.L0|mlp.L2 +117 (18); attn.L0|attn.L1 +87
(7.6; 87 + 36); attn.L1|mlp.L3 +47; attn.L0|attn.L2 +28. Negative: attn.L2|mlp.L3 −8.
Head level: head.L0.0 has substitutes in head.L1.1 (+21), head.L0.2 (+11), head.L0.1
(+10); median non-additive share of a joint head-pair deletion is 9 %.

**Findings.**
1. **Fine edits are additive; coarse edits are not.** At head / neuron-group granularity
   the joint cost of two deletions is the sum to ~10 % (median |R| 0.4 mnats) — an
   independent-saliency compiler is adequate there. At sublayer granularity the
   interactions are as large as the unary costs themselves (mlp.L1|mlp.L2: 19 + 56 + 118).
2. **The sign is almost always positive: layers substitute for one another.** Every MLP
   pair and the early attention pairs are redundant partners — removing one is tolerable
   because the others compensate, removing two is catastrophic. This is the mechanism
   behind "retraining erases the starting-point differences": the network reroutes across
   layers. It also says the docx's product-term picture (g₁²g₂² branches) is real at the
   layer level and is *redundancy*, not circuitry.
3. **Curvature gets the sign but not the size.** The exact cross-block Hessian term
   correlates with R at 0.84 but under-predicts the large ones 3–15× (mlp.L1|mlp.L2: 7.7 vs
   118) — finite-radius, like everything else at deletion scale.
4. Consistent with E4's additivity break: per-tensor allocations stay additive until the
   knapsack starts deleting whole tensors in different layers (≤ 60 kB), at which point
   the layer-level redundancy terms dominate.

**Next.** R3.3 (cross-susceptibilities with relaxation) tests whether these redundancy
partners persist after retraining; R1 (adiabatic) whether the unary costs are relaxable.

---

## 2026-08-31 — E3: LLC (SGLD) vs Hessian soft rank across training (geometry program, Tier 1)

Code: `compress/experimental/{geo_llc,run_geo_llc}.py`. Output: `results/compression/geometry/e3_llc_d64.json`.

**Method.** Per checkpoint of the d64 run: test NLL/KL; RTN per-channel knee (bit width at
which ΔL crosses 10 / 50 mnats, interpolated); Lanczos spectral density of the exact-teacher
KL Hessian (2 048 calib seqs, m = 50, 2 probes) → tr H, top eigenvalue, eigenvalue counts,
and λ̂_quad = ½Σᵢ nβhᵢ/(nβhᵢ + γ); SGLD LLC (Lau et al.) with nβ = 16n/log n = 7.4×10⁵
(per-token-loss units), γ = 100, ε = 3×10⁻⁷, 1 000 steps, burn-in 400, 2 chains, batch 256,
λ̂ = nβ(⟨L⟩ − L(w*)). Step-size sweep beforehand: stable for ε ≤ 10⁻⁶, divergent ≥ 3×10⁻⁶.

**Results.**

| ckpt | nll | KL | knee₁₀ | knee₅₀ | λ̂_quad | λ̂_SGLD | tr H | h_max | n(h>10⁻²) | n(h>10⁻¹) |
|---|---|---|---|---|---|---|---|---|---|---|
| epoch 1 | 2.5008 | 0.0288 | 3.97 | 3.03 | 59 276 | −10 928 | 247 | 7.4 | 1 328 | 280 |
| epoch 2 | 2.4910 | 0.0186 | 3.93 | 2.95 | 2 465 | −6 617 | 183 | 5.0 | 4 983 | 236 |
| epoch 3 | 2.4866 | 0.0145 | 3.89 | 2.92 | 3 913 | −4 871 | 155 | 3.9 | 804 | 250 |
| epoch 5 | 2.4814 | 0.0092 | 3.92 | 2.96 | 8 727 | −2 144 | 157 | 4.1 | 692 | 211 |
| epoch 7 | 2.4778 | 0.0057 | 3.95 | 3.18 | 4 984 | −260 | 182 | 4.9 | 720 | 219 |
| epoch 10 | 2.4762 | 0.0039 | 3.98 | 3.28 | 5 211 | 968 | 211 | 5.6 | 1 347 | 231 |
| best | 2.4761 | 0.0040 | 4.00 | 3.28 | 8 437 | 951 (±9) | 211 | 5.6 | 1 413 | 231 |

Chain-length dependence at `best` (from the sweep and R2): λ̂_SGLD = 646 / 951 / 2 127 at
600 / 1 000 / 4 000 steps (ε = 3×10⁻⁷); 277 / 1 602 at ε = 10⁻⁷ / 10⁻⁶ (600 steps).

**Findings.**
1. **The SGLD estimator is only valid at the converged checkpoints** — λ̂ is negative at
   every intermediate one (the chain finds lower loss than w*, i.e. w* is not a minimum)
   and crosses zero only at epoch 7–10. The λ–knee correlation test (Urdshals et al.) is
   therefore not available on this run; the knee itself hardly moves (3.9–4.0 bits).
2. **At the minimum, λ̂_SGLD ≈ 10³ ≪ λ̂_quad ≈ 5–8×10³ ≪ d/2 = 103 064**, but λ̂_SGLD is
   still rising with chain length (2 127 at 4 000 steps) and depends on ε by 5×, so the
   gap to λ̂_quad is at most a factor ~3 once equilibration is accounted for, and possibly
   less. Weak evidence of non-quadratic volume; not a clean measurement (R2 chains show
   the flat directions have not equilibrated at these settings).
3. **The robust number is the stiff-direction count**: ~230 directions with h > 0.1 and
   700–1 400 with h > 0.01 at every checkpoint after epoch 2, out of 206 128 parameters.
   The curvature of the trained model is concentrated in a few hundred directions —
   of the order of the 864-dof generator, though no identification is claimed. λ̂_quad
   itself is noisy across checkpoints (2.5k–59k) because the small-eigenvalue counts
   from 2-probe SLQ are poorly resolved; the h > 10⁻¹ and 10⁻² counts are stable.
4. Practical conclusion: as a *diagnostic* the LLC says what the Hessian soft rank says
   (extremely singular model); as a *tool* it did not resolve anything the cheaper
   measurements did not, and its estimator failed on 5 of 7 checkpoints.

---

## 2026-08-31 — R2.1/R2.2: SGLD posterior variance — non-Gaussianity map and allocation (relaxation program)

Code: `compress/experimental/{geo_thermal,run_geo_thermal}.py`. Outputs:
`results/compression/geometry/r2_nongauss.json`, `r2_alloc.csv`.

**Method.** SGLD chains (as E3) with Welford per-weight variance over post-burn-in samples
(4 000 steps, burn-in 1 000, thin 2 → 1 500 samples): (nβ, γ, ε) = (7.4×10⁵, 100, 3×10⁻⁷),
(7.4×10⁵, 10, 3×10⁻⁷), (7.4×10⁴, 100, 3×10⁻⁶), (7.4×10³, 100, 3×10⁻⁵). Per-weight diagonal
Hessian by Hutchinson (256 probes, 1 024 seqs). ρᵢ = σᵢ² / [T/(hᵢ + Tγ)]. Allocation:
Ωᵢ(b) = (T/2)Σⱼ ΔWᵢⱼ(b)²/σᵢⱼ² on the E4 menu, knapsack, ExtGPTQ, +200 soft-target QAT steps.

**Results.**

| chain | ⟨ΔL⟩ (mnat) | λ̂ | ‖w−w*‖ | per-weight var (all tensors) | median ρ | ρ(W_U) | plateau |
|---|---|---|---|---|---|---|---|
| nβ, γ=100 | 2.9 | 2 127 | 15.0 | 1.45×10⁻⁴ ± 3 % | 0.017 | 1.10 | no |
| nβ, γ=10 | 3.0 | 2 194 | 15.4 | 1.49×10⁻⁴ | 0.004 | 1.09 | no |
| nβ/10, γ=100 | 34 | 2 522 | 37.1 | 1.2×10⁻³ | 0.107 | 1.10 | no |
| nβ/100 | 160 | — | 39.7 | — | — | — | **diverged** |

Allocation (Δ test nll, mnats; identical for the three converged chains because the
variance is uniform): 95 744 B **+32.4** (E4 measured: +3.1); 82 944 B +118–124 (+8.1);
70 272 B +365 (+23.4); 60 000 B +652 (+117). QAT-200 does not rescue (+33, +124, +340, +723).

**Findings.**
1. **The chains measure diffusion, not the posterior.** Variance = ε × (effective steps) to
   3 % in every tensor; the SGLD relaxation time in a direction of curvature h is
   ≈ 2/(ε(nβh + γ)) steps, ~10⁵ for the flat directions at these settings. Only W_U
   (h ≈ 5×10⁻², relaxation ~200 steps) equilibrates, and there ρ ≈ 1.1: **Gaussian**.
2. The "non-Gaussianity map" is therefore uninformative except for the one tensor where
   it says the quadratic model is right, and the posterior-variance allocation degenerates
   to minimum-rounding-error allocation, which is worse than every other allocator tried.
3. Combined with E3 (λ̂ still rising at 4 000 steps) and R3 (the h = 0 fine-tune drifts
   ‖δw‖ ≈ 4 in 200 steps at zero cost), the picture is consistent: the flat manifold is
   enormous and the Gibbs measure at any useful temperature is dominated by the
   localization term in those directions, so σ carries no compression information there.
4. **Decision:** R2.3 (finite-temperature GPTQ on sampled covariances) is not built. A
   SWAG-style covariance (SGD-iterate statistics during fine-tuning) would face the same
   equilibration problem in the flat directions; the finite-*radius* measurements (E1/E4)
   remain the practical replacement for the Hessian where it fails.

---

## 2026-08-31 — R1.4: annealed vs direct quantization; soft-target recovery budget (relaxation program)

Code: `compress/experimental/run_geo_annealq.py`. CSV: `results/compression/geometry/r1_annealq.csv`.

**Method.** STE-QAT (ExtSTEQAT: reference loop, exact-teacher soft targets, Adam 1e-4, batch
128) from the fp32 checkpoint. *Direct*: target widths from step 0 for 5N steps.
*Annealed*: width schedule 8 → 6 → 4 → 3 → 2, N steps per stage, **latents kept** (the
bits_map is mutated inside one parametrized run). Targets: uniform 2-bit (70 272 B) and the
E4 measured-sensitivity allocation at 70 272 B (70 239 B; 18 tensors at 2 bits, 5 ternary,
3 at 4, 1 at 3). N ∈ {50, 200}.

A first version that *baked* the lattice between stages (five separate ExtSTEQAT calls)
gave +565 mnats for the same schedule — not a landscape effect: RTN2(RTN3(w*)) is only
2.916 vs RTN2(w*) 2.862, but latents frozen exactly on the 3-bit lattice sit ≥ 1/42 of
the range from every 2-bit cell boundary and cannot flip in 50 steps at lr 1e-4 (2.875
after 50 steps vs 2.573 from fp32 latents). Progressive quantization must keep latents.

**Results** (Δ test nll, mnats vs 2.4761):

| target | steps | direct | annealed (latents kept) | previous best at these bytes |
|---|---|---|---|---|
| uniform 2-bit, 70 272 B | 250 | +37.8 | +52.5 | +41.4 STE-QAT 1 epoch one-hot |
| | 1 000 | +31.6 | +29.3 | +18.1 LSQ 1 epoch |
| measured alloc, 70 239 B | 250 | **+10.4** | +14.9 | +16.1 (E4, 200 soft steps) |
| | 1 000 | **+7.5** | +8.0 | |

**Findings.**
1. **New frontier point: +7.5 mnats @ 70 239 B** (measured-sensitivity mixed precision incl.
   ternary + 1 000 soft-target STE steps ≈ 2 min). This is below the composed
   FWSVD ρ=0.25 + QAT-4 point (+18.8 @ 60 560 B) and makes the 70 kB region
   fixed-architecture again. Bytes frontier now: +1.8 @ 95 744 → +5.6 @ 82 944 → **+7.5 @
   70 239** → +18.8 @ 60 560 (FWSVD+QAT4) → …
2. **The path into the lattice does not matter**: annealing is within ±5 mnats of direct at
   every setting, better once, worse twice. Same conclusion as R1's gate paths — endpoint
   and recovery budget decide, not the route.
3. Soft targets + 1 000 steps beats the one-hot 1-epoch (4 834-step) STE-QAT by 10 mnats at
   uniform 2-bit: the exact teacher is worth more than 5× the steps.
4. The uniform-vs-measured gap (+31.6 vs +7.5 at the same steps) is the allocation's
   contribution after full recovery: mixed precision is not something retraining
   substitutes for — it is the one *static* decision that survives relaxation.

---

## 2026-08-31 — E7: global-Hessian GPTQ (geometry program, Tier 4; ran on hydra)

Code: `compress/experimental/{geo_global_gptq,run_geo_global_gptq}.py`. CSV:
`results/compression/geometry/hydra/e7_global_gptq.csv` (job 12919352, 1 core).

**Method.** Reference GPTQ traversal and rounding; after each of the 27 tensors, one damped
Newton step on the exact-teacher KL (1 024 held-aside train seqs — sees data the GPTQ calib
does not) over the still-fp32 parameters (unquantized tensors + all biases), CG with exact
HVPs (20 iters), λ = damp·mean(diag H) via Hutchinson, halve-then-skip guard (0 skips in
all runs). Quantized tensors stay frozen on their grids. Answers survey §3.3 Q1/Q3: what
does GPTQ's layerwise H = 2XXᵀ proxy lose vs the true global Hessian?

**Results** (Δ test nll, mnats; per-channel asym, act_order off; ~4 min per run on 1 core):

| bits | GPTQ (layerwise) | GlobalGPTQ damp 0.01 | damp 0.1 | best other PTQ at these bytes | best with retraining |
|---|---|---|---|---|---|
| 4 | +2.1 | +1.0 | **+0.7** | +0.8 HAWQ·GPTQ | +0.8 |
| 3 | +8.8 | +4.9 | **+4.1** | +3.1 HAWQ wide | +3.1 |
| 2 | +71.6 (+58.0 act_order) | +34.4 | **+34.3** | +23.4 E4 measured menu | +7.5 R1.4 |

**Findings.**
1. **The layerwise proxy loses about half the recoverable error at every width**: 2.1 → 0.7,
   8.8 → 4.1, 71.6 → 34.3. At 4 bits a *uniform* global-Hessian pass matches mixed-precision
   HAWQ·GPTQ; at 2 bits it is the best uniform-precision PTQ number and beats the 1-epoch
   one-hot STE-QAT (+41.4) with no retraining at all.
2. Insensitive to the Newton damping (0.01 vs 0.1 nearly identical; no guard activations),
   so the gain is robust linear response, not a tuned optimizer effect.
3. Placement on the frontier: still behind the extended-menu measured allocation at equal
   bytes (+23.4 PTQ) and far behind allocation + 1 000 soft steps (+7.5). Global
   compensation and allocation attack different errors and should compose — obvious
   follow-up: GlobalGPTQ under the E4 bits_map.
4. Caveat: the Newton steps optimise a 1 024-seq calibration KL; test KL is reported and
   tracks it (no sign of calibration overfit at these sizes: test ΔKL ≈ calib ΔKL).

---

## 2026-08-31 — R1.1/R1.2: adiabatic annealing and hysteresis (relaxation program; partly on hydra)

Code: `compress/experimental/{geo_anneal,run_geo_anneal}.py`. Outputs:
`results/compression/geometry/r1_adiabatic.json` (local, layers 0–1 + qk/head all layers),
`hydra/hydra_r1main/`, `hydra/hydra_r1qk/`, `hydra/hydra_r1ngrp/` (hydra jobs),
`hydra/r1_hysteresis.json` (r_post job).

**Method.** Hook gates as E1. Per gate: quench (E1), anneal α:1→0 over {50, 200} steps +
100-step hold, and quench-then-recover at equal total steps; fine-tuning = Adam 1e-4,
batch 128, soft targets. Hysteresis (top-16 quench gates): down 200+100, up 200+100,
3 seeds. Linear-response column (OBS block Schur complement by CG on masked HVPs) failed —
H of the retained block is indefinite away from the exact minimum and CG residuals were
0.3–3; the damped rerun on hydra is pending; column not used.

**Results** (ΔKL mnats on calib; 300-step totals):

| gate | quench | anneal(200+100) | quench+recover(300) |
|---|---|---|---|
| qk.L0.0 / .1 / .2 / .3 | 12.3 / 3.2 / 1.4 / 1.1 | 1.8 / 0.6 / −0.2 / −0.1 | 1.6 / 0.5 / −0.2 / −0.2 |
| head.L0.0–3 | 16.6 / 5.5 / 3.7 / 2.0 | 1.9 / 1.2 / 0.0 / 0.1 | 2.0 / 1.1 / 0.0 / 0.1 |
| attn.L0, mlp.L0 | 86.6, 59.0 | 13.3, 7.6 | 14.0, 8.4 |
| attn.L1, mlp.L1 | 36.5, 19.3 | 7.3, 2.0 | 7.3, 2.0 |
| qk.L1.0–3, head.L1.0–3 | 0.3–6.4 | ≤ 0.4 | ≤ 0.4 |

(R3.1's field runs at lr 1e-3 reach attn.L0 ≈ 3.0, mlp.L0 ≈ 2.2 mnats — the residual cost
above is set by the lr-1e-4 recovery budget, so these are upper bounds on the adiabatic
limit.)

**Hysteresis (all 16 top gates, 3 seeds):** none. Every down-and-up loop returns to
−0.2…−0.8 mnats *below* the starting loss (the extra 600 fine-tune steps help more than
the round trip hurts); spread over seeds ≤ 0.05 mnats; displacement at return ≈ 2–2.5.
Examples: mlp.L3 down +5.5 → return −0.3; attn.L0 down +13.3 → return −0.2.

**Findings.**
1. **Path independence is exact within noise**: annealing ≈ quench-then-recover at equal
   compute for all 40 gates. The docx's and the physics picture's adiabatic machinery
   buys nothing here — only the endpoint and the recovery budget matter.
2. **No hysteresis, single basin**: component removal/restoration is fully reversible for
   every high-cost gate; no first-order transitions, hence no basin selection at
   single-component granularity. (FWSVD ρ=0.0625's surviving edge must come from a more
   global, many-component effect, consistent with E2's layer-level cross terms.)
3. **Relaxable fractions R_c ≥ 0.85 everywhere**, ≥ 0.95 for attention components; the
   quench ranking and the relaxed ranking broadly agree on *ordering* (Spearman on the 40
   local gates ≈ 0.9) but the magnitudes compress by ~10×, so any quench-based *budgeting*
   (how many mnats a deletion costs) is wrong by an order of magnitude in the retraining
   regime.

---

## 2026-08-31 — E4 + R1.4 follow-up: measured allocation + 1 000 soft-target QAT steps

CSV: `results/compression/geometry/e4_qat1000/e4_alloc.csv` (same allocations as E4;
recovery = 1 000 STE steps, exact-teacher soft targets, Adam 1e-4, seed 0, ≈ 2 min each).

| bytes | ×HMM | PTQ | +1 000 soft steps | previous frontier |
|---|---|---|---|---|
| 95 744 | 28 | +3.1 | **+0.5** | +1.8 (200 steps) |
| 82 944 | 24 | +8.1 | **+2.4** | +5.5 |
| 70 239 | 20 | +23.4 | **+8.3** (R1.4: +7.5) | +16.1 |
| 59 891 | 17 | +117.3 | **+19.1** | +18.8 @ 60 560 (FWSVD+QAT4) |
| 49 987 | 14 | +296.1 | **+34.8** | +40.5 @ 50 664 (FWSVD+QAT3) |

**Bytes frontier now** (Δ mnats @ bytes): +0.5 @ 95 744 → +2.4 @ 82 944 → +7.5 @ 70 239 →
+19.1 @ 59 891 → +34.8 @ 49 987 → +134 @ 40 768 (FWSVD+QAT2, still standing) → … Single
seed; the ≤ 1 mnat differences (e.g. +7.5 vs +8.3 at 70 kB) are within run-to-run noise.

**Reading.** Three ingredients account for the entire fixed-architecture frontier:
finite-radius measured sensitivities decide *where* the bits go (incl. ternary/delete),
GPTQ decides *how* to round, and a short exact-teacher recovery repairs what rounding
breaks. Every Hessian-only and every SLT-specific quantity tried so far is dominated by
this combination on this benchmark.

---

## 2026-08-31 — E8: residual-stream projection — belief probe vs PCA (geometry program, Tier 4; hydra)

Code: `compress/experimental/{geo_beliefproj,run_geo_beliefproj}.py`. CSVs:
`results/compression/geometry/hydra/e8_beliefproj.csv`, `hydra/hydra_e8b/…` (PCA epochs).

**Method.** Shared projection P ∈ ℝ^{64×d′} of the residual stream; every matrix projected
(W_E P, PᵀW_Q, …), biases likewise; recovery 0 / 200 steps / 1 epoch (one-hot loop). P from:
(a) ridge belief probes at 5 read points (union of ranges, SVD; held-out R² per read point:
0.62 / 0.77 / 0.84 / 0.91 / 0.95 from layer 0 to final), (b) PCA of the residual stream,
(c) random orthonormal control. Sanity: P = I reproduces the model exactly.

**Results** (Δ test nll, mnats; params after projection):

| d′ | params | probe 0 / 200 / 1ep | PCA 0 / 200 / 1ep | rand 0 / 200 |
|---|---|---|---|---|
| 18 | 59 296 | +828 / +356 / +96 | +641 / +186 / **+22** | +1 143 / +415 |
| 24 | 78 448 | +653 / +249 / +48 | +281 / +82 / **+10.6** | +1 090 / +321 |
| 32 | 103 984 | +504 / +185 / +19 | +94 / +25 / **+4.4** | +788 / +204 |

**Findings.**
1. **The belief-probe subspace is the wrong subspace** — PCA beats it at every width and
   budget, by 4–7× before recovery. The residual stream carries much more than the belief
   simplex (positional and token-identity structure that attention needs; layer-0 probe
   R² is only 0.62). The physically-motivated "project onto the belief simplex" compile
   step fails; the *statistically* dominant subspace is the right one.
2. PCA + 1 epoch is competitive with the parameter-axis frontier but does not beat it:
   +10.6 @ 78 448 vs FWSVD +9.0 @ 81 520; +22 @ 59 296 vs tied-block +18.7 @ 56 944 (E9).
   d_model narrowing is *not* the privileged axis the belief picture suggested.
3. Random-subspace recovery is slow (+204 at d′=32 after 200 steps) — the subspace choice
   does matter at fixed short recovery, unlike most quench-level choices.

---

## 2026-08-31 — E9: layer tying (geometry program, Tier 4; hydra)

Code: `compress/experimental/{geo_tie,run_geo_tie}.py`. CSV: `results/compression/geometry/hydra/e9_tie.csv`.

**Method.** Parameters of blocks tied (same nn.Parameter), n_shared ∈ {1, 2} applied over 4
layers, plus 1 shared block applied ×8; init = layer-mean, layers 0..k, or each single
layer (best selected by 200-step recovery, then 1 epoch); recovery 200 steps / 1 epoch
(one-hot). Unique params 56 944 (1 block) / 106 672 (2 blocks) incl. embeddings + unembed.

**Results** (Δ test nll, mnats):

| config | init | 200 steps | 1 epoch |
|---|---|---|---|
| 1 block × 4 | mean / layer0 / best single (L2) | +221 / +221 / +188 | +33.6 / **+18.7** / +41.9 |
| 2 blocks × 4 | mean / layer0+1 | +133 / +89 | +15.6 / **+10.4** |
| 1 block × 8 | mean / layer0 | +200 / **diverged** (nll 18.2) | +25.5 / — |

**Findings.**
1. **One recurrent block applied 4× reaches +18.7 mnats at 56 944 params** — on the
   parameter frontier (beats FWSVD ρ=0.125: +39.5 @ 43 984; close to wanda f=0.875 +12.5 @
   90 544 at far fewer params). The belief-update-as-recurrence picture holds up well.
2. ×8 application of one block is *worse* than ×4 (+25.5 vs +18.7 mean-init) and layer0
   init diverges at ×8 — more recurrence is not better for this unrolled architecture.
3. Init matters at fixed 1-epoch recovery (layer0 > mean > best-single for ×4) — a rare
   case where the starting point survives a full epoch, i.e. tying creates genuinely
   different basins, consistent with tying being a *global* constraint (E2's layer-level
   cross terms) rather than a local edit.
4. Discrete-choice character: unlike single-component deletions (R1.2: no hysteresis,
   fully relaxable), the tying decision is exactly the kind of edit where selection
   information persists — the regime the docx's compiler claims. Here the selection was
   made by enumeration (3 inits × 3 configs), which the model's small size allows.

**Sub-50 kB extension** (`results/compression/geometry/e4_low/e4_alloc.csv`, same recipe):
+41.0 @ 44 908 B (9 tensors deleted), **+68.8 @ 39 953** (11 deleted), **+74.7 @ 34 877**
(12 deleted, 7 ternary — 10× the HMM floor). All previously-frontier composed points fall
(FWSVD+QAT2 was +134 @ 40 768, +176 @ 30 592). Below ~35 kB the allocator is deleting
nearly half the tensors — the "fixed architecture" is being carved into a new one by the
menu, which is where this track hands over to genuine architecture change (E9 tying:
+18.7 @ 56 944 *params*; bytes comparison requires quantizing the tied block — untested).

---

## 2026-08-31 — R3.1/R3.2: field response and the global-field phase diagram (relaxation program)

Code: `compress/experimental/{geo_field,run_geo_field,run_geo_globalfield}.py`. Outputs:
`results/compression/geometry/r3_field.json` (+ `hydra/hydra_r3{main,ngrp,qk}/`),
`r3_globalfield.json`, logs alongside.

**R3.1 (per-component fields; gauge-invariant Ω: ‖W_Q W_Kᵀ‖², ‖W_V W_O‖², Σ‖W_in‖·‖W_out‖,
sublayer output power; 200 steps at lr 1e-3 per h; targets {1, 10, 100} mnats + h = 0 drift
control).** Highlights over the 72 components:
- QK logit scales collapse to 8–20 % of trained norm at ≤ 0.1 mnat everywhere (the
  over-sharpening seen in E1, now quantified with relaxation: ≥ 80 % of the attention
  logit norm is free to remove).
- Whole layer-0 sublayers: attn m → 0.02 at +3.0 mnats, mlp m → 0.01 at +2.2 (test) —
  97 % of the E1 quench cost (86.6 / 59.0) relaxes in 200 steps at lr 1e-3.
- Every individual head is drivable to m ≤ 0.01 at ≤ 1.2 mnats; low-variance neuron
  groups at ≈ 0.
- The h = 0 control drifts ‖δw‖ ≈ 4.2 in 200 steps while *improving* KL by 0.1 mnat —
  the flat manifold is macroscopic.

**R3.2 (global group-lasso field, penalty h·Σ_c √(Ω_c/Ω_c(w*)), annealed up over
h ∈ 0.1 → 100 mnats then back down; 500 steps/h at lr 1e-3; at each upward h the
components with m < 0.05 are hard-deleted and recovered 200 soft steps).**

Upward leg — loss vs parameters after hard-delete + 200-step recovery:

| h (mnat) | dead (qk/head/ngrp/attn/mlp) | params left | Δ test nll (mnats) | old frontier at ≈ params |
|---|---|---|---|---|
| 0.1 | 0/0/0/0/0 | 206 128 | −1.1 | — |
| 0.5 | 6/5/3/1/0 | 166 736 | **−0.4** | +1.1 @ 173k (wanda) |
| 1 | 8/7/5/1/0 | 154 336 | **+0.6** | +2.1 @ 140k |
| 2 | 9/9/14/2/0 | 110 912 | **+2.9** | +4.9 @ 107k |
| 5 | 10/10/21/2/0 | 77 872 | **+7.5** | +9.0 @ 81.5k (FWSVD) |
| 10 | 11/11/23/2/0 | 65 472 | **+13.7** | +18.7 @ 57k (E9 tied) |
| 20 | 12/12/27/3/0 | 40 608 | **+33.2** | +39.5 @ 44k (FWSVD) |
| 50 | 15/13/29/3/3 | 19 744 | **+133.8** | +171 @ 25.2k (FWSVD) |
| 100 | 16/15/31/3/3 | 13 536 | +186.5 | — |

**Findings.**
1. **New parameter-axis frontier at every size below 175k**, produced by one field sweep +
   200 recovery steps per point. At 19 744 params (5.7× the 864-dof floor, 23× fewer than
   the trained model) the loss is +134 mnats; at 40 608 params +33.
2. **The surviving architecture at 20k params is essentially one attention sublayer + one
   MLP** (3 of 4 attn and 3 of 4 mlp sublayers dead, 31 of 32 neuron groups dead) plus
   embeddings/unembedding — the field discovers a 1-layer effective model, consistent
   with the belief update being a single recurrent map (and with E9's tied block).
3. **Strong hysteresis at the collective level** — opposite to R1.2's single-component
   reversibility. On the downward leg nothing regrows (31 neuron groups and all 3 MLPs
   stay dead to h = 0.1; field-trained KL +14.4 mnats at the h where the up-leg was
   −0.55): once the network reroutes and a component's weights decay, the gradient that
   would revive it is gone (dead-unit irreversibility). Deletion order is path-dependent
   *globally* even though each single deletion is reversible — the basin-selection
   phenomenon lives at the collective level, exactly where FWSVD ρ=0.0625 and E9's
   init-dependence put it.
4. Deletion order (heads and QK first, then neuron groups, MLP sublayers last; layer-0
   attention before layer-3 MLP) matches the E4/E1 sensitivity ordering — the relaxed and
   quench *orderings* agree even though the magnitudes differ 10×.

**R1 addendum (hydra r_main: all 24 head/attn/mlp gates, damped CG).** With λ = 1e-2·dᵀHd/‖d‖²
the CG residuals improve (0.03–1.2 for most gates) but the linear-response prediction of the
relaxed cost fails outright: Spearman(linear response, measured relaxed cost) = **−0.53**
(it predicts ≈ −1 to 0 mnats for nearly every gate), while Spearman(quench, relaxed) =
**0.944**. Median |prediction − measured| = 1.15 mnats against a median relaxed cost of
0.03. Conclusion: the relaxed cost is set by nonlinear rerouting that the quadratic
model cannot see even in principle here (H of the retained block is indefinite and the
relevant displacements are far outside its radius). Practical rule for this model:
*rank* structural edits by any finite-radius quench measurement, *budget* them only by
actually relaxing. Confirms E1/E4 and completes the R1 decision rules: quench-vs-adiabatic
orderings agree (so E4-style measured saliencies remain valid selectors), linear response
does not.

---

## 2026-08-31 — R3.3: cross-susceptibilities with relaxation (relaxation program; hydra)

Output: `results/compression/geometry/hydra/hydra_r3main/r3_cross.json` (24 source × 72
target components; χ_cc′ = drift-corrected relative change of Ω_c′ after 200 steps with a
field on c at the h where m_c ≈ ½).

**Results.** (Sign convention from the code: χ_cc′ = relative change of Ω_c′ after squeezing c
with a positive field, drift-corrected; χ > 0 ⇒ the partner *grows*.) Median |χ| = 0.019
(90th pct 0.17): most pairs are independent after relaxation, confirming E2's head-level
additivity survives retraining. The large entries are all sublayer-scale and split by sign:
- **Substitutes (χ > 0, partner grows when c is squeezed).** Squeeze mlp.L2 → attn.L2 grows
  +1.72, mlp.L1 +1.20, attn.L3 +0.79; squeeze mlp.L1 → attn.L2 +0.81; squeeze mlp.L0 →
  heads L1.0/L1.3 +0.34. The layer-2 MLP's function is absorbed by neighbouring attention
  and by MLP-1 — the relaxed counterpart of E2's positive cross-terms.
- **Serial chains (χ < 0, partner shrinks too).** Squeeze attn.L2 → attn.L1 −0.52, mlp.L1
  −0.55 (and its own heads −0.53…−0.58, trivially); squeeze attn.L3 → mlp.L2 −0.66;
  squeeze attn.L1 → head.L1.2 −0.56. Components upstream of a squeezed sublayer lose their
  purpose and decay with it: layer 1 feeds attn.L2, mlp.L2 feeds attn.L3.

**Reading.** The docx's Table-3 relations exist and are measurable, but as *thermo­dynamic
response coefficients under retraining*, not as static loss-expansion terms: merge/
substitute candidates = the χ > 0 pairs; delete-together units = the χ < 0 chains. Consistent with R3.2's endpoint (layers 2–3 attention and MLPs die together while
early layers absorb their function).

---

## 2026-08-31 — Synthesis: what the geometry + relaxation program established

All experiments of `singular_geometry_evaluation.md` (E0–E10) and `relaxation_program.md`
(R1–R3; R2.3 dropped with cause) are complete. One-line verdicts against the driving
questions:

| question | verdict | decisive evidence |
|---|---|---|
| Is there beyond-quadratic structure? | **Yes, and it matters for structural edits only** | E1 (QK |α−1|^3.5, Hessian 19× off at deletion radius), E4 (trace allocator deletes QK, +123), E7 (proxy loses half the compensation) |
| Is it the SLT normal-form kind? | **No** — one-sided saturation plateaus, not algebraic zeros; polynomial fits fail on the sharpen side | E1 profiles |
| Does the LLC help? | No — estimator invalid off-minimum, ε- and chain-length-dependent at the minimum; adds nothing beyond the (also noisy) soft rank | E3, R2 |
| Does thermal/posterior volume help? | No, negatively — SGLD chains measure their own diffusion in the flat directions at any usable temperature | R2 |
| Does linear response (global OBS) predict relaxed costs? | **No** (Spearman −0.53) — but it is the best pure-PTQ *compensator* (E7: half the layerwise error recovered) | R1 addendum, E7 |
| Do quench measurements at least rank edits correctly? | **Yes** (Spearman 0.94 vs relaxed) — but magnitudes are ~10× too large as budgets | R1 |
| Is deletion path-dependent? | Per component, no (no hysteresis); collectively, strongly yes (dead components never regrow) | R1.2 vs R3.2 |
| Where does basin selection live? | In global/discrete choices: layer tying init (E9), collective deletion order (R3.2), extreme low-rank (FWSVD ρ=0.0625) | E9, R3.2 |
| What actually builds the frontier? | Measured finite-radius sensitivities (where) + GPTQ (how) + short exact-teacher recovery (repair); fields/global-lasso for the parameter axis | E4+R1.4 (bytes), R3.2 (params) |

**Final frontiers** (Δ mnats vs fp32 2.4761; single seed):
- Bytes (fixed 206k-param architecture, mixed precision incl. ternary/delete):
  +0.5 @ 95 744 → +2.4 @ 82 944 → +7.5 @ 70 239 → +19.1 @ 59 891 → +34.8 @ 49 987 →
  +74.7 @ 34 877 B (10.1× the 3 456-B floor).
- Params (global-field carving + 200-step recovery): −0.4 @ 166.7k → +2.9 @ 110.9k →
  +7.5 @ 77.9k → +33.2 @ 40.6k → +133.8 @ 19.7k (5.7× the 864-dof floor); survivor ≈ a
  1-layer effective transformer, consistent with E9's tied recurrent block (+18.7 @ 56.9k
  with a full epoch).

**For the docx** (`singular_landscape_transformer_compression.docx`): its warnings are
validated (flat ≠ removable; curvature mis-prices both the flattest and stiffest tensors),
its diagnosis of *where* extra information lives (structural, finite-radius, collective) is
right, but every mechanism it proposes for extracting that information (normal forms, LLC,
monomial exponents, quotient canonicalization) is dominated on this testbed by direct
finite-radius measurement plus relaxation — which its own falsification condition (§12)
anticipated. The surviving research direction is the collective/irreversible regime:
hysteretic deletion order, tying-init basins, and the coarse-grained-process picture
(which k-state HMM does a carved model implement — untested here).

**Loose ends** (not blocking): r1ngrp rerun finishing on hydra; E0's dedicated variance
comparison; 1-epoch recovery on the R3.2 points; quantizing the E9 tied block; seeds for
the ≤1-mnat frontier gaps; GlobalGPTQ under the E4 bits_map.

**R1 ngrp completion (hydra rerun, in-place-hook fix):** all 32 neuron-group gates:
relaxed (300-step) cost median −0.52 mnats, max +1.29 (top-variance group of layer 3;
quench 212). Confirms: every 32-neuron group is individually removable at ≈ 0 cost after
relaxation. Program complete; all hydra jobs finished, results synced under
`results/compression/geometry/hydra/`.

---

## 2026-09-01 — R4: finite-temperature correlators of coarse observables (§4.3 of the writeup; hydra)

Code: `compress/experimental/{geo_correlators,run_geo_correlators}.py`. Outputs:
`results/compression/geometry/hydra/r4_corr_{adam,sgd,sgld}.json`, `r4_summary.json`,
`r4_delete.csv`, `r4_tgptq.csv` (job 12921819, 1 core, 35 min).

**Method.** Trajectories from w* — Adam lr 1e-3 (2 000 steps), SGD lr 1e-2 (2 000), SGLD
ε = 3×10⁻⁷, nβ = 7.4×10⁵, γ = 100 (4 000) — recording every 5 steps the 72 gauge-invariant
component observables Ω_c (geo_field), the batch KL, ‖w − w*‖, and the calib KL every 50.
Fluctuations x_c = Ω_c/Ω_c(w*) − 1 after 25 % burn-in: covariance C, correlation, loss
correlator cov(ΔL, x_c), integrated autocorrelation time τ_c. Tests: FDT (∂Ω_c′/∂h_c =
−β C_cc′) against R3.3's χ; C_cc against R3.1's free fraction (1 − m at the strongest
field) and E1's quench cost; deletion order by C_cc (all components / fine components
only) vs quench order vs random, each hard-deleted + 200 soft recovery steps at R3.2's
parameter counts; ThermalGPTQ = reference rounding with ⟨2XXᵀ⟩ over 50 Adam states vs
H(w*) without propagation vs the propagated reference.

**Trajectory facts.** ‖w − w*‖ after 2 000 steps: Adam 14.0 (KL falls to 2.4 mnats), SGD
0.05 (KL 3.7), SGLD 15.1 (KL 7.7). the integrated autocorrelation time saturates the estimator's 60-lag cap for *every*
observable under every dynamics (τ_c ≥ 300 steps; the autocorrelation never crosses zero
within the window): a single slow drift mode along the flat manifold dominates — the
"correlators" are drift statistics, not equilibrium fluctuations. (An earlier version of
this entry quoted τ ≈ 450–520 steps; that number was the cap, i.e. a lower bound.)

**Correlator tests** (Spearman unless stated):

| test | Adam | SGD | SGLD | reading |
|---|---|---|---|---|
| corr(x_c, x_c′) vs χ_cc′ (R3.3, 24×71 pairs) | −0.09 | +0.03 | −0.12 | **FDT fails** |
| sign agreement with −χ, all / \|χ\| > 0.2 (66 pairs) | 0.45 / 0.52 | 0.42 / 0.64 | 0.37 / 0.56 | ≈ chance |
| var(x_c) vs free fraction (R3.1) | 0.53 | 0.62 | **0.80** | thermal variance *does* rank how decorative a component is |
| var(x_c) vs quench cost (E1) | 0.02 | 0.42 | 0.13 | not a cost signal |
| cov(ΔL, x_c) vs field cost (R3.1) | 0.41 | −0.14 | −0.27 | no consistent loss correlator |

Loosest observables (largest var): Adam — attn.L3, attn.L2, mlp.L2, qk.L3.2; SGLD — layer-3
and layer-2 heads/attention. Tightest: layer-0 QK, top neuron groups of layers 0 and 3 —
the same layer ordering as E4/R3 (layer 0 pinned, layers 2–3 loose).

**Deletion by thermal variance** (Δ mnats after 200 recovery steps; params left ≈):

| order | 165k | 108k | 74k | 39k | 16k |
|---|---|---|---|---|---|
| thermal var, fine components | +4.0 | +18.9 | +81 | +205 | +666 |
| thermal var, incl. sublayers | (+8.5 @ 140k) | +48 @ 86k | +216 @ 57k | +406 @ 40k | +1118 @ 7k |
| quench cost ascending (E1) | **+0.3** | **+6.5** | **+19.9** | **+99** | **+248** |
| random | +19.6 | +46.9 | +157 | +230 | +440 (@11k) |
| R3.2 global field (for reference) | −0.4 | +2.9 | +7.5 | +33 | +134 (@19.7k) |

**ThermalGPTQ** (Δ mnats): 4-bit 2.2 / 3-bit 9.7 / 2-bit 113 vs H(w*)-no-propagation 2.0 /
9.6 / 105 and propagated reference 2.1 / 8.8 / 71.6. Averaging activation correlators over
the trajectory does not help and slightly hurts at 2 bits; sequential propagation is worth
34 mnats at 2 bits.

**Findings.**
1. **Short-trajectory correlators of coarse observables do not reproduce the T = 0 response
   (FDT test fails, sign agreement at chance).** The reason is visible in τ_c: one
   relaxation time ≈ the window, so the second moments measure the slow drift along the
   flat manifold, not equilibrium fluctuations. Coarse observables equilibrate faster than
   weights (R2), but not within 2 000–4 000 steps.
2. **The one positive:** the thermal variance of a component's observable ranks how much of
   it is decorative (Spearman 0.80 with the field-measured free fraction under SGLD). This
   is a *scale-redundancy* detector (e.g. the over-sharpened QK logit scales), not a
   deletion or cost detector: as a deletion selector it is worse than the static quench
   ordering at every parameter count, and far worse than the adaptive field sweep.
3. Thermalising the GPTQ Hessian is useless here; what matters for rounding at 2 bits is
   sequential propagation (34 mnats), i.e. conditioning on the already-quantized
   upstream, not on the temperature.
4. Conclusion for §4.3 (before R4b): finite-temperature information would need
   equilibrated trajectories; R4b below tests whether 30 000–60 000 steps suffice.

**R4b — long trajectories** (hydra job 12921984, 2 h 27 min; `hydra/r4b/`): SGD 30 000
steps (‖w−w*‖ 0.32, KL 3.35 mnats), SGLD 60 000 (‖w−w*‖ **39.6**, KL rising to 11.8), Adam
30 000 (‖w−w*‖ 48.9, KL still falling, 0.96), recorded every 10 steps (2 250–4 500 samples).
τ again saturates the cap (≥ 600 steps) in all three. FDT test: Spearman(corr, χ) = +0.05 /
−0.11 / −0.08, sign agreement 0.42 / 0.37 / 0.53 (large-|χ| 0.50 / 0.56 / 0.58) — **fails at
every length tried**. Variance vs free fraction: 0.64 / 0.70 / 0.25 (SGD / SGLD / Adam) —
the scale-redundancy detector is robust under SGD/SGLD, absent under Adam (whose
preconditioning re-weights the observables). Verdict: on this model trajectory sampling
never reaches equilibrium at any affordable length — SGLD is a random walk on a flat
manifold whose extent (‖δw‖ ≳ 40) exceeds anything the localization pins; finite-T
correlators are not an available tool here, and the one usable finite-T quantity (thermal
variance → decorative fraction) is already known from the T = 0 field sweep.
