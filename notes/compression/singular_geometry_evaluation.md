# Landscape information for compression: evaluation, results, and the observables that matter

*Written 2026-08-31, revised 2026-09-01 after the full program ran. Evaluates the ideas in
`singular_landscape_transformer_compression.docx` (draft of 2026-08-17) against the data in
`experiment_log.md`, reports what the geometry program (E0–E10) and the relaxation program
(R1–R3, `relaxation_program.md`) found, and then sets out — the part that survived —
which observables beyond the static Hessian carry compression information and what each
one lets you do.*

Companion files: `compression_survey.md` (method definitions, honesty rules, code layout),
`experiment_log.md` (every number below, with settings), `relaxation_program.md` (plans for
R1–R3), `results/compression/geometry/` (raw outputs; hydra runs under `hydra/`).

---

## 1. Testbed and where the frontier ended up

| | |
|---|---|
| Model | L4 d64 H4, d_head 16, d_mlp 256, ReLU, no LayerNorm, ctx 16, vocab 48; 206 128 params; fp32 test loss 2.4761 |
| Process | `cylinder_graph` 6 × 3 = 18 hidden states, non-unifilar; Bayes-optimal at ctx 16 = 2.4724; hₓ = 2.4397 |
| Floor | exact predictor = 864 dof = 3 456 B fp32 |
| Loss unit | mnats = 10⁻³ nats/token; the fp32 model is 3.7 mnats above Bayes, so 1 mnat is resolvable |

Frontiers at the end of the program (Δ mnats vs fp32; single seed):

| axis | points | recipe |
|---|---|---|
| bytes (206k params fixed) | +0.5 @ 95 744 → +2.4 @ 82 944 → +7.5 @ 70 239 → +19.1 @ 59 891 → +34.8 @ 49 987 → +74.7 @ 34 877 B (10× floor) | measured finite-radius sensitivities → knapsack over {del, 1, ternary, 2…8} → GPTQ rounding → 1 000 exact-teacher STE steps |
| params (fp32) | −0.4 @ 166.7k → +2.9 @ 110.9k → +7.5 @ 77.9k → +33.2 @ 40.6k → +133.8 @ 19.7k (5.7× floor) | one group-lasso field sweep over all 72 components, hard-delete at each h, 200 recovery steps |
| params, discrete | +18.7 @ 56.9k (one block × 4, layer-0 init, 1 epoch); +10.4 @ 106.7k (two blocks) | layer tying, init chosen by enumeration |

Every reference method (RTN, GPTQ, AWQ, HAWQ-V2, STE-QAT, LSQ, FWSVD, magnitude/wanda
pruning) and every previously composed point is below these.

## 2. What the data said about the docx's claims

The docx's thesis: the Hessian at w* misses finite-radius and singular structure; extracting
that structure (normal forms in the Hessian nullspace, LLC, monomial exponents, quotient
canonicalization) should give a better structural compiler. Verdicts:

| claim / mechanism | outcome | evidence |
|---|---|---|
| Flat ≠ removable; curvature mis-prices edits | **Confirmed, strongly.** Layer-0 QK: tr H ≈ 0 yet deletion costs 12–76 mnats (Hessian 19–270× under). W_U: stiffest tensor, yet quantization costs 7–9× *less* than the quadratic prediction. | E1, E4 |
| Higher-order directions exist | **Yes**: ΔKL ∝ |α−1|^p, p ≈ 2.7–3.8, R² = 1.00 on the shrink side of layer-0/1 QK | E1 |
| …and are algebraic (normal-form) singularities | **No**: one-sided saturation plateaus (flat when sharpening, steep when flattening) — no finite Taylor order reproduces them | E1 |
| They matter for compression | **Only for structural edits.** At quantization radii (2–8 bits) the quadratic-isotropic model is right up to a constant factor 0.43 (IQR 0.33–0.52), so it *ranks* bit widths correctly; at deletion radius it fails and in the dangerous direction (trace allocator with a delete option: +123 mnats at 95 kB) | E4 |
| Pairwise product/branch terms | **Real at sublayer scale, negligible at head scale**: head pairs additive to ~10 %; MLP/attn sublayer pairs have cross terms as large as the unary costs, positive sign = layers substitute for each other | E2 |
| LLC as compressibility measure | Not useful: estimator invalid at 5/7 checkpoints (negative λ̂), ε- and chain-length-dependent at the minimum (277–2 127); the RTN knee barely varies (3.9–4.0 bits) so no correlation test possible | E3 |
| Hessian-proxy LLC ≠ LLC | Technically confirmed (λ̂_SGLD ≈ 10³ vs soft rank ≈ 8×10³) but the gap is mostly non-equilibration | E3, R2 |
| Quotient canonicalization | ≤ 1 mnat, sometimes negative; per-channel grids already absorb the scale gauges | E10 |
| Monomial-aware step sizes | Nothing to act on at quantization radii | E1, E4 |
| The compiler's target regime | Right (structural, extreme, collective) — but the information that worked there was measured, not derived | R3.2, E9 |

**Net**: the diagnosis is right, the proposed extraction machinery is not the tool. The
docx's own falsification condition (§12) was met: the higher-order surrogate did not
predict held-out structured perturbations better than measuring them, and its selected
architectures were not needed to beat the baselines.

## 3. The relaxation results (why retraining changes the question)

Compression restricts the loss to a submanifold M (lattice, hyperplane, determinantal
variety, diagonal); PTQ evaluates L at the nearest point of M to w*, compress+retrain
evaluates min_M L. The program measured the difference directly:

| observation | number | consequence |
|---|---|---|
| Relaxable fraction of a component's quench cost | ≥ 0.85 for all 40 attention/MLP gates; ≥ 0.95 for attention; whole layer-0 attention 86.6 → 3.0 mnats at lr 10⁻³ | one-shot costs are upper bounds off by ~10× |
| Quench ranking vs relaxed ranking | Spearman 0.94 | quench measurements remain valid *selectors* |
| Linear response (global OBS block / Schur complement) vs relaxed cost | Spearman −0.53 | the relaxation is nonlinear rerouting; no quadratic object predicts it |
| Path dependence, single component | none: anneal ≡ quench-then-recover; hysteresis loops close to −0.7 mnats (below start) | no basin selection per component |
| Path dependence, collective | strong: on the global-field down-leg nothing regrows (31/32 neuron groups, 3/4 MLPs stay dead; KL +14 vs −0.6 on the up-leg at the same h) | basin selection is a collective phenomenon |
| Init dependence under a full epoch | tying: layer-0 init +18.7 vs mean +33.6 vs best-single +41.9 | discrete/global edits keep their selection information |
| Exact-teacher targets vs one-hot | 1–10 mnats better at every budget; 1 000 soft steps beat 4 834 one-hot steps | label noise, not steps, limited recovery |
| Bit-width annealing (latents kept) | within ±5 mnats of direct | the path into the lattice is irrelevant; the endpoint and the budget are not |

So the useful map is: **rank with finite-radius quench measurements, budget by relaxing,
and reserve landscape-selection effort for collective/discrete choices** (which layers
share a block, in what order components die, which subspace to narrow into).

---

## 4. Beyond the static Hessian: observables and the compression each enables

This is the part that generalises. "Landscape information" is any statistic of the loss
around the trained model; the Hessian is the cheapest and, per §2–3, the wrong one for
exactly the decisions that reach the floor. Below, each observable is defined, its status
on this testbed recorded, and the compression action it informs stated. Notation:
L(w) = exact-teacher KL (population loss; label noise removed); Ω_c(w) = a gauge-invariant
component observable (‖W_Q W_Kᵀ‖², ‖W_V W_O‖², Σ_j‖W_in[:,j]‖‖W_out[j,:]‖, sublayer output
power); m_c = √(Ω_c/Ω_c(w*)).

### 4.1 Static, T = 0

| observable | definition | status | informs |
|---|---|---|---|
| Hessian diagonal / traces | tr H_i by Hutchinson, exact HVPs | measured (HAWQ) | bit widths ≥ 2 per tensor (correct ranking, constant-factor error) |
| Activation correlators | XXᵀ per matrix input; the exact Hessian of the layer reconstruction | measured (GPTQ) | *how* to round: OBS compensation within a matrix |
| Global-Hessian compensation | one damped Newton step on the true L over still-fp32 parameters after each tensor | measured (E7) | recovers half of what the layerwise proxy leaves: +0.7 / +4.1 / +34 at 4 / 3 / 2 bits vs +2.1 / +8.8 / +72 |
| Finite-radius profiles | L(w* + (α−1)d) on architecture-aligned lines d, α ∈ [0, 2] | measured (E1) | classifies each component as free / quadratic / plateau+cliff / power; the only reliable *structural* cost at T = 0 |
| Measured sensitivities | ΔL of one tensor alone at each width of the menu (243 GPTQ runs, 15 min) | measured (E4) | the allocation signal that built the bytes frontier; extends the menu below 2 bits |
| Pairwise non-additivity | R(α₁, α₂) = ΔL(α₁,α₂) − ΔL(α₁,1) − ΔL(1,α₂) | measured (E2) | tells you at which granularity independent saliency stops (heads: fine; sublayers: not) |
| Gauge orbits | GL(d_head) per head, ReLU scaling, permutations | measured (E10) | canonicalization; negligible here |

### 4.2 Dynamical / relaxational

| observable | definition | status | informs |
|---|---|---|---|
| Relaxed (adiabatic) cost | L*(α=0) − L*(1) with the retained parameters re-optimised; anneal or quench-then-recover | measured (R1) | the true budget of a structural edit; 10× below quench; path-independent |
| Relaxation time τ_c | convergence rate of the relaxed cost with recovery steps | partially (50/200/600-step points) | how much recovery a given edit needs; slow τ = the retained parameters must move along flat directions |
| Field response m_c(h) | Ω_c at the minimum of L + hΩ_c; a rate–distortion curve with a free endpoint | measured (R3.1) | what fraction of a component is decorative (QK logit scale: 80–90 %), and the cost of the functional remainder; continuous vs jump collapse → "shrink" vs "delete" |
| Cross-susceptibilities χ_cc′ | ∂m_c′/∂h_c, drift-corrected | measured (R3.3) | circuit discovery with relaxation: χ < 0 = substitutes (merge candidates; attention across layers), χ > 0 = serial circuits (delete together; attn+mlp within a layer) |
| Global-field phase diagram | all m_c(h) under one group-lasso field, up and down in h | measured (R3.2) | the parameter-axis frontier and the natural deletion *order*; hysteresis marks the irreversible, collective regime |
| Drift at h = 0 | ‖w − w*‖ after free fine-tuning at fixed loss | measured (≈ 4 in 200 steps) | size of the flat manifold at the recovery scale; sets expectations for any sampler |

### 4.3 Finite temperature (correlators)

The natural object is the Gibbs measure π_T ∝ exp(−L/T − γ‖w − w*‖²/2) at a temperature
matched to the loss budget, ⟨ΔL⟩_T ≈ ε. Its second moments are, by fluctuation–dissipation,
the linear-response susceptibilities at that scale:

    ⟨δw δwᵀ⟩_T = T·χ,        χ → H⁻¹ as T → 0 in a quadratic basin,
    ⟨δΩ_c δΩ_c′⟩_T = T·χ_cc′  (the finite-T version of R3.3).

| observable | status | what happened / what would make it work |
|---|---|---|
| Per-weight posterior variance (→ posterior-variance bit allocation, finite-T OBS) | **measured, failed** (R2) | SGLD chains equilibrate only directions with relaxation time 2/(ε(nβh+γ)) ≲ chain length — here only W_U. Everything else reported ε × steps (pure diffusion); the map was uninformative and the allocation degenerate. Weight-space correlators of a 2×10⁵-dim flat manifold are not obtainable by sampling at any useful temperature. |
| Component-observable correlators ⟨δΩ_c δΩ_c′⟩ | **not run; the right target** | 72 observables instead of 2×10⁵ coordinates; each Ω_c is a coarse, smooth function that equilibrates on the time-scale of the *retained* dynamics, not the slowest weight direction. Estimator: covariance of Ω_c across SGD/SGLD iterates during a short fine-tune at fixed lr (SWAG-style; effective T ~ η/B). Would give R3.3's χ matrix from one run instead of 24 field runs, plus its diagonal (each component's thermal variance = how loosely it is pinned). |
| Loss–observable correlators ⟨δL δΩ_c⟩ | not run | the finite-T analogue of the envelope derivative ∂L*/∂h_c: components whose norm fluctuations do not correlate with loss fluctuations are decorative at that scale — a one-run replacement for the h-sweep. |
| Activation correlators at finite T | not run | XXᵀ averaged over π_T instead of at w*: the GPTQ Hessian robustified against the very lattice perturbation it is compensating (probably a small effect, cheap to test). |
| Time correlators / relaxation spectrum | not run | autocorrelation of Ω_c along a fine-tune: its decay time is τ_c of §4.2 measured without perturbing anything; the spectrum of decay times is the "relaxation spectrum" of the network. |

The lesson from R2 is not that finite-temperature information is empty; it is that
weights are the wrong variables to thermalise. Coarse observables (component norms,
sublayer output power, per-position loss) are the analogue of collective coordinates —
low-dimensional, smooth, fast — and every finite-T quantity should be defined on them.

### 4.4 Data-side and representational observables

| observable | status | informs |
|---|---|---|
| Per-position loss profile | recorded in every CSV | which context lengths a compressed model sacrifices first; the memory low-pass hypothesis is testable from existing data |
| Loss conditioned on the true belief state / on process modes | not run | which parts of the 18-simplex the model gives up (slow modes of the transition matrix first?); the process-level view of what compression removes |
| Residual-stream subspace: belief probe vs PCA | measured (E8) | *PCA wins by 4–7×*: the stream carries position and token identity that attention needs; narrow d_model into the statistical subspace, not the physically motivated one |
| Cross-layer block similarity (weights or functions) | implicit in E9 | tying candidates; the init that survives a full epoch is layer 0, not the mean |
| Best k-state HMM fit to a compressed model's predictions | not run | whether carved models are predictors of coarse-grained processes; the "effective number of states vs parameters" curve would be the cleanest process-level statement of the compression flow |

### 4.5 What compression each observable class enables — the pipeline that follows

1. **Bit allocation** (per tensor, menu incl. ternary/delete): measured sensitivities
   (4.1). Curvature is acceptable only if the menu has no structural entries.
2. **Rounding**: activation correlators (GPTQ); global-Hessian Newton steps on top (E7)
   when no retraining is allowed. Untested but obvious: GlobalGPTQ under the measured
   allocation.
3. **Recovery**: short, exact-teacher, direct (no annealing). This is not "landscape
   information" but it decides what the other observables are worth: only rankings
   survive it, magnitudes do not.
4. **Which components to delete / how much to shrink**: field response m_c(h) and the
   global-field phase diagram (4.2). The up-leg of one sweep is the parameter frontier;
   its deletion order is the compile order, and it is *irreversible*, so it must be
   chosen once (the collective regime where selection information matters).
5. **Merge / delete-together**: cross-susceptibilities (4.2), or their finite-T
   correlator version (4.3) once measured. Compensating pairs → merge; cooperating → treat
   as one unit.
6. **Tie / recur**: block similarity plus enumeration of inits (E9); the surviving
   architecture of R3.2 (one attention + one MLP) says a single recurrent block is the
   natural target; combining tying with the field sweep is the untested composition.
7. **Narrow the residual stream**: PCA subspace, not the belief probe (E8).
8. **Leave the transformer**: process-level observables (4.4) to identify the
   coarse-grained HMM a compressed model implements, then distil into it. The only route
   to the 864-dof floor.

### 4.6 The physical picture that survives

- Flat directions come in three kinds — gauge (exact, small), redundant (removable), and
  saturated (a dynamical consequence of training, not removable). Static curvature cannot
  distinguish the last two; finite-radius profiles and field response can.
- Single components live in one basin (reversible), collectives do not (dead components
  never regrow: once the network reroutes, the reviving gradient is gone). The
  "singular" structure worth studying is this collective irreversibility, not local
  algebraic geometry at w*.
- The relaxed landscape L|_M has a well-defined effective architecture at each
  parameter count (one layer at 20k params here), reached along an ordered path that the
  field sweep discovers. Whether that endpoint is the predictor of a coarse-grained
  process is the open question that would connect the compression flow to the physics
  of the generating HMM.

---

## Appendix A — Experiment index

| id | what | where | status |
|---|---|---|---|
| E0 | exact-teacher loss protocol | `teacher.py` | adopted everywhere; dedicated variance test not run |
| E1 | structured finite-radius profiles + controls | `geo_probes.py`, log entry E1 | done |
| E2 | pairwise gate grids | same, entry E2 | done |
| E3 | SGLD LLC vs Hessian soft rank, 7 checkpoints | `geo_llc.py`, entry E3 | done (estimator limits documented) |
| E4 | measured-sensitivity allocation, extended menu; +1 000-step soft QAT | `geo_alloc.py`, entries E4 ×3 | done — bytes frontier |
| E5 / E6 | monomial step sizes / branch compiler | — | not built (gated out by E1/E2) |
| E7 | global-Hessian GPTQ | `geo_global_gptq.py`, entry E7 (hydra) | done |
| E8 | residual-stream projection (probe / PCA / random) | `geo_beliefproj.py`, entry E8 (hydra) | done |
| E9 | layer tying | `geo_tie.py`, entry E9 (hydra) | done — params frontier point |
| E10 | gauge canonicalization | `geo_gauge.py`, entry E10 | done (negligible) |
| R1.1–1.2 | adiabatic costs, linear response, hysteresis | `geo_anneal.py`, entry R1 (+ addendum, hydra) | done |
| R1.3 | adiabatic compiler | — | superseded by R3.2 |
| R1.4 | annealed vs direct quantization | `run_geo_annealq.py`, entry R1.4 | done — 70 kB frontier |
| R2.1–2.2 | SGLD non-Gaussianity map, posterior-variance allocation | `geo_thermal.py`, entry R2 | done, negative |
| R2.3 | finite-T GPTQ | — | dropped with cause |
| R3.1 | per-component field sweeps | `geo_field.py`, entry R3 (+ hydra) | done |
| R3.2 | global-field phase diagram | `run_geo_globalfield.py`, entry R3 | done — params frontier |
| R3.3 | cross-susceptibilities | `run_geo_field.py --stage r33`, entry R3.3 (hydra) | done |

## Appendix B — Protocol notes carried forward

- Reference quantizers untouched; everything new lives in `src/compress/experimental/`;
  deviations are named arguments and appear in the CSVs.
- Calibration and fine-tuning draw from the train split only; the test split is reported,
  never selected on.
- Two implementation traps found and fixed, worth knowing: (i) hooks that edit a ReLU
  output in place break autograd during recovery (clone first); (ii) progressive
  quantization must keep fp32 latents — baking between stages freezes every latent on the
  coarser lattice and STE can no longer flip cells (+565 mnats vs +29 with latents kept).
- Hydra: single-core jobs (`-n 1 -m 2`) run ~10× faster than the contended local box;
  `~/HMM_compression` is compute-only, results are synced back.
