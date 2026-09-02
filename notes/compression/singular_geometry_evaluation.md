# Landscape information for compression: evaluation, results, and the observables that matter

*Written 2026-08-31, revised 2026-09-01 after the full program ran. Evaluates the ideas in
`singular_landscape_transformer_compression.docx` (draft of 2026-08-17) against the data in
`experiment_log.md`, reports what the geometry program (E0–E10), the relaxation program
(R1–R3, `relaxation_program.md`) and the finite-temperature correlator program (R4–R4c)
found, and then sets out — the part that survived — which observables beyond the static
Hessian carry compression information and what each one lets you do.*

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
| Cross-susceptibilities χ_cc′ | ∂m_c′/∂h_c, drift-corrected | measured (R3.3) | circuit discovery with relaxation (χ_cc′ = relative change of Ω_c′ when c is squeezed): χ > 0 = substitutes (partner grows; mlp.L2 ↔ attn.L2 / mlp.L1 / attn.L3 → merge candidates), χ < 0 = serial chains (partner decays too; layer 1 → attn.L2, mlp.L2 → attn.L3 → delete together) |
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
| Component-observable correlators ⟨δΩ_c δΩ_c′⟩ | **measured (R4), FDT test failed** | 72 observables along Adam / SGD / SGLD trajectories of 2 000–4 000 steps: Spearman(corr, χ_R3.3) = −0.09 / +0.03 / −0.12, sign agreement with −χ at chance. Cause: the autocorrelation never decays within the window (τ_c saturates the 60-lag cap under every dynamics) — one slow drift mode, so the second moments are drift statistics. **R4b** (30 000–60 000 steps, γ = 100): still no equilibration — the chain was filling the γ-ball (√(N/γ) ≈ 45) with τ_flat = 2/(εγ) ≈ 7×10⁴ steps. **R4c (γ scan)**: at γ = 10⁵ the SGLD chain equilibrates (τ = 130 steps, 230 τ window, ‖δw‖ = 1.43 = √(N/γ)) and the FDT sign relation to χ emerges on the strongly coupled pairs (0.727 vs chance 0.5); full-matrix ranks stay at the sampling-noise floor. γ/nβ is the curvature-scale dial (10⁵ ↔ the ~230 stiffest directions). Only SGLD noise is thermal — SGD never decorrelates, Adam is white. |
| Thermal variance var(x_c) as a selector | **measured (R4, R4c): partial positive** | Ranks the field-measured *free fraction* of a component at Spearman 0.71–0.80 under SGLD, robust across γ ∈ {10², 10³, 10⁴, 10⁵} and so not an equilibration artefact — a scale-redundancy detector (over-sharpened QK). Weaker under SGD (0.62–0.67) and unreliable under Adam (0.25–0.61, preconditioning re-weights the observables). Not a cost or deletion signal: as a deletion order it loses to the static quench order at every size (+81 vs +20 mnats at 74k params) and to the field sweep by 10×. |
| Loss–observable correlators ⟨δL δΩ_c⟩ | **measured (R4): negative** | Spearman with the field cost 0.41 / −0.14 / −0.27 across dynamics — no consistent signal at these trajectory lengths. |
| Activation correlators at finite T (ThermalGPTQ) | **measured (R4): negative** | ⟨2XXᵀ⟩ over 50 Adam states: 2.2 / 9.7 / 113 mnats at 4 / 3 / 2 bits vs 2.0 / 9.6 / 105 for H(w*) without propagation; the propagated reference (71.6 at 2 bits) shows that conditioning on already-quantized upstream layers, not temperature, is what matters. |
| Time correlators / relaxation spectrum | **measured (R4, R4b, R4c)** | At γ = 100 the autocorrelation of every observable stays positive over the whole window at 2 000 and at 30 000–60 000 steps (τ only bounded below, no spectrum). Localization resolves it: SGLD τ ≈ 3 850 / 1 050 / 130 steps at γ = 10³ / 10⁴ / 10⁵ (OU flat-direction prediction 2/(εγ) = 6 670 / 670 / 67) — the ~1/γ scaling holds and each point sits within a factor ~2 of the prediction, so the observables' slow time is set by the localization, not by the model. A modest genuine spread appears once resolved (median 130 vs max 240 steps at γ = 10⁵). |

The lesson from R2 was that weights are the wrong variables to thermalise; R4 adds that
coarse observables are the right variables but still need a genuinely equilibrated
trajectory — at the γ = 100 used there, no observable's autocorrelation decayed inside any
window tried (up to 60 000 steps), so the second moments were drift statistics, not
fluctuations. What survived from R4 alone: the thermal variance is a usable detector of
*scale* redundancy (which parts of a component are decorative), and nothing
finite-temperature improved rounding or deletion over the T = 0 measurements.
R4c settled the rest: the R4/R4b failure was under-localization. With γ chosen so that
τ_flat = 2/(εγ) fits many times into the window (γ = 10⁵ here: τ = 130 steps, 230 τ of
stationary data, ‖δw‖ pinned at √(N/γ)), one SGLD chain recovers the *sign structure* of
the strong response coefficients χ (0.727 agreement on |χ| > 0.2 pairs vs 0.5 chance) and
the scale-redundancy ranking (ρ = 0.76) — a validated coarse screen for merge/
delete-together relations at ~1/24 the cost of the field sweeps. The precision numbers
(small χ, ranks, actual costs) still require the T = 0 field sweeps: the correlator noise
floor at affordable sample counts (~0.07) sits above the median |χ| (0.02). γ/nβ acts as
the curvature-scale dial selecting which directions the chain thermalises — the operational
form of "finite-scale" landscape information. Optimizer noise is no shortcut: SGD's η/B
noise never decorrelates the observables and Adam's preconditioned noise is white.

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
| R2.3 | finite-T GPTQ (weight-space covariance) | — | dropped with cause; the activation-space version was later run as R4's ThermalGPTQ — negative |
| R3.1 | per-component field sweeps | `geo_field.py`, entry R3 (+ hydra) | done |
| R3.2 | global-field phase diagram | `run_geo_globalfield.py`, entry R3 | done — params frontier |
| R3.3 | cross-susceptibilities | `run_geo_field.py --stage r33`, entry R3.3 (hydra) | done |
| R4 | coarse-observable correlators; thermal-variance deletion order; ThermalGPTQ | `geo_correlators.py`, entry R4 (hydra) | done — FDT negative at γ = 100, deletion/rounding negative |
| R4b | long trajectories (30k–60k steps) | same runner, entry R4 addendum (hydra) | done — no equilibration; diagnosed as under-localization |
| R4c | γ-localization scan {10³, 10⁴, 10⁵} × {SGLD, SGD, Adam} | same runner `--gammas`, entry R4c (hydra) | done — equilibrated SGLD recovers the FDT sign structure of χ |

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
