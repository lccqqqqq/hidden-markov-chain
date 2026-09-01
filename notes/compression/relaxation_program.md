# Relaxation-aware compression: adiabatic annealing, finite-temperature OBS, field response

*Plan, 2026-08-31. Follows `singular_geometry_evaluation.md` (E0–E10) and the discussion of
what compression does to the landscape. Nothing here has been run.*

## 0. Why these three, and what they share

Every method benchmarked so far estimates a **quench** cost — the loss increase from
perturbing w* with everything else frozen — and then either accepts it (PTQ) or retrains
away as much of it as it can (QAT / fine-tune). The data say the two regimes behave very
differently:

| fact (from `experiment_log.md`) | what it says |
|---|---|
| FWSVD vs SVD: 68 mnats apart pre-FT, ≤ 0.7 after 1 epoch (ρ = 0.25); wanda vs magnitude the same | quench costs differ; **relaxed** costs barely do |
| ρ = 0.0625: Fisher edge *survives* FT by 28 mnats | at extreme compression, the projection selects a basin |
| E1: layer-0 QK head costs 12 mnats to remove, Hessian says 0.65; power p ≈ 3.5 | the quench cost is already non-quadratic at finite radius |
| E4: trace-based allocation with a delete option: +123 mnats at 95 744 B (deletes QK) | curvature ranks *structural* edits wrongly |
| E4: additive model holds to 70 kB, fails at 60 kB (40 predicted vs 117 joint) | cross-terms appear when deletes/1-bit enter |
| E4: 200 soft-target QAT steps beat one-epoch LSQ at 70 kB | most relaxation is fast and the exact teacher helps |

The missing quantity is the **relaxed** cost of an edit — the loss after the retained
parameters re-adjust — and its structure (which components relax, how fast, whether they
relax into the same basin). The three ideas are three ways to measure it at increasing
generality:

| idea | what is measured | relaxation | temperature | prescribes endpoint? |
|---|---|---|---|---|
| 1. adiabatic annealing | ΔL_ad(c) = L*(α=0) − L*(1) along a slow gate path | full, nonlinear | 0 | yes (component removed) |
| 2. finite-temperature OBS | posterior covariance Σ_T; OBS with Σ_T/T in place of H⁻¹ | thermal, linear response at scale T | T = tolerance | yes (lattice / zero) |
| 3. field response | F(h) = min_w [L + hΩ_c], m_c(h), χ_c, χ_cc′ | full, nonlinear | 0 (or T) | **no** (model chooses how much to give up) |

Common theory. For a gate α on component c with L*(α) = min_w L(w, α):

    dL*/dα = ∂L/∂α |_{w*(α)}                                (envelope theorem)
    d²L*/dα² = L_αα − L_αwᵀ H⁻¹ L_wα                        (Schur complement)

so first-order saliency already includes relaxation; the second-order term is the linear
response (what OBS/GPTQ compute with H⁻¹, what E7 computes globally). Ideas 1 and 3 measure
the full nonlinear response; idea 2 measures the linear response at finite temperature,
which sees plateaus and cliffs that H at w* cannot. All three reduce to the Hessian picture
when the landscape is quadratic at the relevant scale — which is the null hypothesis.

Shared protocol (inherits `compression_survey.md` §5 and the geometry program):
d64 model; exact-teacher KL (`compress/experimental/teacher.py`) for all measurements;
recovery = Adam 1e-4, batch 128 (the STE-QAT loop, `geo_recover.py`), soft targets by
default with one-hot as a logged variant; calibration/fine-tune data from the train split
only; test split for reporting; 3 seeds for anything stochastic; new code under
`compress/experimental/`; results under `results/compression/geometry/`; each experiment
logged in `experiment_log.md`.

Gate vocabulary (already implemented in `geo_probes.py`, exact parameter-space lines):
qk(l,h), head(l,h) [= W_O[h]], attn(l), mlp(l), ngrp(l,g) [neuron groups of 32 by activation
variance], plus tensor-level widths from `geo_alloc.py` (del / 1 / T / 2 … 8).

---

## R1. Adiabatic annealing with retraining

### R1.1 Per-component adiabatic cost

**Procedure.** For each gate c (16 heads via `head`, 16 via `qk`, 8 sublayers, 32 neuron
groups = 72 components): anneal α from 1 to 0 linearly over T_anneal steps while
fine-tuning (soft targets); at α = 0 continue for T_hold steps; record the calib KL along
the path and the test KL at the end. Schedules T_anneal ∈ {50, 200, 800}, T_hold = 100.
Baselines per component: quench ΔL_quench = ΔKL(α=0) from E1; "quench then recover" =
α set to 0 instantly, then T_anneal + T_hold recovery steps (same total compute).

**Quantities.**
- ΔL_ad(c; T_anneal) and its extrapolation in 1/T_anneal to the adiabatic limit.
- Relaxable fraction R_c = 1 − ΔL_ad/ΔL_quench.
- Rank correlation (Spearman) across components between ΔL_quench, ΔL_ad, and the
  Schur-complement prediction S_c = L_αα − L_αwᵀH⁻¹L_wα (computed with CG on exact HVPs,
  E7 infrastructure).
- Relaxation time: fit ΔL(T_anneal) − ΔL_ad ∝ T_anneal^(−ν) or exp(−T/τ_c); τ_c per
  component.

**Decision rules.**
- Spearman(ΔL_quench, ΔL_ad) > 0.9 and S_c predicts ΔL_ad within 25 % ⇒ linear response
  is sufficient; landscape-guided selection can use global OBS (E7) and nothing beyond.
- Spearman < 0.7 ⇒ quench-based rankings (HAWQ, OBS, wanda, Fisher-SVD) target the wrong
  quantity in the retraining regime; use ΔL_ad as the saliency for the compiler (R1.3).
- If ΔL_ad ≈ "quench then recover" at equal compute for all c ⇒ the path does not matter,
  only the endpoint and the recovery budget; annealing has no value.
- Prediction to check: layer-0 QK heads (quench 3–12 mnats, E1) have R_c > 0.8 (attention
  reroutes); the top neuron group of layer 3 (quench 212 mnats) has R_c < 0.5.

**Cost.** 72 components × 3 schedules × (T_anneal + T_hold) ≈ 72 × 1 350 steps ≈ 10⁵
steps ≈ 20 epochs-equivalent of d64 fine-tuning (each step ≈ 20–40 ms on CPU) ≈ 1 h.

### R1.2 Hysteresis and basin multiplicity

**Procedure.** For the 16 components with the largest ΔL_quench: anneal 1 → 0 (T_anneal =
200, hold 100) then 0 → 1 (same) and measure the loss and the weight displacement
‖w − w*‖ at the return point. Repeat with 3 seeds of data order.

**Quantities.** Loop area ∮ (∂L/∂α) dα; return loss L(α=1, return) − L(w*); return
displacement; per-position loss profile at the return point vs the original.

**Decision.** Return loss < 1 mnat and displacement small ⇒ reversible, single basin.
Return loss ≫ 1 mnat or a different per-position profile ⇒ the component's role was taken
over irreversibly (first-order transition in α): this is direct evidence for the basin
multiplicity seen at FWSVD ρ = 0.0625, and the component is one whose deletion is a
discrete, path-dependent decision — the docx's "branch" in operational form.

### R1.3 Adiabatic structured compression (the compiler with the right saliency)

**Procedure.** Greedy structural compression using ΔL_ad from R1.1 as the saliency, with
re-measurement after each accepted edit (the adiabatic cost of the next component is
measured on the already-compressed, recovered model — the Schur-complement logic done
exactly). Grammar: delete head, delete neuron group, delete sublayer. Targets: 150k, 100k,
60k, 40k, 25k params. Baselines at identical param count and identical total recovery
compute: (a) greedy by quench cost (E1 unary), (b) wanda/magnitude pruning + FT (refset),
(c) FWSVD + FT, (d) scratch-trained model of the same size (sweep). Recovery reported at
{as-annealed, +1 epoch}.

**Decision.** > 5 mnats better than (a)–(c) at ≤ 60k params ⇒ relaxation-aware selection
is the missing ingredient. Matches (d) ⇒ compress-then-anneal is only a compute shortcut;
report the compute ratio. Worse than (a) ⇒ re-measurement/greedy path is over-fitting the
calibration set — check with the held-out calib split.

### R1.4 Annealed quantization

**Procedure.** Bit-width annealing 8 → 6 → 4 → 3 → 2 with N steps of STE-QAT (soft
targets) at each stage vs direct 2-bit STE-QAT with 4N steps, N ∈ {50, 200}; and soft-to-
hard rounding: fake-quant strength β from 0 to 1 over the same budget. Uniform 2-bit and
the E4 measured-sensitivity allocation at 70 272 B.

**Decision.** Beats the current 70 272 B frontier (+16.1 mnats, E4 measured + 200 soft
QAT steps) by > 3 mnats at equal steps ⇒ the path into the lattice matters; log as a new
frontier point. Otherwise quantization is endpoint-only and R1 is a structural tool.

---

## R2. Finite-temperature OBS

### R2.1 The non-Gaussianity map (diagnostic, no compression)

**Procedure.** SGLD chains from E3 (`geo_llc.py`; nβ = 7.4×10⁵, γ = 100, ε = 3×10⁻⁷) plus
two hotter chains (nβ/10, nβ/100) so that ⟨ΔL⟩_T ≈ 1, 10, 100 mnats. Thin to ~1 000
samples per chain; store per-weight mean and variance σ_i² and, per output row of every
matrix, the row covariance (64 × 64). Also run with γ = 10 to check that the localization
is not what caps the variance.

**Quantities.** Gaussian prediction σ_i,quad² = T/(h_i + Tγ) with h_i the diagonal
curvature (Hutchinson per tensor, or the exact diagonal — affordable for 206k parameters
via 206k HVPs? no; use per-tensor Hutchinson diagonals with 256 probes). Ratio map
ρ_i = σ_i²/σ_i,quad² per tensor, and its distribution.

**Reading.** ρ ≪ 1 ⇒ cliff (variance capped by finite extent; expected for layer-0/1 QK);
ρ ≫ 1 ⇒ saturation / sub-quadratic growth (expected wherever λ̂_SGLD ≪ λ̂_quad is coming
from); ρ ≈ 1 ⇒ quadratic. This map is the direct test of the docx's premise at the scale
of a given loss budget, and it costs nothing beyond E3's chains.

**Decision.** If ρ ≈ 1 for > 90 % of weights at the 10-mnat temperature, the posterior is
Gaussian at compression scales and R2.2–R2.3 will not beat OBS/GPTQ; run one budget as a
check and stop.

### R2.2 Posterior-variance bit allocation

**Procedure.** For each tensor, bits from σ: b_i = ⌈log₂(range_i/σ_i)⌉ per channel, then
clamp to the menu; or, equivalently, feed Ω_i(b) = Σ_j (ΔW_ij(b))²/σ_ij² (the Mahalanobis
quantization error under the diagonal posterior) to the knapsack of `geo_alloc.py`.
Budgets 95 744, 82 944, 70 272, 60 000 B. Compare with the E4 measured-sensitivity
allocation (which costs 243 GPTQ runs) and the trace-based one, at PTQ and +200 soft QAT.

**Decision.** Within 1 mnat of measured-sensitivity at every budget ⇒ one SGLD chain
replaces the O(tensors × menu) measurement — a practical method. Beats it ⇒ the
per-weight (rather than per-tensor) resolution of σ is buying something; move to R2.3.
Deletes QK like the trace allocator did ⇒ the chain did not reach the cliff (γ too
strong or chain too short); check with R2.1 before concluding.

Prior art to cite: Louizos, Ullrich & Welling 2017 (Bayesian compression: prune where
σ/|μ| large); van Baalen et al. 2020 (Bayesian Bits); Hinton & van Camp 1993 (bits-back).
The new element is the temperature matched to the loss budget and the head-to-head with
curvature and measured sensitivities on a system with a known floor.

### R2.3 Finite-temperature GPTQ (OBS compensation with the sampled covariance)

**Procedure.** Subclass GPTQ (as `ExtGPTQ` does — reference untouched): for each matrix,
replace H = 2XXᵀ (d_in × d_in, shared across rows) by, per output row r, the inverse of
the sampled row covariance Σ_T,r/T (64 × 64 from ~1 000 samples; shrinkage toward the
XXᵀ proxy with a logged coefficient to control the estimation noise). Same column order,
same grid, same OBS update. Bits {4, 3, 2} per-channel asym; temperature = the chain whose
⟨ΔL⟩ matches the expected cost at that width (≈ 1, 5, 20 mnats).

**Variants.** (a) SWAG covariance instead of SGLD: covariance of SGD iterates during a
1-epoch fine-tune at fixed lr (η/B sets the effective temperature) — the cheapest version
and the one that scales. (b) Hybrid: Σ⁻¹ = H/T in the top-k Hessian eigen-subspace, sampled
elsewhere.

**Decision.** At 2 bits: beats GPTQ act_order (+58) by > 10 mnats without retraining ⇒
volume information beyond curvature is real and usable in a one-shot method — the docx's
finite-scale thesis confirmed in its practical form. Within noise of GPTQ ⇒ the row
posteriors are Gaussian at these scales; the SGLD/quad discrepancy in E3 lives in
directions quantization does not touch. Worse ⇒ estimation noise dominates; increase
shrinkage and samples, report the curve.

**Cost.** Chains: 3 temperatures × 2 000 steps ≈ 30 min each under load. GPTQ variants:
minutes.

---

## R3. Field response

### R3.1 Per-component field sweeps

**Procedure.** For component c, gauge-invariant field Ω_c: for heads, Ω = ‖W_Q[h]W_K[h]ᵀ‖_F²
(qk) or ‖W_V[h]W_O[h]‖_F² (vo); for neuron groups, Ω = Σ_j ‖W_in[:,j]‖·‖W_out[j,:]‖; for
sublayers, Ω = mean squared output activation on calib. Sweep h over 6 log-spaced values
chosen so that h·Ω_c(w*) spans 0.1–100 mnats; at each h fine-tune 300 steps (soft targets)
from w*, record m_c(h) = Ω_c(w*(h)) and ΔL(h) = KL(w*(h)) − KL(w*).

**Quantities.** The parametric curve (m_c/m_c(0), ΔL) — the relaxed rate–distortion curve
of the component with a free endpoint; χ_c = −dm_c/dh at small h; the order of the
collapse (continuous vs jump: is there an h at which m_c drops by > 50 % between adjacent
h values?); comparison at matched m_c with R1's adiabatic cost (field ≤ adiabatic ≤ quench).

**Decision / reading.** Jump ⇒ all-or-nothing component ⇒ grammar edit "delete";
continuous ⇒ "shrink / rank-reduce". Components whose field cost at m_c → 0 is < 1 mnat
are free after relaxation regardless of their quench cost — the list to compare with E1's
"flat" list and R1's R_c ranking. Note the gauge trap: h‖W_Q‖² alone is absorbed by
W_Q → W_Q A, W_K → W_K A⁻ᵀ; only product/gate/activation fields are admissible, and the
runner must assert that the fp32 function is unchanged under the gauge before trusting a
response.

### R3.2 Global-field phase diagram

**Procedure.** One field on the group norm Ω = Σ_c ‖component c‖ (heads via vo product,
neuron groups via in·out product, sublayers via output norm), swept in 12 values of h;
at each h, 1 000 fine-tune steps from the previous h (annealed in h, so this is also
adiabatic); record every m_c(h). Then increase back down to h = 0 for hysteresis.

**Quantities.** Death order of components vs h; sharpness of each death; hysteresis on
the way back; the loss–params curve obtained by hard-deleting every component with
m_c < 1 % of its initial value at each h, plus 200 recovery steps.

**Decision.** This is group-lasso structured pruning (Wen et al. 2016; Louizos et al. 2018
L0; Sheared LLaMA) read as a phase diagram. Compare its loss–params curve to R1.3 and to
the scratch sweep. First-order deaths with hysteresis for attention heads and continuous
shrinkage for MLP groups would confirm that heads want "delete" and MLPs want "narrow".

### R3.3 Cross-susceptibilities (circuit discovery with relaxation)

**Procedure.** For the 16 heads and 8 sublayer gates: apply the field to c alone at one
moderate h (the value where m_c falls to ~50 %), fine-tune 300 steps, and record the
change of every other component's Ω_c′: χ_cc′ = Δm_c′/Δh_c. Symmetrize; compare with
E2's quench-level non-additivity R(0,0) for the same pairs.

**Reading.** χ_cc′ < 0 (c′ grows when c is squeezed) ⇒ compensating partner ⇒ exchangeable
⇒ merge candidate (docx Table 3 "loss depends on g₁ − g₂"); χ_cc′ > 0 ⇒ cooperating ⇒
one circuit ⇒ delete both or neither; χ ≈ 0 ⇒ independent (E2's additive case). This is
the relaxed version of the pairwise term the docx wants and E2 measures at the quench
level; the two together say whether cross-terms are an artefact of freezing.

**Decision.** If |χ_cc′| is negligible for all pairs where E2 found |R| > 1 mnat, the
non-additivity seen in E2/E4 (below 60 kB) is relaxable and the compiler can treat
components independently once it retrains. If the compensating pairs persist, they are
the merge candidates for R1.3's grammar.

**Cost.** R3.1: 72 × 6 × 300 ≈ 1.3×10⁵ steps (≈ 1 h); R3.2: 24 × 1 000 ≈ 2.4×10⁴ steps;
R3.3: 24 × 300 steps. All cheap.

---

## 4. Order, gates, and what would settle it

| step | experiment | gate | compute |
|---|---|---|---|
| 1 | R2.1 non-Gaussianity map | needs E3 chains (running) + 2 hotter chains | 1 h chains, minutes analysis |
| 2 | R1.1 adiabatic costs + Schur comparison | none | ~1 h |
| 3 | R3.1 field sweeps (72 components) | none | ~1 h |
| 4 | R2.2 posterior-variance allocation | after 1 | minutes |
| 5 | R1.2 hysteresis, R3.3 cross-susceptibilities | after 2, 3 | 30 min |
| 6 | R2.3 finite-T GPTQ | only if R2.1 shows ρ ≠ 1 where quantization acts | 1 h |
| 7 | R1.3 adiabatic compiler, R3.2 global field | after 2, 3, 5 | 2–3 h each |
| 8 | R1.4 annealed quantization | independent | 30 min |

**Interpretation grid.**
- Quench and adiabatic rankings agree, Schur predicts ΔL_ad, ρ ≈ 1 everywhere ⇒ the
  Hessian is the whole story once retraining is allowed; the frontier improvements come
  from *where* the bits go (E4) and from recovery (soft targets), not from geometry beyond
  curvature. Docx: not useful.
- Rankings disagree, hysteresis loops exist, cross-susceptibilities persist ⇒ relaxation
  is nonlinear and path-dependent; the relaxed saliencies (R1/R3) are the right input to a
  structural compiler, and the docx's branch/merge grammar has measurable targets — but
  the measurements are dynamical (annealing, fields), not static normal forms.
- R2.3 beats GPTQ at 2 bits ⇒ finite-scale volume information is usable one-shot; the
  right generalisation of OBS is thermal, and the SLT quantity that matters is Σ_T, not λ.
- R1.3/R3.2 match the scratch-trained curve ⇒ the minimal transformer for this task is
  unique (a single fixed point); compression is a compute shortcut and the remaining
  physics is in the process-coarse-graining picture, to be taken up separately.
