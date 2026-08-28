"""
New / modified compression methods live here and ONLY here.

Anything in this package is compared *against* the frozen reference implementations in
`compress.*`; it never edits them. Candidate directions (notes §3.3):
  * global-Hessian OBS compensation across layers (tractable at 2e5 params)
  * rounding as projection onto the lattice under the H-metric ‖δw‖_H (cf. SqueezeLLM's
    Fisher-weighted codebook — read that first)
  * GPTQ with the true Fisher/Hessian in place of the layerwise proxy 2XXᵀ
"""
