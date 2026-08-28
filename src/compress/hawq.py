"""
HAWQ-V2 — Dong et al., NeurIPS 2020 (V1: ICCV 2019).   [NOT YET IMPLEMENTED]

A bit-*allocation* policy, not a rounding rule:
  * per quantized tensor, estimate tr(H) by Hutchinson (Rademacher probes, Hessian-vector
    products through the true cross-entropy loss) — this is where the landscape enters.
  * sensitivity Ω_i = tr(H_i)/n_i · ‖Q(W_i) − W_i‖² ; solve an ILP assigning bits ∈ {2,4,8}
    per tensor under a total byte budget, minimising Σ Ω_i.
  * then apply `base` (RTN, later STE-QAT) with the resulting {name: bits} dict.
Yields a mixed-precision {param_name: bits} map; pass it to `count_bytes`.
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer


class HAWQv2(Quantizer):
    name = "HAWQ-V2"
    citation = "Dong et al., NeurIPS 2020"

    def __init__(self, byte_budget: int, base: type[Quantizer], choices=(2, 4, 8), n_probes: int = 64):
        super().__init__(bits=max(choices))
        self.byte_budget, self.base, self.choices, self.n_probes = byte_budget, base, choices, n_probes

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        raise NotImplementedError("HAWQ-V2 not implemented yet — see module docstring.")
