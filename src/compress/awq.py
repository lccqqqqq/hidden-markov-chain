"""
AWQ — Lin et al., MLSys 2024 (arXiv:2306.00978).   [NOT YET IMPLEMENTED]

Reference algorithm:
  * For each linear layer collect per-input-channel activation magnitudes ‖x_i‖ (mean |x|).
  * Scale W ← W · diag(s), and fold diag(s)⁻¹ into the preceding op, with s_i = ‖x_i‖^α.
  * α ∈ [0, 1] searched on a grid of 20 points per layer, minimising ‖W X − Q(W diag s) diag(s)⁻¹ X‖².
  * Then RTN on the scaled weight. Weight-only. Landscape info: diag(X Xᵀ) only.
Note: folding diag(s)⁻¹ into the *previous* op is only free when that op is linear (no
LayerNorm here, so residual-stream inputs cannot absorb it — must be handled explicitly).
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer


class AWQ(Quantizer):
    name = "AWQ"
    citation = "Lin et al., MLSys 2024, arXiv:2306.00978"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False, alpha_grid: int = 20):
        super().__init__(bits, granularity, symmetric)
        self.alpha_grid = alpha_grid

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        raise NotImplementedError("AWQ not implemented yet — see module docstring.")
