"""
GPTQ — Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023 (arXiv:2210.17323).   [NOT YET IMPLEMENTED]

Reference algorithm (must be reproduced exactly; deviations are named constructor args):
  * Layerwise objective ‖W X − Ŵ X‖² with H = 2 X Xᵀ from calibration inputs to that layer
    (collect X with TransformerLens hooks: hook_resid_pre / attn.hook_q-input / mlp hook_pre ...).
  * Dampening: H += damp * mean(diag H) * I, damp = 0.01.
  * Columns quantized in fixed index order (act_order=False) or by descending diag(H)
    (act_order=True, a documented option).
  * OBS error compensation via the upper Cholesky factor of H⁻¹, lazy-batched with
    blocksize=128 (numerical, not algorithmic, difference).
  * Grid: RTN's min/max per row (i.e. the same `Quantizer.grid` as RTN).
  * Calibration: n_calib=128 random train sequences.

Ancestor: OBQ (Frantar & Alistarh, NeurIPS 2022) — exact OBS per row, greedy column order.
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer


class GPTQ(Quantizer):
    name = "GPTQ"
    citation = "Frantar et al., ICLR 2023, arXiv:2210.17323"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 n_calib: int = 128, damp: float = 0.01, act_order: bool = False, blocksize: int = 128):
        super().__init__(bits, granularity, symmetric)
        self.n_calib, self.damp, self.act_order, self.blocksize = n_calib, damp, act_order, blocksize

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(n_calib=self.n_calib, damp=self.damp, act_order=self.act_order, blocksize=self.blocksize)
        return s

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        # TODO(gptq):
        #   1. run `calib[:, :-1]` through the model with hooks, accumulating H = 2 X Xᵀ for
        #      the input of every quantized matrix (per layer, per matrix; heads share X).
        #   2. for each matrix: dampen H, Cholesky(H⁻¹) upper, loop over column blocks:
        #      quantize column j with `grid`/`fake_quant`, err = (w_j − ŵ_j)/Hinv_jj,
        #      w_{j+1:} −= err · Hinv_{j, j+1:}.
        #   3. W_E / W_pos / W_U: inputs are one-hot / identity — GPTQ degenerates to RTN
        #      there (H diagonal); document this.
        raise NotImplementedError("GPTQ not implemented yet — see module docstring.")
