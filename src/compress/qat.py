"""
Quantization-aware training.   [NOT YET IMPLEMENTED]

Two distinct named methods — do not blend them:

STEQAT — straight-through estimator (Bengio, Léonard & Courville 2013).
  * forward uses ŵ = fake_quant(w) with a *fixed* grid (min/max recomputed each step),
    backward passes dL/dŵ straight to w:  w_eff = w + (ŵ − w).detach()
  * fine-tune from the saved checkpoint at low LR (default 1e-4), 1–2 epochs.
  * Ternary via this route is "ternary STE-QAT", not BitNet (BitNet trains from scratch
    with absmean scaling and 8-bit activations).

LSQ — Esser et al., ICLR 2020.
  * as STE-QAT but the scale s is a learned parameter per channel, with gradient scale
    1/sqrt(N · Q_max); initialised s = 2·mean|w|/sqrt(Q_max).

Implementation plan: factor the inner loop of train.py into a reusable `train_steps(model,
loader, ...)`, wrap quantized parameters with a parametrization (torch.nn.utils.parametrize)
that applies the STE, then call the shared loop.
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer


class STEQAT(Quantizer):
    name = "STE-QAT"
    citation = "Bengio et al. 2013 (STE)"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 epochs: int = 1, lr: float = 1e-4):
        super().__init__(bits, granularity, symmetric)
        self.epochs, self.lr = epochs, lr

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        raise NotImplementedError("STE-QAT not implemented yet — see module docstring.")


class LSQ(Quantizer):
    name = "LSQ"
    citation = "Esser et al., ICLR 2020"

    def __init__(self, bits: int, granularity: str = "per_channel", epochs: int = 1, lr: float = 1e-4):
        super().__init__(bits, granularity, symmetric=True)
        self.epochs, self.lr = epochs, lr

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        raise NotImplementedError("LSQ not implemented yet — see module docstring.")
