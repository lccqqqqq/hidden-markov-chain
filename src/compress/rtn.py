"""
RTN — round-to-nearest. The calibration-free baseline; no paper.

  ŵ = s · (clamp(round(w/s) + z, qmin, qmax) − z)

with (s, z) from the min/max (asymmetric) or max|w| (symmetric) of each channel. No error
compensation, no data. Everything else in this package should beat it.
"""
from __future__ import annotations

import copy

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param


class RTN(Quantizer):
    name = "RTN"
    citation = "baseline (no reference)"

    @t.no_grad()
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        model = copy.deepcopy(model)
        for name, p in model.named_parameters():
            if not is_quantized_param(name):
                continue
            w2 = self._channel_view(p.data, self.granularity, name)
            scale, zero, qmin, qmax = self.grid(w2, self._bits(name), self.symmetric)
            fq = self.fake_quant(w2, scale, zero, qmin, qmax)
            p.data.copy_(self._from_channel_view(fq, p.shape, name))
        return model
