"""
Shared protocol for all quantizers.

Protocol choices (applied identically to every method, recorded once here):
  * Weight-only quantization. Activations stay fp32.
  * Quantized tensors: every parameter whose name ends in one of WEIGHT_SUFFIXES
    (embeddings, positional embeddings, attention and MLP matrices, unembedding).
  * Biases are kept in fp32 (they are <1% of parameters).
  * Granularity "per_channel" means one (scale, zero) per *output* channel, i.e. per slice
    along the last dimension of the TransformerLens weight (TL stores weights as
    (..., d_in, d_out)). "per_tensor" means one (scale, zero) for the whole tensor.
    Group-wise g128 is meaningless for 64-wide matrices and is not offered.
  * Fake quantization: weights remain fp32 tensors constrained to the b-bit grid, so the
    model runs unchanged through TransformerLens. Bytes are *computed* by `count_bytes`,
    not measured.
  * Bytes accounting: quantized tensor = numel * bits / 8, plus per (scale, zero) pair
    2 + 2 bytes (fp16 each). Unquantized parameters count 4 bytes (fp32).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Iterable

import torch as t
from transformer_lens import HookedTransformer

WEIGHT_SUFFIXES = ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U")


def is_quantized_param(name: str) -> bool:
    return name.split(".")[-1] in WEIGHT_SUFFIXES


@dataclass
class QuantSpec:
    """What was done to the model, for logging. One entry per quantizer run."""
    method: str
    bits: int
    granularity: str            # "per_channel" | "per_tensor"
    symmetric: bool
    citation: str = ""
    extra: dict = field(default_factory=dict)   # any non-default args (must be logged)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d


class Quantizer(ABC):
    """
    Base class. Subclasses implement `quantize`, which returns a *new* model on the
    quantized grid, and expose `spec` describing exactly what they do.
    """
    name: str = "abstract"
    citation: str = ""

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False):
        assert granularity in ("per_channel", "per_tensor")
        assert 1 <= bits <= 16
        self.bits = bits
        self.granularity = granularity
        self.symmetric = symmetric

    @property
    def spec(self) -> QuantSpec:
        return QuantSpec(self.name, self.bits, self.granularity, self.symmetric, self.citation)

    @abstractmethod
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        """calib: (n_seq, seq_len+1) token tensor from the *train* split, or None."""

    # ---- helpers shared by subclasses -------------------------------------------------
    @staticmethod
    def _channel_view(w: t.Tensor, granularity: str) -> t.Tensor:
        """Return a 2D view (rows, channels) where each column shares one (scale, zero)."""
        if granularity == "per_tensor":
            return w.reshape(-1, 1)
        return w.reshape(-1, w.shape[-1])

    @staticmethod
    def grid(w2: t.Tensor, bits: int, symmetric: bool) -> tuple[t.Tensor, t.Tensor, int, int]:
        """Uniform affine grid per column of a (rows, channels) tensor.

        Returns (scale, zero, qmin, qmax) with scale/zero of shape (channels,).
        Asymmetric: q = round(w/s) + z in [0, 2^b - 1], z = round(-min/s).
        Symmetric : q = round(w/s)     in [-2^(b-1), 2^(b-1) - 1], z = 0.
        """
        eps = 1e-12
        if symmetric and bits < 2:
            raise ValueError("symmetric grid needs bits >= 2 (1-bit symmetric has q_max = 0)")
        if symmetric:
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            scale = w2.abs().amax(dim=0).clamp_min(eps) / qmax
            zero = t.zeros_like(scale)
        else:
            qmin, qmax = 0, 2 ** bits - 1
            wmin, wmax = w2.amin(dim=0), w2.amax(dim=0)
            scale = ((wmax - wmin) / qmax).clamp_min(eps)
            zero = t.round(-wmin / scale)
        return scale, zero, qmin, qmax

    @staticmethod
    def fake_quant(w2: t.Tensor, scale: t.Tensor, zero: t.Tensor, qmin: int, qmax: int) -> t.Tensor:
        q = t.clamp(t.round(w2 / scale) + zero, qmin, qmax)
        return (q - zero) * scale


# ---- accounting -------------------------------------------------------------------------

def count_params(model: t.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_bytes(model: t.nn.Module, bits: int | dict[str, int] | None, granularity: str = "per_channel") -> int:
    """
    bits: None -> everything fp32; int -> all WEIGHT_SUFFIXES tensors at that width;
          dict {param_name: bits} for mixed precision (missing names -> fp32).
    """
    total = 0
    for name, p in model.named_parameters():
        b = None
        if is_quantized_param(name):
            if isinstance(bits, int):
                b = bits
            elif isinstance(bits, dict):
                b = bits.get(name)
        if b is None:
            total += p.numel() * 4
        else:
            n_ch = 1 if granularity == "per_tensor" else p.shape[-1]
            total += (p.numel() * b + 7) // 8 + n_ch * 4
    return total
