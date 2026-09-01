"""
COMPOSITION (experimental namespace): FWSVD factors + quantization.

Not a new algorithm — a composition of two frozen reference methods, labeled
"FWSVD+RTN" (PTQ) and "FWSVD+QAT" (STE fine-tuning of the factors). Rules of §5 apply:
the FWSVD factorization and the RTN/STE grids are exactly the reference ones
(per-output-channel asymmetric, biases fp32); only the object being quantized changes
(the factors A (d_out, r), B (r, d_in) instead of dense W). W_E/W_pos stay dense matrices
and are RTN-quantized at the same width, as in the uniform quantization protocol.

Bytes = Σ_factors [numel·b/8 + channels·4] + Σ_emb [numel·b/8 + channels·4] + biases·4,
with channels = last dim (A: r, B: d_in, W_E/W_pos: d_model).
"""
from __future__ import annotations

import copy

import torch as t
from torch.nn.utils import parametrize
from transformer_lens import HookedTransformer

from compress.base import Quantizer
from compress.fwsvd import _LowRank


def _fq(w: t.Tensor, bits: int) -> t.Tensor:
    """RTN per-output-channel asymmetric fake-quant of a 2D tensor (reference grid)."""
    w2 = w.reshape(-1, w.shape[-1])
    s_, z_, lo, hi = Quantizer.grid(w2, bits, symmetric=False)
    return Quantizer.fake_quant(w2, s_, z_, lo, hi).reshape(w.shape)


class _LowRankSTE(t.nn.Module):
    """W = STE[Q(A)] @ STE[Q(B)], mapped back to the TL shape."""
    def __init__(self, A, B, name, shape, bits):
        super().__init__()
        self.A, self.B = t.nn.Parameter(A), t.nn.Parameter(B)
        self.pname, self.shape, self.bits = name, shape, bits

    def forward(self, w):
        from compress.fwsvd import _from2d
        A = self.A + (_fq(self.A, self.bits) - self.A).detach()
        B = self.B + (_fq(self.B, self.bits) - self.B).detach()
        return _from2d(self.pname, A @ B, self.shape)


def apply_lowrank_quant(model: HookedTransformer, factors: dict, bits: int, mode: str,
                        train_data: t.Tensor | None = None, epochs: int = 1, lr: float = 1e-4,
                        batch_size: int = 128, seed: int = 0) -> HookedTransformer:
    """mode='ptq': quantize cached factors, done. mode='qat': STE fine-tune factors first."""
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    wrapped = []
    for name, (A, B) in factors.items():
        mod = model.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        p = getattr(mod, attr)
        A, B = A.to(device), B.to(device)
        if mode == "ptq":
            parametrize.register_parametrization(mod, attr, _LowRank(_fq(A, bits), _fq(B, bits), name, p.shape))
        else:
            parametrize.register_parametrization(mod, attr, _LowRankSTE(A, B, name, p.shape, bits))
        wrapped.append((mod, attr))

    if mode == "qat":
        assert train_data is not None
        opt = t.optim.Adam(model.parameters(), lr=lr)
        g = t.Generator().manual_seed(seed)
        model.train()
        for _ in range(epochs):
            for idx in t.randperm(len(train_data), generator=g).split(batch_size):
                batch = train_data[idx].to(device)
                logits = model(batch[:, :-1])
                loss = t.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()

    for mod, attr in wrapped:
        parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
    # dense embeddings at the same width (uniform protocol)
    for name in ("embed.W_E", "pos_embed.W_pos"):
        p = dict(model.named_parameters())[name]
        p.data.copy_(_fq(p.data, bits))
    return model.eval()


def composed_bytes(model: HookedTransformer, factors: dict, bits: int) -> int:
    total = 0
    fact_names = set(factors)
    for name, p in model.named_parameters():
        if name in fact_names:
            A, B = factors[name]
            total += (A.numel() + B.numel()) * bits // 8 + (A.shape[-1] + B.shape[-1]) * 4
        elif name in ("embed.W_E", "pos_embed.W_pos"):
            total += p.numel() * bits // 8 + p.shape[-1] * 4
        else:
            total += p.numel() * 4     # biases fp32
    return total
