"""
Quantization-aware training.

STEQAT — straight-through estimator (Bengio, Léonard & Courville 2013). Implemented.
  * Every quantized weight is wrapped in a torch parametrization whose forward returns
        w_eff = w + (fake_quant(w) - w).detach()
    so the forward pass sees the b-bit lattice value while gradients flow to the
    underlying fp32 weight. Grid (min/max per output channel, asymmetric by default) is
    recomputed from the current w each step under no_grad (i.e. scale/zero are constants
    to the optimizer — the fixed-grid STE variant; learned scales are LSQ, kept separate).
  * Fine-tunes from the saved checkpoint on the TRAIN split (`calib` argument = the
    fine-tuning data here), Adam, no weight decay, no scheduler, fixed seed; then the
    parametrization is removed with leave_parametrized=True, baking the lattice weights.
  * Supports `bits_map` for mixed precision (HAWQ-V2 allocations), like RTN/GPTQ.
  * Note: this deliberately does NOT reuse train.py's loop (wandb/checkpoints/scheduler
    are not wanted here); the ~25-line loop is duplicated instead of refactoring a script
    that is in active use.
  * Ternary/2-bit via this route is "ternary STE-QAT", not BitNet (BitNet trains from
    scratch with absmean scaling and 8-bit activations).

LSQ — Esser et al., ICLR 2020 ("Learned Step Size Quantization"). Implemented.
  * Weights quantized symmetrically, w_q = clamp(round(w/s), Qn, Qp) * s with
    Qn = -2^(b-1), Qp = 2^(b-1)-1 (paper's weight quantizer, no zero point).
  * The step s is a LEARNED parameter per output channel, initialised
    s0 = 2 * mean|w| / sqrt(Qp), with its gradient scaled by g = 1/sqrt(N_w * Qp)
    (grad-scale trick: value unchanged, gradient multiplied by g).
  * Round/clamp pass gradients straight through to w; s receives the paper's STE gradient
    through the same expression. Same fine-tuning loop settings as STE-QAT.
"""
from __future__ import annotations

import copy

import torch as t
from torch.nn.utils import parametrize
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param


class _STE(t.nn.Module):
    def __init__(self, q: "STEQAT", name: str):
        super().__init__()
        self.q, self.pname = q, name

    def forward(self, w: t.Tensor) -> t.Tensor:
        q = self.q
        w2 = q._channel_view(w, q.granularity, self.pname)
        with t.no_grad():
            scale, zero, qmin, qmax = q.grid(w2, q._bits(self.pname), q.symmetric)
        fq = q.fake_quant(w2, scale, zero, qmin, qmax)
        return q._from_channel_view(w2 + (fq - w2).detach(), w.shape, self.pname)


class STEQAT(Quantizer):
    name = "STE-QAT"
    citation = "Bengio et al. 2013 (STE)"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 epochs: int = 1, lr: float = 1e-4, batch_size: int = 128, seed: int = 0,
                 max_steps: int | None = None):
        super().__init__(bits, granularity, symmetric)
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.max_steps = max_steps

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
                       max_steps=self.max_steps, mixed=self.bits_map is not None)
        return s

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        """`calib` = fine-tuning data (the train split), shape (n, seq_len+1)."""
        assert calib is not None and len(calib) > 1000, "STE-QAT fine-tunes: pass the train split"
        model = copy.deepcopy(model)
        device = next(model.parameters()).device

        wrapped = []
        for name, _ in list(model.named_parameters()):
            if not is_quantized_param(name):
                continue
            mod = model.get_submodule(name.rsplit(".", 1)[0])
            attr = name.rsplit(".", 1)[1]
            parametrize.register_parametrization(mod, attr, _STE(self, name))
            wrapped.append((mod, attr))

        opt = t.optim.Adam(model.parameters(), lr=self.lr)
        g = t.Generator().manual_seed(self.seed)
        model.train()
        step = 0
        for _ in range(self.epochs):
            for idx in t.randperm(len(calib), generator=g).split(self.batch_size):
                batch = calib[idx].to(device)
                logits = model(batch[:, :-1])
                loss = t.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
                if self.max_steps and step >= self.max_steps:
                    break
            else:
                continue
            break

        for mod, attr in wrapped:
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        return model.eval()


def _grad_scale(x: t.Tensor, g: float) -> t.Tensor:
    return x * g + (x * (1.0 - g)).detach()


def _round_pass(x: t.Tensor) -> t.Tensor:
    return (x.round() - x).detach() + x


class _LSQParam(t.nn.Module):
    def __init__(self, q: "LSQ", name: str, w: t.Tensor):
        super().__init__()
        self.q, self.pname = q, name
        w2 = q._channel_view(w.detach(), q.granularity, name)
        Qp = 2 ** (q._bits(name) - 1) - 1
        self.s = t.nn.Parameter(2.0 * w2.abs().mean(dim=0) / (Qp ** 0.5))
        self.g = 1.0 / (w2.shape[0] * Qp) ** 0.5

    def forward(self, w: t.Tensor) -> t.Tensor:
        q = self.q
        bits = q._bits(self.pname)
        Qn, Qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        w2 = q._channel_view(w, q.granularity, self.pname)
        s = _grad_scale(self.s.abs().clamp_min(1e-12), self.g)
        wq = _round_pass(t.clamp(w2 / s, Qn, Qp)) * s
        return q._from_channel_view(wq, w.shape, self.pname)


class LSQ(Quantizer):
    name = "LSQ"
    citation = "Esser et al., ICLR 2020"

    def __init__(self, bits: int, granularity: str = "per_channel", epochs: int = 1,
                 lr: float = 1e-4, batch_size: int = 128, seed: int = 0):
        super().__init__(bits, granularity, symmetric=True)
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)
        return s

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None and len(calib) > 1000, "LSQ fine-tunes: pass the train split"
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        wrapped = []
        for name, p in list(model.named_parameters()):
            if not is_quantized_param(name):
                continue
            mod = model.get_submodule(name.rsplit(".", 1)[0])
            attr = name.rsplit(".", 1)[1]
            parametrize.register_parametrization(mod, attr, _LSQParam(self, name, p))
            wrapped.append((mod, attr))
        opt = t.optim.Adam(model.parameters(), lr=self.lr)
        g = t.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            for idx in t.randperm(len(calib), generator=g).split(self.batch_size):
                batch = calib[idx].to(device)
                logits = model(batch[:, :-1])
                loss = t.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
        for mod, attr in wrapped:
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        return model.eval()
