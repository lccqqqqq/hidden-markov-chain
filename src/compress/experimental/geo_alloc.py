"""
E4 — measured-sensitivity mixed precision with an extended menu
(notes/compression/singular_geometry_evaluation.md §4, Tier 2).

Two changes relative to the HAWQ-V2 reference (compress/hawq.py, left untouched):

  1. Sensitivity of tensor i at width b is MEASURED, not predicted from curvature:
         S_i(b) = L( GPTQ applied to tensor i alone at b bits, all else fp32 ) - L(w*)
     with L the exact-teacher KL on a calibration set (train split). This is the
     finite-scale, "compression-shaped perturbation" sensitivity of the docx §10.2 in its
     most direct form. The trace-based Omega_i(b) of HAWQ-V2 is also computed on the same
     menu (option sensitivity="trace") so the two signals can be compared with everything
     else held fixed.
  2. The menu extends below 2 bits: 0 = delete (tensor set to zero, bias kept),
     TERNARY = absmean {-s, 0, +s} per output channel (BitNet-b1.58-style rounding, but
     post-training with OBS compensation, i.e. "ternary GPTQ"), and 1 = 1-bit asymmetric.

Rounding backend: ExtGPTQ = the reference GPTQ per-matrix routine, extended only by the
two extra rounding rules above. Integer widths >= 1 run the reference code path unchanged.
Bytes: delete = 0; ternary = ceil(numel*log2(3)/8) + 2 B per channel (one fp16 scale);
otherwise the shared accounting of compress/base.py.
"""
from __future__ import annotations

import copy
import math

import torch as t
from torch.nn.utils import parametrize
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param
from compress.gptq import GPTQ
from compress.qat import STEQAT

TERNARY = -3          # sentinel width in bits_map
DELETE = 0
MENU_DEFAULT = (DELETE, 1, TERNARY, 2, 3, 4, 5, 6, 8)


def width_label(b: int) -> str:
    return {DELETE: "del", TERNARY: "T", 1: "1"}.get(b, str(b))


def bits_per_weight(b: int) -> float:
    return {DELETE: 0.0, TERNARY: math.log2(3)}.get(b, float(b))


def ternary_fq(w2: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """absmean ternary per column of (rows, channels): returns (fake-quant, scale)."""
    s = w2.abs().mean(dim=0).clamp_min(1e-12)
    return t.clamp(t.round(w2 / s), -1, 1) * s, s


class ExtGPTQ(GPTQ):
    name = "GPTQ+ext"

    def _gptq_matrix(self, W: t.Tensor, H: t.Tensor, bits: int | None = None) -> t.Tensor:
        b = self.bits if bits is None else bits
        if b == DELETE:
            return t.zeros_like(W)
        if b != TERNARY:
            return super()._gptq_matrix(W, H, bits)
        # ternary with OBS compensation (same loop as the reference, different rounding)
        W = W.clone(); H = H.clone()
        d_out, d_in = W.shape
        dead = t.diag(H) == 0
        H[dead, dead] = 1.0; W[:, dead] = 0.0
        s = W.abs().mean(dim=1).clamp_min(1e-12)          # per output row, from original W
        H += self.damp * t.diag(H).mean() * t.eye(d_in, device=H.device)
        Hinv = t.linalg.cholesky(t.cholesky_inverse(t.linalg.cholesky(H)), upper=True)
        for j in range(d_in):
            w = W[:, j]
            wq = t.clamp(t.round(w / s), -1, 1) * s
            err = (w - wq) / Hinv[j, j]
            W[:, j] = wq
            if j + 1 < d_in:
                W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:][None, :]
        return W

    @t.no_grad()
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        # embeddings with special widths are handled here (their GPTQ == RTN anyway);
        # the reference then sees them at 16 bits, which is lossless to ~1e-5 relative.
        bm = dict(self.bits_map or {})
        model = copy.deepcopy(model)
        for name in ("embed.W_E", "pos_embed.W_pos"):
            b = bm.get(name, self.bits)
            if b in (DELETE, TERNARY):
                p = dict(model.named_parameters())[name]
                w2 = self._channel_view(p.data, self.granularity, name)
                fq = t.zeros_like(w2) if b == DELETE else ternary_fq(w2)[0]
                p.data.copy_(self._from_channel_view(fq, p.shape, name))
                bm[name] = 16
        old = self.bits_map
        self.bits_map = bm if old is not None else None
        try:
            return super().quantize(model, calib)
        finally:
            self.bits_map = old


def count_bytes_ext(model: t.nn.Module, bits_map: dict[str, int]) -> int:
    total = 0
    for name, p in model.named_parameters():
        b = bits_map.get(name) if is_quantized_param(name) else None
        if b is None:
            total += p.numel() * 4
        elif b == DELETE:
            continue
        else:
            n_ch = Quantizer._channel_view(p, "per_channel", name).shape[1]
            total += math.ceil(p.numel() * bits_per_weight(b) / 8) + n_ch * (2 if b == TERNARY else 4)
    return total


def knapsack(names: list[str], value: dict[str, dict[int, float]], cost: dict[str, dict[int, int]],
             budget: int, menu: tuple[int, ...], res: int = 8) -> dict[str, int]:
    """min sum_i value[i][b_i]  s.t.  sum_i cost[i][b_i] <= budget (exact DP at `res`-byte resolution)."""
    cap = budget // res
    INF = float("inf")
    dp = [INF] * (cap + 1); dp[0] = 0.0
    choice = [[None] * (cap + 1) for _ in names]
    for i, n in enumerate(names):
        ndp = [INF] * (cap + 1)
        for c in range(cap + 1):
            if dp[c] == INF:
                continue
            for b in menu:
                nc = c + -(-cost[n][b] // res)
                if nc <= cap and dp[c] + value[n][b] < ndp[nc]:
                    ndp[nc] = dp[c] + value[n][b]
                    choice[i][nc] = (b, c)
        dp = ndp
    best = min(range(cap + 1), key=lambda c: dp[c])
    assert dp[best] < INF, "budget infeasible"
    alloc, c = {}, best
    for i in range(len(names) - 1, -1, -1):
        b, c = choice[i][c]
        alloc[names[i]] = b
    return alloc


# ---- QAT with the extended widths -------------------------------------------------------------

class _ExtSTE(t.nn.Module):
    def __init__(self, q: "ExtSTEQAT", name: str):
        super().__init__()
        self.q, self.pname = q, name

    def forward(self, w: t.Tensor) -> t.Tensor:
        q = self.q
        b = q._bits(self.pname)
        if b == DELETE:
            return t.zeros_like(w)
        w2 = q._channel_view(w, q.granularity, self.pname)
        with t.no_grad():
            if b == TERNARY:
                fq = ternary_fq(w2)[0]
            else:
                scale, zero, qmin, qmax = q.grid(w2, b, q.symmetric)
                fq = q.fake_quant(w2, scale, zero, qmin, qmax)
        return q._from_channel_view(w2 + (fq - w2).detach(), w.shape, self.pname)


class ExtSTEQAT(STEQAT):
    """STE-QAT (reference loop settings) with the extended widths and optional exact-teacher
    soft targets. The training loop is the reference's, duplicated because the reference
    hard-codes its parametrization class."""
    name = "STE-QAT+ext"

    def __init__(self, *a, soft_targets=None, process=None, **k):
        super().__init__(*a, **k)
        self.soft_targets, self.process = soft_targets, process

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        from compress.experimental.teacher import kl_loss, teacher_probs
        assert calib is not None and len(calib) > 1000
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        wrapped = []
        for name, _ in list(model.named_parameters()):
            if not is_quantized_param(name):
                continue
            mod = model.get_submodule(name.rsplit(".", 1)[0]); attr = name.rsplit(".", 1)[1]
            parametrize.register_parametrization(mod, attr, _ExtSTE(self, name))
            wrapped.append((mod, attr))
        opt = t.optim.Adam(model.parameters(), lr=self.lr)
        g = t.Generator().manual_seed(self.seed)
        model.train(); step = 0
        for _ in range(self.epochs):
            for idx in t.randperm(len(calib), generator=g).split(self.batch_size):
                batch = calib[idx].to(device)
                if self.soft_targets:
                    loss = kl_loss(model, batch, teacher_probs(self.process, batch))
                else:
                    logits = model(batch[:, :-1])
                    loss = t.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if self.max_steps and step >= self.max_steps:
                    break
            else:
                continue
            break
        for mod, attr in wrapped:
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        # deleted tensors: make sure they are exactly zero after baking
        for name, p in model.named_parameters():
            if is_quantized_param(name) and self._bits(name) == DELETE:
                p.data.zero_()
        return model.eval()
