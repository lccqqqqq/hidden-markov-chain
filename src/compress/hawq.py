"""
HAWQ-V2 — Dong et al., NeurIPS 2020 (V1: ICCV 2019).

A bit-*allocation* policy, not a rounding rule:
  * Per quantized tensor W_i, estimate the Hessian trace tr(H_i) of the TRUE cross-entropy
    loss by Hutchinson: tr(H_i) = E_v[vᵀ H_i v], v Rademacher, with exact Hessian-vector
    products through autograd (double backward). This is where the loss landscape enters —
    the global loss, not a layerwise proxy.
  * Sensitivity of assigning b bits to tensor i:  Ω_i(b) = (tr(H_i)/n_i) · ‖Q_b(W_i) − W_i‖²
    with Q_b the base quantizer's grid (RTN per-output-channel here).
  * Choose bits b_i ∈ choices minimising Σ_i Ω_i(b_i) subject to Σ_i bytes_i(b_i) ≤ budget.
    The paper phrases this as an ILP / Pareto frontier; with ~27 tensors we solve the
    knapsack exactly by dynamic programming (8-byte resolution).
  * Apply `base` (RTN or GPTQ) with the resulting {param_name: bits} map.

Deviations from the paper (named, logged):
  * no_finetune=True: the paper follows allocation with QAT fine-tuning; we evaluate the
    allocation itself (fine-tuning is the separate STE-QAT experiment).
  * Negative trace estimates (possible away from an exact minimum) are clamped to 0 for
    the allocation and reported.
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param


class HAWQv2(Quantizer):
    name = "HAWQ-V2"
    citation = "Dong et al., NeurIPS 2020"

    def __init__(self, byte_budget: int, base: Quantizer, choices=(2, 4, 8),
                 n_probes: int = 64, seed: int = 0):
        super().__init__(bits=max(choices), granularity=base.granularity, symmetric=base.symmetric)
        self.byte_budget, self.base, self.choices = byte_budget, base, tuple(sorted(choices))
        self.n_probes, self.seed = n_probes, seed
        self.traces: dict[str, float] = {}
        self.allocation: dict[str, int] = {}

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(byte_budget=self.byte_budget, base=self.base.name, choices=self.choices,
                       n_probes=self.n_probes, no_finetune=True)
        return s

    # ---- Hutchinson block traces of the true loss Hessian --------------------------------
    def hessian_traces(self, model: HookedTransformer, calib: t.Tensor) -> dict[str, float]:
        device = next(model.parameters()).device
        inputs, targets = calib[:, :-1].to(device), calib[:, 1:].to(device)
        names, params = zip(*[(n, p) for n, p in model.named_parameters() if is_quantized_param(n)])
        logits = model(inputs)
        loss = t.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        grads = t.autograd.grad(loss, params, create_graph=True)
        g = t.Generator(device="cpu").manual_seed(self.seed)
        tr = t.zeros(len(params))
        for _ in range(self.n_probes):
            vs = [(t.randint(0, 2, p.shape, generator=g).float().to(device) * 2 - 1) for p in params]
            gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
            Hvs = t.autograd.grad(gv, params, retain_graph=True)
            tr += t.stack([(hv * v).sum() for hv, v in zip(Hvs, vs)]).cpu()
        tr /= self.n_probes
        return dict(zip(names, tr.tolist()))

    # ---- sensitivity + exact knapsack ----------------------------------------------------
    def _tensor_bytes(self, p: t.Tensor, name: str, bits: int) -> int:
        n_ch = 1 if self.granularity == "per_tensor" else self._channel_view(p, self.granularity, name).shape[1]
        return (p.numel() * bits + 7) // 8 + n_ch * 4

    def allocate(self, model: HookedTransformer, calib: t.Tensor) -> dict[str, int]:
        self.traces = self.hessian_traces(model, calib)
        named = [(n, p) for n, p in model.named_parameters() if is_quantized_param(n)]
        bias_bytes = sum(p.numel() * 4 for n, p in model.named_parameters() if not is_quantized_param(n))
        budget_q = self.byte_budget - bias_bytes
        res = 8  # DP resolution in bytes

        omega, cost = [], []   # per tensor: {bits: value}, {bits: cost units}
        for n, p in named:
            trace = max(self.traces[n], 0.0)
            row_o, row_c = {}, {}
            for b in self.choices:
                w2 = self._channel_view(p.data, self.granularity, n)
                s_, z_, lo, hi = self.grid(w2, b, self.symmetric)
                err = (self.fake_quant(w2, s_, z_, lo, hi) - w2).pow(2).sum().item()
                row_o[b] = trace / p.numel() * err
                row_c[b] = -(-self._tensor_bytes(p, n, b) // res)   # ceil
            omega.append(row_o); cost.append(row_c)

        cap = budget_q // res
        NEG = float("inf")
        dp = [NEG] * (cap + 1); dp[0] = 0.0
        choice = [[None] * (cap + 1) for _ in named]
        for i in range(len(named)):
            ndp = [NEG] * (cap + 1)
            for c in range(cap + 1):
                if dp[c] == NEG:
                    continue
                for b in self.choices:
                    nc = c + cost[i][b]
                    if nc <= cap and dp[c] + omega[i][b] < ndp[nc]:
                        ndp[nc] = dp[c] + omega[i][b]
                        choice[i][nc] = (b, c)
            dp = ndp
        best_c = min(range(cap + 1), key=lambda c: dp[c])
        assert dp[best_c] < NEG, "byte budget infeasible even at the smallest bit-width"
        alloc = {}
        c = best_c
        for i in range(len(named) - 1, -1, -1):
            b, c = choice[i][c]
            alloc[named[i][0]] = b
        self.allocation = alloc
        return alloc

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None, "HAWQ-V2 needs calibration sequences (Hessian probes + base method)"
        self.allocate(model, calib)
        self.base.bits_map = self.allocation
        try:
            return self.base.quantize(model, calib)
        finally:
            self.base.bits_map = None
