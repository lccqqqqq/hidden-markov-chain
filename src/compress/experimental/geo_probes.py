"""
E1 / E2 — structured finite-radius profiles and pairwise gate interactions
(notes/compression/singular_geometry_evaluation.md §4, Tier 1).

Every structured coordinate is a scalar gate alpha multiplying one functional component.
Each gate is *exactly* a straight line in parameter space, so the same perturbation can be
scored by the exact quadratic model (gradient + Hessian-vector product along the line) and
compared with the measured finite-radius loss:

  gate           implemented as                       == parameter-space direction d
  qk    (l,h)    attn scores of head h  * alpha       (W_Q[h], b_Q[h])      (query scaled)
  W_Q   (l,h)    parameter edit                       W_Q[h]                (weight only)
  W_K   (l,h)    parameter edit                       W_K[h]
  head  (l,h)    hook_z[..., h, :]      * alpha       W_O[h]
  attn  (l)      hook_attn_out          * alpha       (W_O, b_O)
  mlp   (l)      hook_mlp_out           * alpha       (W_out, b_out)
  ngrp  (l,g)    hook_post[..., idx_g]  * alpha       W_out[idx_g, :]   (neuron group g)

Loss L(alpha) is measured on a fixed calibration set (train split) with BOTH the one-hot
NLL and the exact-teacher KL (teacher.py). The quadratic prediction is
  dL_quad(alpha) = (alpha-1) g.d + 1/2 (alpha-1)^2 d.Hd
with g, H of the exact-teacher KL on the same calibration set.

Nothing here modifies reference quantizers.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.teacher import evaluate_teacher, kl_loss


@dataclass(frozen=True)
class Gate:
    kind: str          # qk | W_Q | W_K | head | attn | mlp | ngrp
    layer: int
    idx: int = -1      # head or group index; -1 for sublayer gates

    @property
    def label(self) -> str:
        return f"{self.kind}.L{self.layer}" + (f".{self.idx}" if self.idx >= 0 else "")


class GateModel:
    """Holds one model plus neuron groupings; applies gates via hooks / param edits."""

    def __init__(self, model: HookedTransformer, calib: t.Tensor, group_size: int = 32):
        self.model = model
        self.cfg = model.cfg
        self.calib = calib
        self.group_size = group_size
        self.groups = self._neuron_groups()

    # ---- neuron groups by activation variance (descending) -----------------------------
    @t.no_grad()
    def _neuron_groups(self) -> dict[int, list[t.Tensor]]:
        store = {}
        names = [f"blocks.{l}.mlp.hook_post" for l in range(self.cfg.n_layers)]
        self.model.run_with_hooks(
            self.calib[:2048, :-1], return_type=None,
            fwd_hooks=[(n, lambda a, hook: store.__setitem__(hook.name, a.reshape(-1, a.shape[-1]).var(0)))
                       for n in names])
        groups = {}
        for l in range(self.cfg.n_layers):
            order = t.argsort(store[names[l]], descending=True)
            groups[l] = list(order.split(self.group_size))
        return groups

    def all_gates(self) -> list[Gate]:
        L, H = self.cfg.n_layers, self.cfg.n_heads
        gates = []
        for l in range(L):
            gates += [Gate("qk", l, h) for h in range(H)]
            gates += [Gate("W_Q", l, h) for h in range(H)]
            gates += [Gate("W_K", l, h) for h in range(H)]
            gates += [Gate("head", l, h) for h in range(H)]
            gates += [Gate("attn", l), Gate("mlp", l)]
            gates += [Gate("ngrp", l, g) for g in range(len(self.groups[l]))]
        return gates

    # ---- hooks -------------------------------------------------------------------------
    def _hook(self, gate: Gate, alpha: float):
        l, i = gate.layer, gate.idx
        if gate.kind == "qk":
            def f(scores, hook):            # (b, h, q, k); masked entries are very negative
                s = scores[:, i]
                scores[:, i] = t.where(s > -1e4, s * alpha, s)
                return scores
            return (f"blocks.{l}.attn.hook_attn_scores", f)
        if gate.kind == "head":
            def f(z, hook):                 # (b, p, h, d)
                z[:, :, i] = z[:, :, i] * alpha
                return z
            return (f"blocks.{l}.attn.hook_z", f)
        if gate.kind == "attn":
            return (f"blocks.{l}.hook_attn_out", lambda a, hook: a * alpha)
        if gate.kind == "mlp":
            return (f"blocks.{l}.hook_mlp_out", lambda a, hook: a * alpha)
        if gate.kind == "ngrp":
            idx = self.groups[l][i]
            def f(a, hook):
                # clone: hook_post is the ReLU output, which ReLU's backward needs intact
                # (in-place edits break autograd when the gate is used during training)
                a = a.clone()
                a[..., idx] = a[..., idx] * alpha
                return a
            return (f"blocks.{l}.mlp.hook_post", f)
        raise ValueError(gate.kind)

    @contextmanager
    def applied(self, settings: list[tuple[Gate, float]]):
        """Apply several gates at once. Param-edit gates (W_Q/W_K) are restored on exit."""
        hooks, restore = [], []
        for gate, alpha in settings:
            if gate.kind in ("W_Q", "W_K"):
                w = getattr(self.model.blocks[gate.layer].attn, gate.kind)
                restore.append((w, gate.idx, w.data[gate.idx].clone()))
                w.data[gate.idx] *= alpha
            else:
                hooks.append(self._hook(gate, alpha))
        try:
            for name, f in hooks:
                self.model.add_hook(name, f)
            yield
        finally:
            self.model.reset_hooks()
            for w, idx, orig in restore:
                w.data[idx] = orig

    def measure(self, settings: list[tuple[Gate, float]], tokens: t.Tensor, probs: t.Tensor) -> dict:
        with self.applied(settings):
            return evaluate_teacher(self.model, tokens, probs)

    # ---- parameter-space direction of each gate ------------------------------------------
    def direction(self, gate: Gate) -> dict[str, t.Tensor]:
        """d such that gate(alpha) == params + (alpha-1) d."""
        blk = self.model.blocks[gate.layer]
        l, i = gate.layer, gate.idx
        d = {}
        if gate.kind == "qk":
            d[f"blocks.{l}.attn.W_Q"] = t.zeros_like(blk.attn.W_Q); d[f"blocks.{l}.attn.W_Q"][i] = blk.attn.W_Q.data[i]
            d[f"blocks.{l}.attn.b_Q"] = t.zeros_like(blk.attn.b_Q); d[f"blocks.{l}.attn.b_Q"][i] = blk.attn.b_Q.data[i]
        elif gate.kind in ("W_Q", "W_K"):
            w = getattr(blk.attn, gate.kind)
            d[f"blocks.{l}.attn.{gate.kind}"] = t.zeros_like(w); d[f"blocks.{l}.attn.{gate.kind}"][i] = w.data[i]
        elif gate.kind == "head":
            d[f"blocks.{l}.attn.W_O"] = t.zeros_like(blk.attn.W_O); d[f"blocks.{l}.attn.W_O"][i] = blk.attn.W_O.data[i]
        elif gate.kind == "attn":
            d[f"blocks.{l}.attn.W_O"] = blk.attn.W_O.data.clone()
            d[f"blocks.{l}.attn.b_O"] = blk.attn.b_O.data.clone()
        elif gate.kind == "mlp":
            d[f"blocks.{l}.mlp.W_out"] = blk.mlp.W_out.data.clone()
            d[f"blocks.{l}.mlp.b_out"] = blk.mlp.b_out.data.clone()
        elif gate.kind == "ngrp":
            idx = self.groups[l][i]
            d[f"blocks.{l}.mlp.W_out"] = t.zeros_like(blk.mlp.W_out); d[f"blocks.{l}.mlp.W_out"][idx] = blk.mlp.W_out.data[idx]
        return d


class QuadraticModel:
    """Exact gradient and Hessian-vector products of the exact-teacher KL on a calibration set."""

    def __init__(self, model: HookedTransformer, tokens: t.Tensor, probs: t.Tensor):
        self.model = model
        self.names, self.params = zip(*model.named_parameters())
        loss = kl_loss(model, tokens, probs)
        self.loss0 = loss.item()
        self.grads = t.autograd.grad(loss, self.params, create_graph=True)

    def _vec(self, d: dict[str, t.Tensor]) -> list[t.Tensor]:
        return [d.get(n, None) for n in self.names]

    def hvp(self, d: dict[str, t.Tensor]) -> dict[str, t.Tensor]:
        vs = self._vec(d)
        gv = sum((g * v).sum() for g, v in zip(self.grads, vs) if v is not None)
        Hv = t.autograd.grad(gv, self.params, retain_graph=True)
        return dict(zip(self.names, [h.detach() for h in Hv]))

    def gdot(self, d: dict[str, t.Tensor]) -> float:
        return float(sum((g.detach() * v).sum() for g, v in zip(self.grads, self._vec(d)) if v is not None))

    @staticmethod
    def dot(a: dict[str, t.Tensor], b: dict[str, t.Tensor]) -> float:
        return float(sum((a[k] * b[k]).sum() for k in b if k in a))
