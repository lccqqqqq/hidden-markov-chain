"""
E7 — global-Hessian GPTQ (notes/compression/singular_geometry_evaluation.md §4, Tier 4).

Reference GPTQ compensates each rounding error only *within the same matrix*, through the
layerwise proxy H = 2XXᵀ. Here the reference per-matrix routine (`GPTQ._gptq_matrix`, the
same grids, damping, column order, sequential activation propagation) is kept unchanged,
and after each of the 27 tensors is quantized one damped Newton step on the exact-teacher
KL (teacher.py) is taken over everything that is still fp32:

    free  = not-yet-quantized WEIGHT_SUFFIXES tensors  +  all biases
    solve (H_ff + λ I) δ = −g_f   by conjugate gradient with exact Hessian-vector products
    λ     = damp_newton · mean(diag H_ff),  mean(diag) = (1/n) E_v[vᵀ H v] (Hutchinson, Rademacher)

Quantized tensors are frozen on their lattice. Guard: if the step raises the calibration
KL it is halved up to 3 times, then skipped (count reported in `newton_skips`). Newton
calibration sequences (n_newton_calib, train split) are disjoint from the 128 GPTQ
calibration sequences. The final model has every WEIGHT_SUFFIXES tensor on its grid and
biases fp32, exactly like the reference — only the values of later-quantized tensors and
biases differ. This is a *new method* (survey §5 honesty principle), never labelled GPTQ.
"""
from __future__ import annotations

import copy

import torch as t
from transformer_lens import HookedTransformer

from compress.base import is_quantized_param
from compress.experimental.teacher import kl_loss
from compress.gptq import GPTQ


class GlobalGPTQ(GPTQ):
    name = "GlobalGPTQ"
    citation = "experimental (this repo); rounding = GPTQ, Frantar et al. 2023"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 n_calib: int = 128, damp: float = 0.01, act_order: bool = False,
                 damp_newton: float = 1e-2, cg_iters: int = 20, n_newton_calib: int = 1024,
                 n_hutch: int = 4, seed: int = 0):
        super().__init__(bits, granularity, symmetric, n_calib, damp, act_order)
        self.damp_newton, self.cg_iters, self.n_newton_calib = damp_newton, cg_iters, n_newton_calib
        self.n_hutch, self.seed = n_hutch, seed
        self.newton_calib: t.Tensor | None = None
        self.newton_probs: t.Tensor | None = None
        self.newton_skips = 0
        self.newton_log: list[dict] = []

    @property
    def spec(self):
        s = super().spec
        s.extra.update(damp_newton=self.damp_newton, cg_iters=self.cg_iters,
                       n_newton_calib=self.n_newton_calib, newton_skips=self.newton_skips)
        return s

    # ---- one damped Newton step on the free parameters --------------------------------------
    def _newton_step(self, model: HookedTransformer, free_names: set[str], tag: str) -> None:
        names, params = zip(*[(n, p) for n, p in model.named_parameters() if n in free_names])
        loss = kl_loss(model, self.newton_calib, self.newton_probs)
        loss0 = loss.item()
        grads = t.autograd.grad(loss, params, create_graph=True)
        g = t.cat([x.reshape(-1) for x in grads]).detach()
        dim = g.numel()

        def hvp(v: t.Tensor) -> t.Tensor:
            vs, off = [], 0
            for p in params:
                vs.append(v[off:off + p.numel()].view_as(p)); off += p.numel()
            gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
            Hv = t.autograd.grad(gv, params, retain_graph=True)
            return t.cat([h.reshape(-1) for h in Hv]).detach()

        gen = t.Generator().manual_seed(self.seed)
        tr = 0.0
        for _ in range(self.n_hutch):
            v = t.randint(0, 2, (dim,), generator=gen).float() * 2 - 1
            tr += float(v @ hvp(v))
        lam = self.damp_newton * max(tr / self.n_hutch, 0.0) / dim
        lam = max(lam, 1e-12)

        # CG on (H + lam I) d = -g
        x = t.zeros(dim); r = -g.clone(); p_ = r.clone(); rs = float(r @ r)
        for _ in range(self.cg_iters):
            Ap = hvp(p_) + lam * p_
            pAp = float(p_ @ Ap)
            if pAp <= 0:            # negative curvature: stop with the current iterate
                break
            a = rs / pAp
            x += a * p_; r -= a * Ap
            rs_new = float(r @ r)
            if rs_new < 1e-20:
                break
            p_ = r + (rs_new / rs) * p_; rs = rs_new
        del grads, loss

        # apply with halving guard
        scale, applied = 1.0, False
        with t.no_grad():
            for _ in range(4):
                off = 0
                for p in params:
                    p.add_(scale * x[off:off + p.numel()].view_as(p)); off += p.numel()
                new = kl_loss(model, self.newton_calib, self.newton_probs).item()
                if new < loss0:
                    applied = True
                    break
                off = 0
                for p in params:
                    p.sub_(scale * x[off:off + p.numel()].view_as(p)); off += p.numel()
                scale *= 0.5
        if not applied:
            self.newton_skips += 1
        self.newton_log.append(dict(after=tag, kl_before=loss0, kl_after=new if applied else loss0,
                                    step_norm=float(x.norm()) * (scale if applied else 0.0),
                                    lam=lam, applied=applied, n_free=dim))

    # ---- traversal: reference order, Newton after each tensor -------------------------------
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None, "GlobalGPTQ needs GPTQ calibration sequences (train split)"
        assert self.newton_calib is not None and self.newton_probs is not None, \
            "set .newton_calib (tokens) and .newton_probs (teacher_probs) before quantize()"
        model = copy.deepcopy(model)
        for p in model.parameters():
            p.requires_grad_(True)
        device = next(model.parameters()).device
        tokens = calib[:, :-1].to(device)
        self.newton_skips, self.newton_log = 0, []
        free = {n for n, _ in model.named_parameters()}     # everything starts free

        def acts(hook_name: str) -> t.Tensor:
            store = {}
            with t.no_grad():
                model.run_with_hooks(tokens, return_type=None,
                                     fwd_hooks=[(hook_name, lambda a, hook: store.update(x=a.detach()))])
            return store["x"]

        def done(name: str):
            free.discard(name)
            self._newton_step(model, free, name)

        with t.no_grad():
            for name in ("embed.W_E", "pos_embed.W_pos"):
                p = dict(model.named_parameters())[name]
                w2 = self._channel_view(p.data, self.granularity, name)
                s_, z_, lo, hi = self.grid(w2, self._bits(name), self.symmetric)
                p.data.copy_(self._from_channel_view(self.fake_quant(w2, s_, z_, lo, hi), p.shape, name))
        done("embed.W_E"); done("pos_embed.W_pos")      # W_pos step: W_E already frozen

        n_layers = model.cfg.n_layers
        for l in range(n_layers):
            blk = model.blocks[l]
            X = acts(f"blocks.{l}.hook_resid_pre")
            H = self._hessian(X)
            for wname in ("W_Q", "W_K", "W_V"):
                with t.no_grad():
                    w = getattr(blk.attn, wname)
                    W2 = w.data.permute(0, 2, 1).reshape(-1, w.shape[1])
                    W2 = self._gptq_matrix(W2, H, self._bits(f"blocks.{l}.attn.{wname}"))
                    w.data.copy_(W2.reshape(w.shape[0], w.shape[2], w.shape[1]).permute(0, 2, 1))
                done(f"blocks.{l}.attn.{wname}")
            Z = acts(f"blocks.{l}.attn.hook_z")
            H = self._hessian(Z.reshape(-1, Z.shape[-2] * Z.shape[-1]))
            with t.no_grad():
                w = blk.attn.W_O
                W2 = w.data.reshape(-1, w.shape[-1]).T
                w.data.copy_(self._gptq_matrix(W2, H, self._bits(f"blocks.{l}.attn.W_O")).T.reshape(w.shape))
            done(f"blocks.{l}.attn.W_O")
            H = self._hessian(acts(f"blocks.{l}.hook_resid_mid"))
            with t.no_grad():
                w = blk.mlp.W_in
                w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits(f"blocks.{l}.mlp.W_in")).T)
            done(f"blocks.{l}.mlp.W_in")
            H = self._hessian(acts(f"blocks.{l}.mlp.hook_post"))
            with t.no_grad():
                w = blk.mlp.W_out
                w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits(f"blocks.{l}.mlp.W_out")).T)
            done(f"blocks.{l}.mlp.W_out")

        H = self._hessian(acts(f"blocks.{n_layers - 1}.hook_resid_post"))
        with t.no_grad():
            w = model.unembed.W_U
            w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits("unembed.W_U")).T)
        done("unembed.W_U")                              # final step: biases only
        assert all(not is_quantized_param(n) for n in free), free
        return model.eval()
