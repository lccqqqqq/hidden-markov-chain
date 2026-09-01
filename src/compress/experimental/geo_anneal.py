"""
R1 — adiabatic annealing with retraining (notes/compression/relaxation_program.md).

For a hook gate alpha on component c (qk / head / attn / mlp / ngrp, see geo_probes.py):
  quench            L(w*, alpha=0) - L(w*)                         (E1)
  quench+recover    alpha=0 at step 0, then N recovery steps
  anneal            alpha: 1 -> 0 linearly over T_anneal steps while fine-tuning, then
                    T_hold steps at alpha=0
  linear response   OBS block formula  1/2 w_r^T [ (H^-1)_rr ]^-1 w_r
                    = 1/2 ( d^T H d - v_k^T H_kk^-1 v_k ),  v = H d,  k = retained set,
                    solved by CG on masked exact HVPs (exact-teacher KL, calib set)
where d is the parameter-space line of the gate and r the parameter block that is frozen
once the component is removed (its gradients vanish exactly when alpha = 0, so Adam
leaves it untouched during the hold).

All losses are exact-teacher KL; fine-tuning uses soft targets unless stated.
"""
from __future__ import annotations

import copy
import math

import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.geo_probes import Gate, GateModel
from compress.experimental.teacher import evaluate_teacher, kl_loss, teacher_probs


class Annealer:
    def __init__(self, model: HookedTransformer, train: t.Tensor, process, calib: t.Tensor,
                 p_calib: t.Tensor, test: t.Tensor, p_test: t.Tensor, group_size: int = 32):
        self.model0, self.train, self.process = model, train, process
        self.calib, self.p_calib, self.test, self.p_test = calib, p_calib, test, p_test
        self.gm = GateModel(model, calib, group_size)
        self.base_c = evaluate_teacher(model, calib, p_calib)
        self.base_t = evaluate_teacher(model, test, p_test)

    # ---- gate hook reading a mutable holder ------------------------------------------------
    def _hook(self, gate: Gate, holder: dict):
        l, i = gate.layer, gate.idx
        if gate.kind == "qk":
            def f(scores, hook):
                s = scores[:, i]
                scores[:, i] = t.where(s > -1e4, s * holder["a"], s)
                return scores
            return (f"blocks.{l}.attn.hook_attn_scores", f)
        if gate.kind == "head":
            def f(z, hook):
                z[:, :, i] = z[:, :, i] * holder["a"]
                return z
            return (f"blocks.{l}.attn.hook_z", f)
        if gate.kind == "attn":
            return (f"blocks.{l}.hook_attn_out", lambda a, hook: a * holder["a"])
        if gate.kind == "mlp":
            return (f"blocks.{l}.hook_mlp_out", lambda a, hook: a * holder["a"])
        if gate.kind == "ngrp":
            idx = self.gm.groups[l][i]
            def f(a, hook):
                a = a.clone()                      # ReLU output: never edit in place
                a[..., idx] = a[..., idx] * holder["a"]
                return a
            return (f"blocks.{l}.mlp.hook_post", f)
        raise ValueError(gate.kind)

    # ---- removed parameter block (frozen at alpha = 0) --------------------------------------
    def removed_mask(self, gate: Gate, model: HookedTransformer) -> dict[str, t.Tensor]:
        l, i = gate.layer, gate.idx
        blk = model.blocks[l]
        m = {}
        pre = f"blocks.{l}."
        if gate.kind == "qk":
            for n in ("W_Q", "W_K", "b_Q", "b_K"):
                mk = t.zeros_like(getattr(blk.attn, n), dtype=t.bool); mk[i] = True; m[pre + "attn." + n] = mk
        elif gate.kind == "head":
            mk = t.zeros_like(blk.attn.W_O, dtype=t.bool); mk[i] = True; m[pre + "attn.W_O"] = mk
        elif gate.kind == "attn":
            for n in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"):
                m[pre + "attn." + n] = t.ones_like(getattr(blk.attn, n), dtype=t.bool)
        elif gate.kind == "mlp":
            for n in ("W_in", "b_in", "W_out", "b_out"):
                m[pre + "mlp." + n] = t.ones_like(getattr(blk.mlp, n), dtype=t.bool)
        elif gate.kind == "ngrp":
            idx = self.gm.groups[l][i]
            mk = t.zeros_like(blk.mlp.W_in, dtype=t.bool); mk[:, idx] = True; m[pre + "mlp.W_in"] = mk
            mk = t.zeros_like(blk.mlp.b_in, dtype=t.bool); mk[idx] = True; m[pre + "mlp.b_in"] = mk
            mk = t.zeros_like(blk.mlp.W_out, dtype=t.bool); mk[idx] = True; m[pre + "mlp.W_out"] = mk
        return m

    # ---- one run ------------------------------------------------------------------------------
    def run(self, gate: Gate, T_anneal: int, T_hold: int, lr: float = 1e-4, batch_size: int = 128,
            seed: int = 0, soft: bool = True, return_up: bool = False) -> dict:
        """T_anneal = 0 -> quench then T_hold recovery steps. return_up=True: after the hold,
        anneal back 0 -> 1 over T_anneal steps and hold T_hold (hysteresis run)."""
        model = copy.deepcopy(self.model0)
        holder = {"a": 1.0}
        name, f = self._hook(gate, holder)
        model.add_hook(name, f)
        opt = t.optim.Adam(model.parameters(), lr=lr)
        g = t.Generator().manual_seed(seed)
        w0 = [p.detach().clone() for p in model.parameters()]
        schedule = [max(0.0, 1.0 - s / T_anneal) if T_anneal > 0 else 0.0 for s in range(T_anneal)] + [0.0] * T_hold
        if return_up:
            schedule += [min(1.0, s / T_anneal) if T_anneal > 0 else 1.0 for s in range(T_anneal)] + [1.0] * T_hold
        trace, kl_marks = [], {}
        model.train()
        for s, a in enumerate(schedule):
            holder["a"] = a
            idx = t.randint(0, len(self.train), (batch_size,), generator=g)
            batch = self.train[idx]
            if soft:
                loss = kl_loss(model, batch, teacher_probs(self.process, batch))
            else:
                logits = model(batch[:, :-1])
                loss = t.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 10 == 0:
                trace.append((a, loss.item()))
            if s + 1 == T_anneal + T_hold:            # end of the down-leg
                model.eval()
                kl_marks["down_calib_kl"] = evaluate_teacher(model, self.calib, self.p_calib)["kl"]
                model.train()
        model.eval()
        rc = evaluate_teacher(model, self.calib, self.p_calib)
        rt = evaluate_teacher(model, self.test, self.p_test)
        disp = math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(model.parameters(), w0)))
        model.reset_hooks()
        out = dict(gate=gate.label, T_anneal=T_anneal, T_hold=T_hold, soft=soft, return_up=return_up,
                   end_calib_kl=rc["kl"], end_test_kl=rt["kl"], end_test_nll=rt["nll"],
                   end_test_kl_per_pos=rt["kl_per_pos"], displacement=disp, trace=trace, **kl_marks)
        return out

    # ---- linear response (OBS block) via CG on masked HVPs ---------------------------------
    def linear_response(self, gate: Gate, n_hvp_calib: int = 1024, cg_iters: int = 30, damp: float = 1e-6) -> dict:
        model = self.model0
        names, params = zip(*model.named_parameters())
        tokens, probs = self.calib[:n_hvp_calib], self.p_calib[:n_hvp_calib]
        loss = kl_loss(model, tokens, probs)
        grads = t.autograd.grad(loss, params, create_graph=True)
        d = self.gm.direction(gate)
        dvec = t.cat([d.get(n, t.zeros_like(p)).reshape(-1) for n, p in zip(names, params)])
        rm = self.removed_mask(gate, model)
        keep = t.cat([(~rm.get(n, t.zeros_like(p, dtype=t.bool))).reshape(-1) for n, p in zip(names, params)]).float()

        def hvp(v):
            vs, off = [], 0
            for p in params:
                vs.append(v[off:off + p.numel()].view_as(p)); off += p.numel()
            gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
            Hv = t.autograd.grad(gv, params, retain_graph=True)
            return t.cat([h.reshape(-1) for h in Hv]).detach()

        gvec = t.cat([gi.detach().reshape(-1) for gi in grads])
        Hd = hvp(dvec)
        dHd = float(Hd @ dvec)
        gd = float(gvec @ dvec)
        # CG for (H_kk + lam I) x = v_k,  v = Hd restricted to retained set.
        # H_kk is generally indefinite away from an exact minimum, so use a scale-aware
        # Tikhonov damping lam = damp_rel * (d^T H d / |d|^2) (logged) instead of an
        # absolute constant; the solve is then the damped-Newton relaxation of E7.
        damp_rel = 1e-2
        lam = max(damp, damp_rel * abs(dHd) / max(float(dvec @ dvec), 1e-30))
        v = Hd * keep
        x = t.zeros_like(v); r = v.clone(); p_ = r.clone(); rs = float(r @ r)
        res_hist = []
        for _ in range(cg_iters):
            Ap = hvp(p_ * keep) * keep + lam * p_
            alpha = rs / max(float(p_ @ Ap), 1e-30)
            x += alpha * p_; r -= alpha * Ap
            rs_new = float(r @ r); res_hist.append(math.sqrt(rs_new))
            if rs_new < 1e-14:
                break
            p_ = r + (rs_new / rs) * p_; rs = rs_new
        relax = float(v @ x)
        return dict(gate=gate.label, g_dot_d=gd, dHd=dHd, quench_quad=0.5 * dHd - gd,
                    relaxation_quad=0.5 * relax, linear_response=0.5 * dHd - gd - 0.5 * relax,
                    cg_residual=res_hist[-1] / max(math.sqrt(float(v @ v)), 1e-30))
