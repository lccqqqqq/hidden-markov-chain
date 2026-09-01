"""
R3 — field response (notes/compression/relaxation_program.md §R3).

A gauge-invariant "field" Omega_c(w) is conjugate to each structured component c (same 72
components as E1's hook gates). Fine-tuning on  L + h * Omega_c  from w* gives the
relaxed response w*(h); the parametric curve (m_c(h), dKL(h)) with m_c = Omega_c(w*(h)) /
Omega_c(w*) is the component's rate-distortion curve WITH relaxation and a free endpoint.

Fields (all invariant under the model's exact reparameterisations):
  qk   (l,h)  ||W_Q[h] W_K[h]^T||_F^2          (GL(d_head) gauge W_Q A, W_K A^-T)
  head (l,h)  ||W_V[h] W_O[h]||_F^2             (GL(d_head) gauge W_V A, A^-1 W_O)
  ngrp (l,g)  sum_j ||W_in[:,j]|| * ||W_out[j,:]||   (ReLU scaling gauge)
  attn (l)    mean_{b,t} ||attn_out||^2         (activation field, via hook)
  mlp  (l)    mean_{b,t} ||mlp_out||^2
Non-invariant fields such as h||W_Q||^2 are absorbed by the gauge at zero cost and are
deliberately not offered.
"""
from __future__ import annotations

import copy
import math

import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.geo_probes import Gate, GateModel
from compress.experimental.teacher import evaluate_teacher, kl_loss, teacher_probs

TARGETS_MNAT = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FIELD_KINDS = ("qk", "head", "attn", "mlp", "ngrp")


def field_gates(gm: GateModel) -> list[Gate]:
    return [g for g in gm.all_gates() if g.kind in FIELD_KINDS]


# ---- fields ----------------------------------------------------------------------------------

def param_field(model: HookedTransformer, gate: Gate, groups) -> t.Tensor:
    """Differentiable Omega for the parameter fields (qk / head / ngrp)."""
    blk = model.blocks[gate.layer]
    if gate.kind == "qk":
        WQ, WK = blk.attn.W_Q[gate.idx], blk.attn.W_K[gate.idx]          # (d_model, d_head)
        return (WQ @ WK.T).pow(2).sum()
    if gate.kind == "head":
        WV, WO = blk.attn.W_V[gate.idx], blk.attn.W_O[gate.idx]          # (d_model,d_head),(d_head,d_model)
        return (WV @ WO).pow(2).sum()
    if gate.kind == "ngrp":
        idx = groups[gate.layer][gate.idx]
        return (blk.mlp.W_in[:, idx].norm(dim=0) * blk.mlp.W_out[idx, :].norm(dim=1)).sum()
    raise ValueError(gate.kind)


def act_hook_name(gate: Gate) -> str:
    return f"blocks.{gate.layer}.hook_attn_out" if gate.kind == "attn" else f"blocks.{gate.layer}.hook_mlp_out"


class FieldModel:
    """Evaluates Omega for every component (no grad) and provides the training-step loss."""

    def __init__(self, model: HookedTransformer, gm: GateModel, process, act_tokens: t.Tensor):
        self.gm, self.process = gm, process
        self.groups = gm.groups
        self.gates = field_gates(gm)
        self.act_tokens = act_tokens[:, :-1]           # fixed subset for activation fields

    @t.no_grad()
    def omegas(self, model: HookedTransformer) -> dict[str, float]:
        out = {}
        store = {}
        names = sorted({act_hook_name(g) for g in self.gates if g.kind in ("attn", "mlp")})
        if names:
            model.run_with_hooks(self.act_tokens, return_type=None,
                                 fwd_hooks=[(n, lambda a, hook: store.__setitem__(hook.name, a.pow(2).sum(-1).mean().item()))
                                            for n in names])
        for g in self.gates:
            out[g.label] = store[act_hook_name(g)] if g.kind in ("attn", "mlp") else param_field(model, g, self.groups).item()
        return out

    def loss_with_field(self, model: HookedTransformer, batch: t.Tensor, gate: Gate, h: float) -> tuple[t.Tensor, t.Tensor]:
        """Returns (total loss, Omega) for one training batch; differentiable."""
        probs = teacher_probs(self.process, batch)
        if gate.kind in ("attn", "mlp"):
            store = {}
            model.add_hook(act_hook_name(gate), lambda a, hook: store.__setitem__("a", a))
            try:
                kl = kl_loss(model, batch, probs)
            finally:
                model.reset_hooks()
            omega = store["a"].pow(2).sum(-1).mean()
        else:
            kl = kl_loss(model, batch, probs)
            omega = param_field(model, gate, self.groups)
        return kl + h * omega, omega


# ---- gauge-invariance check ------------------------------------------------------------------

@t.no_grad()
def assert_gauge_invariant(model: HookedTransformer, fm: FieldModel, tokens: t.Tensor, seed: int = 0):
    m = copy.deepcopy(model)
    g = t.Generator().manual_seed(seed)
    logits0 = model(tokens[:256, :-1])
    om0 = fm.omegas(m)
    l, h = 1, 0
    d = t.exp(0.5 * t.randn(m.cfg.d_head, generator=g))
    a = m.blocks[l].attn
    a.W_Q.data[h] *= d; a.b_Q.data[h] *= d; a.W_K.data[h] /= d; a.b_K.data[h] /= d      # QK gauge
    a.W_V.data[h] *= d; a.b_V.data[h] *= d; a.W_O.data[h] /= d[:, None]                 # VO gauge
    c = t.exp(0.5 * t.randn(m.cfg.d_mlp, generator=g))
    mlp = m.blocks[l].mlp
    mlp.W_in.data *= c; mlp.b_in.data *= c; mlp.W_out.data /= c[:, None]                # ReLU gauge
    logits1 = m(tokens[:256, :-1])
    om1 = fm.omegas(m)
    dl = (logits1 - logits0).abs().max().item()
    assert dl < 1e-3, f"gauge changed the function: max|dlogits| = {dl}"
    worst = max((abs(om1[k] - om0[k]) / max(abs(om0[k]), 1e-12), k) for k in om0 if not k.startswith(("attn", "mlp")))
    assert worst[0] < 1e-4, f"field {worst[1]} not gauge invariant: rel change {worst[0]}"
    return dl, worst


# ---- sweeps ----------------------------------------------------------------------------------

def finetune_with_field(model0: HookedTransformer, fm: FieldModel, gate: Gate, h: float, train: t.Tensor,
                        steps: int, lr: float = 1e-4, batch_size: int = 128, seed: int = 0):
    model = copy.deepcopy(model0)
    opt = t.optim.Adam(model.parameters(), lr=lr)
    g = t.Generator().manual_seed(seed)
    model.train()
    trace = []
    for step in range(steps):
        idx = t.randint(0, len(train), (batch_size,), generator=g)
        loss, omega = fm.loss_with_field(model, train[idx], gate, h)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            trace.append((round(loss.item(), 6), round(omega.item(), 6)))
    model.eval()
    return model, trace


def displacement(model: HookedTransformer, model0: HookedTransformer) -> float:
    return math.sqrt(sum((p - q).pow(2).sum().item() for p, q in zip(model.parameters(), model0.parameters())))


def field_sweep(model0: HookedTransformer, fm: FieldModel, gate: Gate, train: t.Tensor,
                calib: t.Tensor, p_calib: t.Tensor, test: t.Tensor, p_test: t.Tensor,
                kl0_calib: float, kl0_test: float, targets=TARGETS_MNAT, steps: int = 300,
                lr: float = 1e-4, batch_size: int = 128, seed: int = 0) -> dict:
    omega0 = fm.omegas(model0)[gate.label]
    h_values = [0.0] + [tg * 1e-3 / max(omega0, 1e-12) for tg in targets]
    res = dict(gate=gate.label, omega0=omega0, targets_mnat=[0.0] + list(targets), h_values=h_values,
               m=[], dkl_calib=[], dkl_test=[], displacement=[], trace=[])
    for h in h_values:
        model, trace = finetune_with_field(model0, fm, gate, h, train, steps, lr, batch_size, seed)
        res["m"].append(fm.omegas(model)[gate.label] / max(omega0, 1e-12))
        res["dkl_calib"].append(evaluate_teacher(model, calib, p_calib)["kl"] - kl0_calib)
        res["dkl_test"].append(evaluate_teacher(model, test, p_test)["kl"] - kl0_test)
        res["displacement"].append(displacement(model, model0))
        res["trace"].append(trace)
    m = res["m"]
    res["jump"] = any(m[i + 1] < 0.5 * m[i] for i in range(1, len(m) - 1))
    return res


def h_for_half(sweep: dict) -> float:
    """h at which m_c crosses 0.5 (log-linear interpolation over the non-zero h values)."""
    hs, ms = sweep["h_values"][1:], sweep["m"][1:]
    for (h0, m0), (h1, m1) in zip(zip(hs, ms), zip(hs[1:], ms[1:])):
        if m0 >= 0.5 > m1:
            f = (m0 - 0.5) / max(m0 - m1, 1e-12)
            return math.exp(math.log(h0) + f * (math.log(h1) - math.log(h0)))
    return hs[-1] if ms[-1] < 0.5 else hs[-1]
