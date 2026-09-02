"""
§4.3 of singular_geometry_evaluation.md — finite-temperature correlators of COARSE observables.

R2 showed that weight-space sampling cannot thermalise a 2e5-dimensional flat manifold. The
objects proposed instead are correlators of the 72 gauge-invariant component observables
Ω_c (geo_field.py) along a stochastic trajectory started at w*:

  trajectory       w_{t+1} = w_t - step(∇L_batch) [+ noise]     dynamics ∈ {adam, sgd, sgld}
  fluctuations     x_c(t) = Ω_c(w_t)/Ω_c(w*) - 1
  correlators      C_cc' = cov(x_c, x_c')            (FDT:  dΩ_c'/dh_c = -β C_cc'  ->  compare R3.3 χ)
  loss correlator  G_c   = cov(ΔL, x_c)              (envelope: components with G_c ≈ 0 are decorative)
  autocorrelation  A_c(τ) -> integrated time τ_c     (relaxation spectrum; compare R1 step dependence)
  activation corr. H_T = <2 X Xᵀ>_trajectory per matrix  (thermal GPTQ Hessian)

Compression uses tested here:
  * deletion order by thermal variance C_cc (loosest components first) + 200-step recovery,
    against the R3.2 global-field frontier at matched parameter counts;
  * ThermalGPTQ: reference GPTQ rounding with H_T instead of H(w*)  (no sequential propagation
    in either arm, so the comparison isolates the Hessian; the propagated reference is also
    reported).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.geo_field import FieldModel
from compress.experimental.teacher import evaluate_teacher, kl_loss, teacher_probs
from compress.gptq import GPTQ


# ---- trajectories ---------------------------------------------------------------------------

def thermal_trajectory(model0: HookedTransformer, fm: FieldModel, train: t.Tensor, process,
                       dynamics: str, steps: int, lr: float, batch_size: int = 128, seed: int = 0,
                       record_every: int = 5, nbeta: float | None = None, gamma: float = 100.0,
                       calib: t.Tensor | None = None, p_calib: t.Tensor | None = None,
                       kl_every: int = 50) -> dict:
    model = copy.deepcopy(model0)
    params = [p for p in model.parameters()]
    w0 = [p.detach().clone() for p in params]
    g = t.Generator().manual_seed(seed)
    ng = t.Generator().manual_seed(10_000 + seed)
    opt = t.optim.Adam(params, lr=lr) if dynamics == "adam" else None
    labels = [gg.label for gg in fm.gates]
    om0 = fm.omegas(model0)
    rec_steps, X, batch_loss, dist, kl_steps, kl_vals = [], [], [], [], [], []
    model.train()
    for s in range(steps):
        idx = t.randint(0, len(train), (batch_size,), generator=g)
        batch = train[idx]
        loss = kl_loss(model, batch, teacher_probs(process, batch))
        gloss = (gamma / nbeta) if (nbeta and dynamics in ("adam", "sgd") and gamma > 0) else 0.0
        if dynamics == "adam":
            opt.zero_grad(); loss.backward()
            if gloss:
                with t.no_grad():
                    for p, p0 in zip(params, w0):
                        p.grad += gloss * (p - p0)          # localization in loss units
            opt.step()
        else:
            grads = t.autograd.grad(loss, params)
            with t.no_grad():
                for p, gr, p0 in zip(params, grads, w0):
                    if dynamics == "sgd":
                        p -= lr * (gr + gloss * (p - p0))
                    elif dynamics == "sgld":            # lr plays the role of eps
                        p -= 0.5 * lr * (nbeta * gr + gamma * (p - p0))
                        p += math.sqrt(lr) * t.randn(p.shape, generator=ng)
                    else:
                        raise ValueError(dynamics)
        if s % record_every == 0:
            model.eval()
            om = fm.omegas(model)
            X.append([om[l] / max(om0[l], 1e-12) - 1.0 for l in labels])
            rec_steps.append(s); batch_loss.append(loss.item())
            dist.append(math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(params, w0))))
            if calib is not None and s % kl_every == 0:
                kl_steps.append(s); kl_vals.append(evaluate_teacher(model, calib, p_calib)["kl"])
            model.train()
        if not math.isfinite(loss.item()) or loss.item() > 5:
            break
    model.eval()
    return dict(dynamics=dynamics, lr=lr, gamma=gamma, steps=steps, labels=labels, rec_steps=rec_steps,
                x=X, batch_loss=batch_loss, dist=dist, kl_steps=kl_steps, kl=kl_vals,
                diverged=len(rec_steps) < steps // record_every, model=model)


# ---- correlator analysis --------------------------------------------------------------------

def analyze(traj: dict, burn_frac: float = 0.25, max_lag: int = 300) -> dict:
    X = np.array(traj["x"]); n = len(X); b = int(burn_frac * n)
    Xs = X[b:]; L = np.array(traj["batch_loss"])[b:]
    mu = Xs.mean(0); C = np.cov(Xs.T)
    sd = np.sqrt(np.clip(np.diag(C), 1e-30, None)); corr = C / np.outer(sd, sd)
    G = np.array([np.cov(L, Xs[:, c])[0, 1] for c in range(Xs.shape[1])])
    Gn = G / (L.std() * sd + 1e-30)
    tau = []
    for c in range(Xs.shape[1]):
        z = Xs[:, c] - mu[c]
        v = float((z * z).mean())
        if v < 1e-30:
            tau.append(float("nan")); continue
        acf = [float((z[:-k] * z[k:]).mean()) / v if k else 1.0 for k in range(min(max_lag, len(z) - 1))]
        s = 1.0
        for a in acf[1:]:
            if a <= 0:
                break
            s += 2 * a
        tau.append(s)
    return dict(labels=traj["labels"], mean=mu.tolist(), var=np.diag(C).tolist(), corr=corr.tolist(),
                loss_corr=Gn.tolist(), loss_cov=G.tolist(), tau_records=tau, n_samples=int(len(Xs)),
                record_every=int(traj["rec_steps"][1] - traj["rec_steps"][0]) if len(traj["rec_steps"]) > 1 else None)


# ---- thermal activation correlators + GPTQ ---------------------------------------------------

_INPUTS = {  # matrix -> hook giving its input, and how to shape it
    "W_Q": ("hook_resid_pre", None), "W_K": ("hook_resid_pre", None), "W_V": ("hook_resid_pre", None),
    "W_O": ("attn.hook_z", "flatten_heads"), "W_in": ("hook_resid_mid", None), "W_out": ("mlp.hook_post", None),
}


@t.no_grad()
def activation_hessians(model: HookedTransformer, tokens: t.Tensor) -> dict[str, t.Tensor]:
    """{param_name: 2 XXᵀ} for every quantized matrix, from one weight state (no propagation)."""
    L = model.cfg.n_layers
    names = [f"blocks.{l}.{h}" for l in range(L) for h in ("hook_resid_pre", "attn.hook_z", "hook_resid_mid", "mlp.hook_post")]
    names.append(f"blocks.{L-1}.hook_resid_post")
    store = {}
    model.run_with_hooks(tokens, return_type=None,
                         fwd_hooks=[(n, lambda a, hook: store.__setitem__(hook.name, a.detach())) for n in names])
    H = {}
    for l in range(L):
        for w, (hk, mode) in _INPUTS.items():
            X = store[f"blocks.{l}.{hk}"]
            if mode == "flatten_heads":
                X = X.reshape(X.shape[0], X.shape[1], -1)
            X = X.reshape(-1, X.shape[-1]).float()
            pre = "attn" if w in ("W_Q", "W_K", "W_V", "W_O") else "mlp"
            H[f"blocks.{l}.{pre}.{w}"] = 2.0 * X.T @ X
    X = store[f"blocks.{L-1}.hook_resid_post"].reshape(-1, model.cfg.d_model).float()
    H["unembed.W_U"] = 2.0 * X.T @ X
    return H


class FixedHGPTQ(GPTQ):
    """Reference per-matrix GPTQ routine driven by a precomputed {param_name: H} dictionary
    (no sequential propagation). With H from w* this is the 'no-propagation' control; with
    H averaged along a thermal trajectory it is ThermalGPTQ."""
    name = "GPTQ-fixedH"

    def __init__(self, bits: int, H: dict[str, t.Tensor], **kw):
        super().__init__(bits, **kw)
        self.H = H

    @t.no_grad()
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        from compress.rtn import RTN
        model = copy.deepcopy(model)
        for name in ("embed.W_E", "pos_embed.W_pos"):
            p = dict(model.named_parameters())[name]
            w2 = self._channel_view(p.data, self.granularity, name)
            s_, z_, lo, hi = self.grid(w2, self._bits(name), self.symmetric)
            p.data.copy_(self._from_channel_view(self.fake_quant(w2, s_, z_, lo, hi), p.shape, name))
        for l in range(model.cfg.n_layers):
            blk = model.blocks[l]
            for wname in ("W_Q", "W_K", "W_V"):
                w = getattr(blk.attn, wname)
                W2 = w.data.permute(0, 2, 1).reshape(-1, w.shape[1])
                W2 = self._gptq_matrix(W2, self.H[f"blocks.{l}.attn.{wname}"], self._bits(f"blocks.{l}.attn.{wname}"))
                w.data.copy_(W2.reshape(w.shape[0], w.shape[2], w.shape[1]).permute(0, 2, 1))
            w = blk.attn.W_O
            W2 = w.data.reshape(-1, w.shape[-1]).T
            w.data.copy_(self._gptq_matrix(W2, self.H[f"blocks.{l}.attn.W_O"], self._bits(f"blocks.{l}.attn.W_O")).T.reshape(w.shape))
            w = blk.mlp.W_in
            w.data.copy_(self._gptq_matrix(w.data.T, self.H[f"blocks.{l}.mlp.W_in"], self._bits(f"blocks.{l}.mlp.W_in")).T)
            w = blk.mlp.W_out
            w.data.copy_(self._gptq_matrix(w.data.T, self.H[f"blocks.{l}.mlp.W_out"], self._bits(f"blocks.{l}.mlp.W_out")).T)
        w = model.unembed.W_U
        w.data.copy_(self._gptq_matrix(w.data.T, self.H["unembed.W_U"], self._bits("unembed.W_U")).T)
        return model


def thermal_hessians(model0: HookedTransformer, fm: FieldModel, train: t.Tensor, process, tokens: t.Tensor,
                     dynamics: str, steps: int, lr: float, n_states: int, seed: int = 0, **kw) -> dict[str, t.Tensor]:
    """Average of activation Hessians over n_states weight states sampled uniformly along a
    trajectory (the first state is w* itself)."""
    every = max(1, steps // n_states)
    acc = None; count = 0
    # re-run the trajectory, sampling H every `every` steps (cheaper than storing models)
    model = copy.deepcopy(model0)
    params = [p for p in model.parameters()]
    w0 = [p.detach().clone() for p in params]
    g = t.Generator().manual_seed(seed); ng = t.Generator().manual_seed(10_000 + seed)
    opt = t.optim.Adam(params, lr=lr) if dynamics == "adam" else None
    nbeta, gamma = kw.get("nbeta"), kw.get("gamma", 100.0)
    for s in range(steps + 1):
        if s % every == 0 and count < n_states:
            model.eval()
            H = activation_hessians(model, tokens)
            acc = H if acc is None else {k: acc[k] + H[k] for k in H}
            count += 1
            model.train()
        if s == steps:
            break
        idx = t.randint(0, len(train), (128,), generator=g)
        batch = train[idx]
        loss = kl_loss(model, batch, teacher_probs(process, batch))
        if dynamics == "adam":
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            grads = t.autograd.grad(loss, params)
            with t.no_grad():
                for p, gr, p0 in zip(params, grads, w0):
                    if dynamics == "sgd":
                        p -= lr * gr
                    else:
                        p -= 0.5 * lr * (nbeta * gr + gamma * (p - p0)); p += math.sqrt(lr) * t.randn(p.shape, generator=ng)
    return {k: v / count for k, v in acc.items()}
