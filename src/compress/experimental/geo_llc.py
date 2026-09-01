"""
E3 — local learning coefficient (SGLD) vs Hessian soft rank (Lanczos), per checkpoint
(notes/compression/singular_geometry_evaluation.md §4, Tier 1).

Conventions (all in per-token-loss units; the Gibbs measure is invariant to rescaling L by c
and nbeta by 1/c, so lambda-hat is convention-free as long as SGLD and the quadratic
formula use the same L and nbeta):

  L(w)     = mean per-token exact-teacher KL (teacher.py); K(w*) > 0 is subtracted.
  pi(w)   ~ exp( -nbeta L(w) - gamma/2 |w - w*|^2 )
  SGLD     w <- w - eps/2 [ nbeta grad L_batch(w) + gamma (w - w*) ] + sqrt(eps) N(0, I)
  lambda_hat_SGLD = nbeta ( E_pi[L] - L(w*) )                       (Lau et al. 2023)
  lambda_hat_quad = 1/2 sum_i nbeta h_i / (nbeta h_i + gamma)        (docx eq. 8),
                    with the eigenvalue sum estimated by stochastic Lanczos quadrature.

nbeta default: n / log n with n = number of training *sequences* and the per-sequence loss
= 16 x per-token loss, i.e. nbeta_token = 16 n / log n.
"""
from __future__ import annotations

import math

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.teacher import kl_loss, teacher_probs


# ---- SGLD ------------------------------------------------------------------------------------

def sgld_llc(model: HookedTransformer, train: t.Tensor, process, nbeta: float, gamma: float,
             eps: float, n_steps: int = 1000, burn_in: int = 200, batch_size: int = 256,
             seed: int = 0, loss_star: float | None = None) -> dict:
    """One chain. Returns lambda_hat, the loss trace, and the final displacement |w - w*|."""
    model = model.eval()
    params = [p for p in model.parameters()]
    w_star = [p.detach().clone() for p in params]
    g = t.Generator().manual_seed(seed)
    tg = t.Generator().manual_seed(10_000 + seed)
    losses, dists = [], []
    for step in range(n_steps):
        idx = t.randint(0, len(train), (batch_size,), generator=g)
        batch = train[idx]
        probs = teacher_probs(process, batch)
        loss = kl_loss(model, batch, probs)
        grads = t.autograd.grad(loss, params)
        with t.no_grad():
            for p, gr, p0 in zip(params, grads, w_star):
                noise = t.randn(p.shape, generator=tg)
                p -= 0.5 * eps * (nbeta * gr + gamma * (p - p0))
                p += math.sqrt(eps) * noise
        losses.append(loss.item())
        if step % 50 == 0:
            with t.no_grad():
                dists.append(math.sqrt(sum((p - p0).pow(2).sum().item() for p, p0 in zip(params, w_star))))
        if not math.isfinite(loss.item()) or loss.item() > 5.0:
            break
    with t.no_grad():
        for p, p0 in zip(params, w_star):
            p.copy_(p0)
    tail = np.array(losses[burn_in:]) if len(losses) > burn_in else np.array(losses)
    L0 = loss_star if loss_star is not None else float(np.mean(losses[:5]))
    lam = nbeta * (float(tail.mean()) - L0) if len(tail) else float("nan")
    return dict(lambda_hat=lam, mean_loss=float(tail.mean()) if len(tail) else float("nan"),
                loss_star=L0, n_kept=int(len(tail)), diverged=len(losses) < n_steps,
                loss_trace=[float(x) for x in losses[::10]], dist_trace=dists)


# ---- Lanczos / SLQ ---------------------------------------------------------------------------

class FlatHVP:
    def __init__(self, model: HookedTransformer, tokens: t.Tensor, probs: t.Tensor):
        self.params = [p for p in model.parameters()]
        loss = kl_loss(model, tokens, probs)
        self.grads = t.autograd.grad(loss, self.params, create_graph=True)
        self.dim = sum(p.numel() for p in self.params)

    def __call__(self, v: t.Tensor) -> t.Tensor:
        vs, off = [], 0
        for p in self.params:
            vs.append(v[off:off + p.numel()].view_as(p)); off += p.numel()
        gv = sum((g * x).sum() for g, x in zip(self.grads, vs))
        Hv = t.autograd.grad(gv, self.params, retain_graph=True)
        return t.cat([h.reshape(-1) for h in Hv]).detach()


def lanczos(hvp: FlatHVP, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-reorthogonalized Lanczos from a random unit start. Returns Ritz values and the
    squared first components of the Ritz vectors (SLQ weights)."""
    g = t.Generator().manual_seed(seed)
    v = t.randn(hvp.dim, generator=g); v /= v.norm()
    V = [v]
    alphas, betas = [], []
    w = hvp(v)
    a = float(w @ v); alphas.append(a)
    w = w - a * v
    for j in range(1, m):
        b = float(w.norm())
        if b < 1e-10:
            break
        betas.append(b)
        v = w / b
        # full reorthogonalization
        for u in V:
            v = v - (u @ v) * u
        v /= v.norm()
        V.append(v)
        w = hvp(v)
        a = float(w @ v); alphas.append(a)
        w = w - a * v - b * V[-2]
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    theta, S = np.linalg.eigh(T)
    return theta, S[0, :] ** 2


def slq_summary(hvp: FlatHVP, m: int = 60, n_probes: int = 4, nbeta_list=(), gamma_list=(),
                seed: int = 0) -> dict:
    """Spectral-density estimate and the derived quantities."""
    thetas, weights = [], []
    for k in range(n_probes):
        th, wt = lanczos(hvp, m, seed + k)
        thetas.append(th); weights.append(wt / n_probes)
    th, wt = np.concatenate(thetas), np.concatenate(weights)     # density: sum wt delta(theta)
    d = hvp.dim
    out = dict(dim=d, ritz=th.tolist(), weights=wt.tolist(),
               trace=float(d * (wt * th).sum()),
               top_ritz=float(th.max()),
               count_above={str(c): float(d * wt[th > c].sum()) for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)},
               lambda_quad={})
    for nb in nbeta_list:
        for gm in gamma_list:
            r = nb * np.clip(th, 0, None) / (nb * np.clip(th, 0, None) + gm)
            out["lambda_quad"][f"{nb:.3e}|{gm:g}"] = float(0.5 * d * (wt * r).sum())
    return out
