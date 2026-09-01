"""
R2.1 / R2.2 — finite-temperature (Gibbs-posterior) geometry
(notes/compression/relaxation_program.md §R2).

  pi_T(w) ~ exp( -nbeta L(w) - gamma/2 |w - w*|^2 ),  T = 1/nbeta,  L = per-token exact-teacher KL.

* sgld_moments : the SGLD chain of geo_llc.sgld_llc, accumulating Welford mean/variance
                 per weight over post-burn-in (thinned) samples instead of only the loss.
* hutchinson_diag : per-weight diagonal Hessian, diag(H)_i ~ mean_v (v * Hv)_i, Rademacher v.
* nongauss_map  : rho_i = var_i / var_quad_i with var_quad_i = T / (max(h_i,0) + T gamma),
                 the Gaussian (quadratic-landscape) prediction at the same (T, gamma).
                 rho << 1 : cliff / finite extent;  rho >> 1 : sub-quadratic / saturation.
* posterior_omega : Omega_i(b) = (T/2) sum_j dW_ij(b)^2 / sigma_ij^2 — HAWQ-V2's sensitivity
                 with the isotropic tr(H)/n replaced by the per-weight posterior precision.
                 Same grids as run_geo_alloc.py's trace branch (RTN grid, ternary_fq, W for delete).
"""
from __future__ import annotations

import math

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param
from compress.experimental.geo_alloc import DELETE, TERNARY, ternary_fq
from compress.experimental.teacher import kl_loss, teacher_probs


def sgld_moments(model: HookedTransformer, train: t.Tensor, process, nbeta: float, gamma: float,
                 eps: float, n_steps: int, burn_in: int, batch_size: int = 256, seed: int = 0,
                 thin: int = 1, loss_star: float | None = None, n_track: int = 64) -> dict:
    """One SGLD chain; returns per-weight posterior mean/variance (Welford over post-burn-in
    samples every `thin` steps), the loss trace, lambda_hat, and a plateau check on a fixed
    random subset of `n_track` coordinates (running variance at burn_in, mid, end). w* restored."""
    model = model.eval()
    named = [(n, p) for n, p in model.named_parameters()]
    params = [p for _, p in named]
    w_star = [p.detach().clone() for p in params]
    g = t.Generator().manual_seed(seed)
    tg = t.Generator().manual_seed(10_000 + seed)
    # Welford accumulators
    cnt = 0
    mean = [t.zeros_like(p) for p in params]
    m2 = [t.zeros_like(p) for p in params]
    # tracked coordinates (flat index into the concatenated parameter vector)
    dim = sum(p.numel() for p in params)
    track_idx = t.randperm(dim, generator=t.Generator().manual_seed(7))[:n_track]
    offsets = [0]
    for p in params:
        offsets.append(offsets[-1] + p.numel())
    def track_var():
        flat_var = t.cat([(mm / max(cnt - 1, 1)).reshape(-1) for mm in m2])
        return flat_var[track_idx].clone()
    plateau = {}
    losses, dists = [], []
    diverged = False
    mid = burn_in + (n_steps - burn_in) // 2
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
        lv = loss.item()
        losses.append(lv)
        if not math.isfinite(lv) or lv > 5.0:
            diverged = True
            break
        if step % 50 == 0:
            with t.no_grad():
                dists.append(math.sqrt(sum((p - p0).pow(2).sum().item() for p, p0 in zip(params, w_star))))
        if step >= burn_in and (step - burn_in) % thin == 0:
            cnt += 1
            with t.no_grad():
                for p, mu, mm in zip(params, mean, m2):
                    delta = p - mu
                    mu += delta / cnt
                    mm += delta * (p - mu)
        if step in (burn_in, mid, n_steps - 1) and cnt > 1:
            plateau[str(step)] = track_var().tolist()
    with t.no_grad():
        for p, p0 in zip(params, w_star):
            p.copy_(p0)
    var = {n: (mm / max(cnt - 1, 1)).detach() for (n, _), mm in zip(named, m2)}
    mu = {n: m.detach() for (n, _), m in zip(named, mean)}
    tail = losses[burn_in:] if len(losses) > burn_in else losses
    L0 = loss_star if loss_star is not None else float(sum(losses[:5]) / max(len(losses[:5]), 1))
    mean_loss = float(sum(tail) / len(tail)) if tail else float("nan")
    lam = nbeta * (mean_loss - L0) if tail else float("nan")
    return dict(mean=mu, var=var, n_samples=cnt, mean_loss=mean_loss, loss_star=L0,
                delta_loss=mean_loss - L0, lambda_hat=lam, diverged=diverged,
                loss_trace=[float(x) for x in losses[::10]], dist_trace=dists, plateau=plateau,
                track_idx=track_idx.tolist())


def hutchinson_diag(model: HookedTransformer, tokens: t.Tensor, probs: t.Tensor,
                    n_probes: int = 256, seed: int = 0) -> dict[str, t.Tensor]:
    """Per-weight diagonal of the exact-teacher-KL Hessian, Hutchinson with Rademacher probes."""
    names, params = zip(*model.named_parameters())
    loss = kl_loss(model, tokens, probs)
    grads = t.autograd.grad(loss, params, create_graph=True)
    g = t.Generator().manual_seed(seed)
    acc = [t.zeros_like(p) for p in params]
    for _ in range(n_probes):
        vs = [(t.randint(0, 2, p.shape, generator=g).float() * 2 - 1) for p in params]
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
        Hvs = t.autograd.grad(gv, params, retain_graph=True)
        for a, hv, v in zip(acc, Hvs, vs):
            a += (hv * v).detach()
    return {n: a / n_probes for n, a in zip(names, acc)}


def nongauss_map(var: dict[str, t.Tensor], diag_h: dict[str, t.Tensor], nbeta: float, gamma: float,
                 only_quantized: bool = True) -> dict[str, dict]:
    """Per-tensor summary of rho_i = var_i / (T / (max(h_i, 0) + T gamma))."""
    T = 1.0 / nbeta
    out = {}
    for n, v in var.items():
        if only_quantized and not is_quantized_param(n):
            continue
        h = diag_h[n].clamp_min(0.0)
        vq = T / (h + T * gamma)
        rho = (v / vq).reshape(-1)
        q = t.quantile(rho, t.tensor([0.25, 0.5, 0.75]))
        mean_v, mean_h = float(v.mean()), float(h.mean())
        out[n] = dict(median=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                      frac_lt_0p3=float((rho < 0.3).float().mean()), frac_gt_3=float((rho > 3).float().mean()),
                      mean_var=mean_v, mean_h=mean_h,
                      tensor_rho=mean_v / (T / (mean_h + T * gamma)),
                      numel=int(v.numel()))
    return out


def posterior_omega(model: HookedTransformer, var: dict[str, t.Tensor], nbeta: float,
                    menu: tuple[int, ...]) -> dict[str, dict[int, float]]:
    """Omega_i(b) = (T/2) sum_j dW_ij(b)^2 / sigma_ij^2, grids as in run_geo_alloc.py (trace branch)."""
    T = 1.0 / nbeta
    omega = {}
    for n, p in model.named_parameters():
        if not is_quantized_param(n):
            continue
        w2 = Quantizer._channel_view(p.data, "per_channel", n)
        v2 = Quantizer._channel_view(var[n], "per_channel", n)
        v2 = v2.clamp_min(1e-12 * float(v2.max()))
        row = {}
        for b in menu:
            if b == DELETE:
                dW = w2
            elif b == TERNARY:
                dW = ternary_fq(w2)[0] - w2
            else:
                s_, z_, lo, hi = Quantizer.grid(w2, b, False)
                dW = Quantizer.fake_quant(w2, s_, z_, lo, hi) - w2
            row[b] = float(0.5 * T * (dW.pow(2) / v2).sum())
        omega[n] = row
    return omega
