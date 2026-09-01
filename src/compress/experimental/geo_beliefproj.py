"""
E8 — belief-probe projection of the residual stream
(notes/compression/singular_geometry_evaluation.md §4, Tier 4).

The residual stream of a Bayes-optimal predictor for an 18-state HMM need only carry the
17-dof belief simplex. We (1) fit linear probes residual -> belief at every read point,
(2) take the shared span of the probe read-out directions as P (64 x d'), (3) compile a new
HookedTransformer with d_model = d' by projecting every matrix through P, (4) recover briefly.

Projection rules (x -> Pᵀx on the residual stream; exact iff resid ∈ span(P), P orthonormal):
  W_E' = W_E P        W_pos' = W_pos P
  W_Q' = Pᵀ W_Q       W_K' = Pᵀ W_K      W_V' = Pᵀ W_V      (per head, first matrix dim)
  W_O' = W_O P        b_O' = b_O P
  W_in' = Pᵀ W_in     W_out' = W_out P   b_out' = b_out P
  W_U' = Pᵀ W_U       b_Q, b_K, b_V, b_in, b_U unchanged

Controls at the same d': P_rand (random orthonormal) and P_pca (top PCs of the pooled
residual stream). This is the docx §9.3 "shared projection", with P supplied by known
structure instead of inferred from loss geometry.
"""
from __future__ import annotations

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from utils import initialize_transformer_from_yaml


def read_points(n_layers: int) -> list[str]:
    return [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)] + [f"blocks.{n_layers-1}.hook_resid_post"]


@t.no_grad()
def collect_residuals(model: HookedTransformer, tokens: t.Tensor, batch_size: int = 2048) -> dict[str, t.Tensor]:
    """{read_point: (N*seq, d_model)} residual activations."""
    names = read_points(model.cfg.n_layers)
    out = {n: [] for n in names}
    for i in range(0, len(tokens), batch_size):
        store = {}
        model.run_with_hooks(tokens[i:i + batch_size, :-1], return_type=None,
                             fwd_hooks=[(n, lambda a, hook: store.__setitem__(hook.name, a.detach())) for n in names])
        for n in names:
            out[n].append(store[n].reshape(-1, store[n].shape[-1]))
    return {n: t.cat(v) for n, v in out.items()}


def ridge(X: t.Tensor, Y: t.Tensor, lam: float = 1e-3) -> tuple[t.Tensor, t.Tensor]:
    """Closed-form ridge with intercept. Returns (W (d x k), b (k,))."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = (X - xm).double(), (Y - ym).double()
    A = Xc.T @ Xc + lam * t.eye(X.shape[1], dtype=t.float64)
    W = t.linalg.solve(A, Xc.T @ Yc)
    b = ym.double() - xm.double() @ W
    return W.float(), b.float()


def r2(X: t.Tensor, Y: t.Tensor, W: t.Tensor, b: t.Tensor) -> float:
    pred = X @ W + b
    ss_res = (Y - pred).pow(2).sum()
    ss_tot = (Y - Y.mean(0)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


def fit_belief_probes(model: HookedTransformer, process, train_tokens: t.Tensor, heldout_tokens: t.Tensor,
                      lam: float = 1e-3) -> tuple[dict[str, t.Tensor], dict[str, float]]:
    """Returns ({read_point: W (64 x 18)}, {read_point: held-out R²})."""
    def beliefs(tok):
        return t.from_numpy(process.mixed_state_presentation(tok[:, :-1].numpy()).astype(np.float32)).reshape(-1, process.num_hidden_states)
    Xtr, Xho = collect_residuals(model, train_tokens), collect_residuals(model, heldout_tokens)
    Ytr, Yho = beliefs(train_tokens), beliefs(heldout_tokens)
    Ws, scores = {}, {}
    for n in Xtr:
        W, b = ridge(Xtr[n], Ytr, lam)
        Ws[n] = W
        scores[n] = r2(Xho[n], Yho, W, b)
    return Ws, scores


def subspace(kind: str, d_prime: int, probe_W: dict[str, t.Tensor] | None = None,
             resid: dict[str, t.Tensor] | None = None, d_model: int = 64, seed: int = 0) -> t.Tensor:
    """Orthonormal P (d_model x d_prime)."""
    if kind == "probe":
        M = t.cat(list(probe_W.values()), dim=1)                 # (64, 18 * n_read)
        U, S, _ = t.linalg.svd(M, full_matrices=False)
        return U[:, :d_prime].contiguous()
    if kind == "pca":
        X = t.cat(list(resid.values()))                          # pooled over read points
        X = X - X.mean(0)
        U, S, Vh = t.linalg.svd(X.double(), full_matrices=False)
        return Vh[:d_prime].T.float().contiguous()
    if kind == "rand":
        g = t.Generator().manual_seed(seed)
        Q, _ = t.linalg.qr(t.randn(d_model, d_prime, generator=g))
        return Q.contiguous()
    if kind == "identity":
        assert d_prime == d_model
        return t.eye(d_model)
    raise ValueError(kind)


@t.no_grad()
def compile_projected(model: HookedTransformer, model_cfg: dict, P: t.Tensor) -> HookedTransformer:
    """New HookedTransformer with d_model = P.shape[1], weights projected through P."""
    d_prime = P.shape[1]
    cfg = dict(model_cfg); cfg["n_embd"] = d_prime
    device = next(model.parameters()).device
    new = initialize_transformer_from_yaml(None, model_cfg=cfg).to(device)   # TL defaults to cuda if present
    P = P.to(device)
    Pt = P.T
    new.embed.W_E.copy_(model.embed.W_E @ P)
    new.pos_embed.W_pos.copy_(model.pos_embed.W_pos @ P)
    for lo, ln in zip(model.blocks, new.blocks):
        ln.attn.W_Q.copy_(t.einsum("ij,hjk->hik", Pt, lo.attn.W_Q))
        ln.attn.W_K.copy_(t.einsum("ij,hjk->hik", Pt, lo.attn.W_K))
        ln.attn.W_V.copy_(t.einsum("ij,hjk->hik", Pt, lo.attn.W_V))
        ln.attn.W_O.copy_(lo.attn.W_O @ P)
        ln.attn.b_Q.copy_(lo.attn.b_Q); ln.attn.b_K.copy_(lo.attn.b_K); ln.attn.b_V.copy_(lo.attn.b_V)
        ln.attn.b_O.copy_(lo.attn.b_O @ P)
        ln.mlp.W_in.copy_(Pt @ lo.mlp.W_in)
        ln.mlp.b_in.copy_(lo.mlp.b_in)
        ln.mlp.W_out.copy_(lo.mlp.W_out @ P)
        ln.mlp.b_out.copy_(lo.mlp.b_out @ P)
    new.unembed.W_U.copy_(Pt @ model.unembed.W_U)
    new.unembed.b_U.copy_(model.unembed.b_U)
    return new.eval()
