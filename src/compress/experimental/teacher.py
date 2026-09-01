"""
Exact-teacher loss (E0 of notes/compression/singular_geometry_evaluation.md).

The generating HMM gives the exact next-token distribution P_HMM(x_{t+1} | x_{<=t}) for any
context via the mixed-state presentation, so the model can be scored against the
*population* conditional instead of a sampled one-hot label:

    K(w) = E_ctx [ KL( P_HMM(.|ctx) || P_w(.|ctx) ) ]  =  softCE - H(P_HMM)

Contexts are still sampled (from the train split for calibration, the test split for
reporting); only the label noise is removed. H(P_HMM) is a constant per dataset, so KL and
soft cross-entropy differ by a constant and have identical gradients/Hessians.

Everything here is CPU/GPU agnostic and used by the geometry experiments only; the
reference quantizers keep the one-hot protocol.
"""
from __future__ import annotations

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from hmm import HMM


def teacher_probs(process: HMM, tokens: t.Tensor) -> t.Tensor:
    """tokens: (N, seq+1) windows. Returns P(x_{t+1} | x_{1..t}) for t = 1..seq as
    a float32 tensor of shape (N, seq, V), aligned with model(tokens[:, :-1]) logits."""
    obs = tokens[:, :-1].cpu().numpy()
    E = process.emission_matrices                       # (V, S, S)
    E_next = E.sum(axis=2)                              # (V, S): P(token j | state i)
    beliefs = process.mixed_state_presentation(obs)     # (N, seq, S)
    pred = np.einsum("nts,vs->ntv", beliefs, E_next)
    return t.from_numpy(pred.astype(np.float32))


def teacher_entropy(probs: t.Tensor) -> t.Tensor:
    """Per-position H(P_HMM) averaged over sequences, shape (seq,)."""
    return -(probs * t.log(probs.clamp_min(1e-30))).sum(-1).mean(0)


def soft_ce(logits: t.Tensor, probs: t.Tensor, reduction: str = "mean") -> t.Tensor:
    """-sum_v p_v log q_v per (n, t); reduced over everything by default."""
    logq = t.log_softmax(logits, dim=-1)
    ce = -(probs * logq).sum(-1)
    return ce.mean() if reduction == "mean" else ce


def kl_loss(model: HookedTransformer, tokens: t.Tensor, probs: t.Tensor) -> t.Tensor:
    """Differentiable mean KL(P_HMM || P_model) over a batch (constant offset = teacher
    entropy removed, so the value is >= 0 and 0 iff the model is Bayes-optimal on the batch)."""
    device = next(model.parameters()).device
    logits = model(tokens[:, :-1].to(device))
    p = probs.to(device)
    return soft_ce(logits, p) - (-(p * t.log(p.clamp_min(1e-30))).sum(-1).mean())


@t.no_grad()
def evaluate_teacher(model: HookedTransformer, tokens: t.Tensor, probs: t.Tensor,
                     batch_size: int = 2048) -> dict:
    """Both scores on the same windows: one-hot NLL (the log's convention) and exact KL.
    Returns {'nll', 'kl', 'nll_per_pos', 'kl_per_pos'} (nats/token)."""
    model.eval()
    device = next(model.parameters()).device
    nll_sum = kl_sum = None
    n = 0
    for i in range(0, len(tokens), batch_size):
        b = tokens[i:i + batch_size].to(device)
        p = probs[i:i + batch_size].to(device)
        logits = model(b[:, :-1])
        logq = t.log_softmax(logits, dim=-1)
        nll = -logq.gather(-1, b[:, 1:, None])[..., 0]                   # (B, seq)
        kl = (p * (t.log(p.clamp_min(1e-30)) - logq)).sum(-1)            # (B, seq)
        nll_sum = nll.sum(0) if nll_sum is None else nll_sum + nll.sum(0)
        kl_sum = kl.sum(0) if kl_sum is None else kl_sum + kl.sum(0)
        n += len(b)
    nll_pp, kl_pp = (nll_sum / n).cpu(), (kl_sum / n).cpu()
    return {"nll": float(nll_pp.mean()), "kl": float(kl_pp.mean()),
            "nll_per_pos": nll_pp.tolist(), "kl_per_pos": kl_pp.tolist()}


def calib_split(train: t.Tensor, n: int, seed: int = 0) -> t.Tensor:
    g = t.Generator().manual_seed(seed)
    return train[t.randperm(len(train), generator=g)[:n]]
