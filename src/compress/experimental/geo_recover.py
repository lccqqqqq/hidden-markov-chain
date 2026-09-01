"""
Shared short-recovery loop for the structural (compile) experiments E8/E9.

Same settings as the STE-QAT loop in compress/qat.py: Adam, lr 1e-4, batch 128, no
scheduler / weight decay, fixed seed. `steps=None` = one full epoch over `train`.
Loss: one-hot cross-entropy (default) or exact-teacher KL (soft_targets=True).
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from compress.experimental.teacher import kl_loss, teacher_probs


def recover(model: HookedTransformer, train: t.Tensor, process, steps: int | None,
            lr: float = 1e-4, batch_size: int = 128, seed: int = 0,
            soft_targets: bool = False) -> HookedTransformer:
    """Fine-tunes `model` in place; returns it in eval mode. steps=0 -> no-op."""
    if steps == 0:
        return model.eval()
    device = next(model.parameters()).device
    # deduplicate tied parameters (E9): Adam must see each Parameter once
    seen, params = set(), []
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p)); params.append(p)
    opt = t.optim.Adam(params, lr=lr)
    g = t.Generator().manual_seed(seed)
    model.train()
    step = 0
    done = False
    while not done:
        for idx in t.randperm(len(train), generator=g).split(batch_size):
            batch = train[idx].to(device)
            if soft_targets:
                loss = kl_loss(model, batch, teacher_probs(process, batch))
            else:
                logits = model(batch[:, :-1])
                loss = t.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if steps is not None and step >= steps:
                done = True; break
        else:
            done = True          # one full epoch when steps is None
    return model.eval()


def recovery_label(steps: int | None) -> str:
    return "1epoch" if steps is None else f"{steps}steps"
