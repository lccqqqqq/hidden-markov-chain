"""
E9 — layer tying (notes/compression/singular_geometry_evaluation.md §4, Tier 4;
survey §2.5: the HMM belief update is one recurrent map, so 4 distinct blocks may be
replaceable by one block applied 4 times).

Implementation: a standard HookedTransformer with n_apply blocks whose block k shares the
nn.Parameter objects of block (k mod n_shared). Sharing is done by re-assigning the same
Parameter into the later blocks' modules, so TransformerLens' forward is untouched;
hooks/caching still see n_apply distinct blocks. Optimizers must deduplicate parameters
(geo_recover.recover does) and param counts must count unique tensors (unique_params).
"""
from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer

from utils import initialize_transformer_from_yaml

BLOCK_PARAMS = ("attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O", "attn.b_Q", "attn.b_K", "attn.b_V", "attn.b_O",
                "mlp.W_in", "mlp.b_in", "mlp.W_out", "mlp.b_out")


def _get(block, dotted: str):
    mod, attr = dotted.split(".")
    return getattr(getattr(block, mod), attr)


def _set(block, dotted: str, param: t.nn.Parameter):
    mod, attr = dotted.split(".")
    setattr(getattr(block, mod), attr, param)


def unique_params(model: t.nn.Module) -> int:
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); n += p.numel()
    return n


def unique_bytes_fp32(model: t.nn.Module) -> int:
    return unique_params(model) * 4


@t.no_grad()
def build_tied(model: HookedTransformer, model_cfg: dict, n_shared: int, n_apply: int,
               init: str = "mean", source_layer: int | None = None) -> HookedTransformer:
    """
    init: "mean"   -> shared block j = mean over trained layers {l : l % n_shared == j}
          "layer0" -> shared block j = trained layer j
          "single" -> (n_shared == 1) shared block = trained layer `source_layer`
    """
    assert n_apply % n_shared == 0
    cfg = dict(model_cfg); cfg["n_layer"] = n_apply
    new = initialize_transformer_from_yaml(None, model_cfg=cfg).to(next(model.parameters()).device)
    new.embed.W_E.copy_(model.embed.W_E); new.pos_embed.W_pos.copy_(model.pos_embed.W_pos)
    new.unembed.W_U.copy_(model.unembed.W_U); new.unembed.b_U.copy_(model.unembed.b_U)
    L = model.cfg.n_layers
    for j in range(n_shared):
        if init == "mean":
            src = [l for l in range(L) if l % n_shared == j]
        elif init == "layer0":
            src = [j]
        elif init == "single":
            assert n_shared == 1 and source_layer is not None
            src = [source_layer]
        else:
            raise ValueError(init)
        for name in BLOCK_PARAMS:
            val = t.stack([_get(model.blocks[l], name).data for l in src]).mean(0)
            _get(new.blocks[j], name).data.copy_(val)
    # tie: block k reuses the Parameter objects of block k mod n_shared
    for k in range(n_shared, n_apply):
        for name in BLOCK_PARAMS:
            _set(new.blocks[k], name, _get(new.blocks[k % n_shared], name))
    return new.eval()


def check_tied(model: HookedTransformer, n_shared: int) -> bool:
    for k in range(n_shared, model.cfg.n_layers):
        for name in BLOCK_PARAMS:
            if _get(model.blocks[k], name) is not _get(model.blocks[k % n_shared], name):
                return False
    return True
