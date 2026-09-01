"""
Structured pruning of MLP neurons, with post-prune fine-tuning.

Two criteria (both defined exactly here; scores per neuron j of each layer, pruned
uniformly per layer, lowest scores first):
  * "magnitude":  s_j = sqrt(||W_in[:, j]||^2 + b_in[j]^2 + ||W_out[j, :]||^2)
    — the neuron's full weight-vector L2 norm; no data. The classic baseline.
  * "wanda":      s_j = ||h_j||_2 * ||W_out[j, :]||_1
    with ||h_j||_2 the RMS of the neuron's post-ReLU activation over calibration tokens —
    Wanda's |W| * ||X|| importance (Sun et al. 2023, arXiv:2306.11695) summed over the
    W_out row, i.e. their structured (per-output-group) variant applied to whole neurons.
Pruning zeroes W_in[:, j], b_in[j], W_out[j, :], which is exactly equivalent to removing
the neuron (ReLU(0) = 0); the parameter count is reported analytically as if removed:
each dropped neuron saves 2*d_model + 1 parameters. Fine-tuning trains all remaining
weights with the pruned neurons frozen at zero (fixed 0/1 mask parametrization), same
loop settings as STE-QAT (1 epoch, Adam lr 1e-4, batch 128, seed 0).
"""
from __future__ import annotations

import copy

import torch as t
from torch.nn.utils import parametrize
from transformer_lens import HookedTransformer

from compress.base import Quantizer


class _Masked(t.nn.Module):
    def __init__(self, mask: t.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, w: t.Tensor) -> t.Tensor:
        return w * self.mask


class NeuronPrune(Quantizer):
    citation = "magnitude: classic baseline; wanda: Sun et al. 2023, arXiv:2306.11695"

    def __init__(self, fraction: float, criterion: str = "magnitude", epochs: int = 1,
                 lr: float = 1e-4, batch_size: int = 128, seed: int = 0, n_calib: int = 128):
        super().__init__(bits=16)   # bits unused; parameter-count method
        assert 0 < fraction < 1 and criterion in ("magnitude", "wanda")
        self.fraction, self.criterion, self.epochs = fraction, criterion, epochs
        self.lr, self.batch_size, self.seed, self.n_calib = lr, batch_size, seed, n_calib
        self.name = f"{criterion}-prune"
        self.dropped: dict[int, int] = {}

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(fraction=self.fraction, criterion=self.criterion, epochs=self.epochs,
                       n_calib=self.n_calib if self.criterion == "wanda" else None)
        return s

    def params_after(self, model: HookedTransformer) -> int:
        total = sum(p.numel() for p in model.parameters())
        return total - sum(n * (2 * model.cfg.d_model + 1) for n in self.dropped.values())

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        act_rms = {}
        if self.criterion == "wanda":
            assert calib is not None, "wanda criterion needs calibration sequences"
            tokens = calib[: self.n_calib, :-1].to(device)
            def make_hook(l):
                def hook(a, hook):
                    act_rms[l] = a.detach().reshape(-1, a.shape[-1]).pow(2).mean(dim=0).sqrt()
                return hook
            with t.no_grad():
                model.run_with_hooks(tokens, return_type=None,
                    fwd_hooks=[(f"blocks.{l}.mlp.hook_post", make_hook(l)) for l in range(model.cfg.n_layers)])

        n_drop = round(self.fraction * model.cfg.d_mlp)
        wrapped = []
        for l in range(model.cfg.n_layers):
            mlp = model.blocks[l].mlp
            if self.criterion == "magnitude":
                score = (mlp.W_in.data.pow(2).sum(0) + mlp.b_in.data.pow(2) + mlp.W_out.data.pow(2).sum(1)).sqrt()
            else:
                score = act_rms[l] * mlp.W_out.data.abs().sum(dim=1)
            drop = t.argsort(score)[:n_drop]
            self.dropped[l] = len(drop)
            keep_mask = t.ones(model.cfg.d_mlp, device=device)
            keep_mask[drop] = 0.0
            with t.no_grad():
                mlp.W_in.data *= keep_mask[None, :]
                mlp.b_in.data *= keep_mask
                mlp.W_out.data *= keep_mask[:, None]
            for attr, m in (("W_in", keep_mask[None, :]), ("b_in", keep_mask), ("W_out", keep_mask[:, None])):
                parametrize.register_parametrization(mlp, attr, _Masked(m))
                wrapped.append((mlp, attr))

        if self.epochs > 0:
            assert calib is not None and len(calib) > 1000, "fine-tuning needs the train split"
            opt = t.optim.Adam(model.parameters(), lr=self.lr)
            g = t.Generator().manual_seed(self.seed)
            model.train()
            for _ in range(self.epochs):
                for idx in t.randperm(len(calib), generator=g).split(self.batch_size):
                    batch = calib[idx].to(device)
                    logits = model(batch[:, :-1])
                    loss = t.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                    opt.zero_grad(); loss.backward(); opt.step()

        for mod, attr in wrapped:
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        return model.eval()
