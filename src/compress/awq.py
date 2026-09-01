"""
AWQ — Lin et al., MLSys 2024 (arXiv:2306.00978).

Reference algorithm: per-input-channel scaling W <- diag(s) W (rows = input channels),
s_i = (mean |x_i|)^alpha, alpha grid-searched per layer on the reconstruction error
||X W - X diag(1/s) Q(diag(s) W)||^2, then RTN; diag(1/s) is folded into the preceding op
so the network function is unchanged pre-quantization. Weight-only; landscape info is
diag(XX^T) only (no inverse-Hessian compensation) — the "poor man's GPTQ" control.

Architecture adaptation (documented, forced by no-LayerNorm): the paper folds 1/s for
Q/K/V/up-proj into the preceding LayerNorm weights. This model has no LN, so scaling is
applied only where a fold target exists, matching the paper's own practice of scaling
only layers with foldable predecessors:
  * W_O   (input z):        1/s folded into W_V, b_V (attention is linear in V);
  * W_out (input ReLU(...)): 1/s folded into W_in cols and b_in (ReLU is positively
    homogeneous and s > 0).
W_Q/K/V, W_in, W_U, W_E, W_pos have residual-stream/one-hot inputs with no fold target and
receive plain RTN (the paper's treatment of unscaled layers). Scale search runs on fp32
activations first; all weights are then RTN-quantized per output channel.
"""
from __future__ import annotations

import copy

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer, is_quantized_param


class AWQ(Quantizer):
    name = "AWQ"
    citation = "Lin et al., MLSys 2024, arXiv:2306.00978"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 alpha_grid: int = 20, n_calib: int = 128):
        super().__init__(bits, granularity, symmetric)
        self.alpha_grid, self.n_calib = alpha_grid, n_calib

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(alpha_grid=self.alpha_grid, n_calib=self.n_calib)
        return s

    def _rtn2d(self, W: t.Tensor) -> t.Tensor:
        """RTN of a (d_in, d_out) matrix, per output channel (paper's grid)."""
        s_, z_, lo, hi = self.grid(W, self.bits, self.symmetric)
        return self.fake_quant(W, s_, z_, lo, hi)

    def _best_scale(self, X: t.Tensor, W: t.Tensor) -> t.Tensor:
        """X: (n, d_in), W: (d_in, d_out). Returns s (d_in,) minimising recon error."""
        sx = X.abs().mean(dim=0).clamp_min(1e-8)
        ref = X @ W
        best, best_err = t.ones_like(sx), float("inf")
        for k in range(self.alpha_grid + 1):
            alpha = k / self.alpha_grid
            s = sx ** alpha
            err = ((X / s) @ self._rtn2d(s[:, None] * W) - ref).pow(2).sum().item()
            if err < best_err:
                best, best_err = s, err
        return best

    @t.no_grad()
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None, "AWQ needs calibration sequences"
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        tokens = calib[: self.n_calib, :-1].to(device)

        def acts(hook_name):
            store = {}
            model.run_with_hooks(tokens, return_type=None,
                                 fwd_hooks=[(hook_name, lambda a, hook: store.update(x=a.detach()))])
            return store["x"]

        for l in range(model.cfg.n_layers):
            blk = model.blocks[l]
            # W_O: scale by z-magnitudes, fold 1/s into W_V (+ b_V)
            Z = acts(f"blocks.{l}.attn.hook_z").reshape(-1, model.cfg.n_heads * model.cfg.d_head)
            WO2 = blk.attn.W_O.data.reshape(-1, model.cfg.d_model)          # (h*dh, d_model)
            s = self._best_scale(Z, WO2)
            blk.attn.W_O.data.copy_((s[:, None] * WO2).reshape(blk.attn.W_O.shape))
            sv = s.reshape(model.cfg.n_heads, model.cfg.d_head)
            blk.attn.W_V.data /= sv[:, None, :]
            blk.attn.b_V.data /= sv
            # W_out: scale by ReLU-output magnitudes, fold 1/s into W_in cols (+ b_in)
            Hp = acts(f"blocks.{l}.mlp.hook_post").reshape(-1, model.cfg.d_mlp)
            s = self._best_scale(Hp, blk.mlp.W_out.data)                    # (d_mlp,)
            blk.mlp.W_out.data.copy_(s[:, None] * blk.mlp.W_out.data)
            blk.mlp.W_in.data /= s[None, :]
            blk.mlp.b_in.data /= s

        # RTN everything (scaled weights included), reference per-output-channel grid
        for name, p in model.named_parameters():
            if not is_quantized_param(name):
                continue
            w2 = self._channel_view(p.data, self.granularity, name)
            s_, z_, lo, hi = self.grid(w2, self.bits, self.symmetric)
            p.data.copy_(self._from_channel_view(self.fake_quant(w2, s_, z_, lo, hi), p.shape, name))
        return model
