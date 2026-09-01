"""
GPTQ — Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023 (arXiv:2210.17323).

Faithful to the reference algorithm:
  * Layerwise objective ‖W X − Ŵ X‖², H = 2 X Xᵀ from calibration inputs to each matrix.
  * Sequential propagation: blocks are quantized in order and calibration activations for
    block l are produced by the already-quantized blocks < l (and, within a block, by the
    already-quantized attention when calibrating the MLP), as in the reference code.
  * Dampening H += damp · mean(diag H) · I, damp = 0.01.
  * Columns in fixed index order (act_order=False) or descending diag(H) (act_order=True).
  * OBS compensation via the upper Cholesky factor of H⁻¹. The reference lazy-batches the
    trailing update in blocks of 128 columns for GPU efficiency; with d_in ≤ 256 we apply
    the mathematically identical unbatched update.
  * Grid: RTN's per-output-channel min/max computed from the original W (same
    `Quantizer.grid` as RTN); calibration n_calib = 128 sequences.
  * Dead inputs (diag(H) = 0): H_jj := 1 and W[:, j] := 0, as in the reference.
  * W_E and W_pos have one-hot inputs, so H is diagonal and the OBS compensation vanishes:
    GPTQ degenerates *exactly* to RTN there, which is what we apply. (The paper does not
    quantize embeddings at all; quantizing them is part of this repo's shared protocol.)

Matrix orientation: GPTQ works on W of shape (d_out, d_in) with H of shape (d_in, d_in).
TL shapes are mapped as
    W_Q/K/V (head, d_model, d_head) -> (head*d_head, d_model), X = resid_pre
    W_O     (head, d_head, d_model) -> (d_model, head*d_head), X = attn.hook_z flattened
    W_in    (d_model, d_mlp)        -> (d_mlp, d_model),       X = resid_mid
    W_out   (d_mlp, d_model)        -> (d_model, d_mlp),       X = mlp.hook_post
    W_U     (d_model, d_vocab)      -> (d_vocab, d_model),     X = last resid_post
(no LayerNorm in these models, so resid_pre/mid feed the matrices directly).
"""
from __future__ import annotations

import copy

import torch as t
from transformer_lens import HookedTransformer

from compress.base import Quantizer
from compress.rtn import RTN


class GPTQ(Quantizer):
    name = "GPTQ"
    citation = "Frantar et al., ICLR 2023, arXiv:2210.17323"

    def __init__(self, bits: int, granularity: str = "per_channel", symmetric: bool = False,
                 n_calib: int = 128, damp: float = 0.01, act_order: bool = False, blocksize: int = 128):
        super().__init__(bits, granularity, symmetric)
        self.n_calib, self.damp, self.act_order, self.blocksize = n_calib, damp, act_order, blocksize

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(n_calib=self.n_calib, damp=self.damp, act_order=self.act_order)
        return s

    # ---- core: quantize one (d_out, d_in) matrix against H ------------------------------
    def _gptq_matrix(self, W: t.Tensor, H: t.Tensor, bits: int | None = None) -> t.Tensor:
        W = W.clone()
        d_out, d_in = W.shape
        H = H.clone()

        dead = t.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        if self.act_order:
            perm = t.argsort(t.diag(H), descending=True)
            W, H = W[:, perm], H[perm][:, perm]

        # grid from the ORIGINAL weights, per output row (or per tensor)
        w2 = W.T if self.granularity == "per_channel" else W.reshape(-1, 1)
        scale, zero, qmin, qmax = self.grid(w2, bits if bits is not None else self.bits, self.symmetric)
        if self.granularity == "per_tensor":
            scale, zero = scale.expand(d_out).clone(), zero.expand(d_out).clone()

        H += self.damp * t.diag(H).mean() * t.eye(d_in, device=H.device)
        Hinv = t.linalg.cholesky(H)
        Hinv = t.cholesky_inverse(Hinv)
        Hinv = t.linalg.cholesky(Hinv, upper=True)   # upper Cholesky of H^-1

        for j in range(d_in):
            w = W[:, j]
            q = t.clamp(t.round(w / scale) + zero, qmin, qmax)
            wq = (q - zero) * scale
            err = (w - wq) / Hinv[j, j]
            W[:, j] = wq
            if j + 1 < d_in:
                W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:][None, :]

        if self.act_order:
            inv = t.argsort(perm)
            W = W[:, inv]
        return W

    @staticmethod
    def _hessian(X: t.Tensor) -> t.Tensor:
        """H = 2 X Xᵀ for X of shape (n_samples, d_in)."""
        X = X.reshape(-1, X.shape[-1]).float()
        return 2.0 * X.T @ X

    # ---- model traversal -----------------------------------------------------------------
    @t.no_grad()
    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None, "GPTQ needs calibration sequences (train split)"
        model = copy.deepcopy(model)
        device = next(model.parameters()).device
        tokens = calib[:, :-1].to(device)

        def acts(hook_name: str) -> t.Tensor:
            store = {}
            model.run_with_hooks(
                tokens, return_type=None,
                fwd_hooks=[(hook_name, lambda a, hook: store.update(x=a.detach()))],
            )
            return store["x"]

        # Embeddings: H diagonal -> exactly RTN (see module docstring).
        rtn = RTN(self.bits, self.granularity, self.symmetric)
        for name in ("embed.W_E", "pos_embed.W_pos"):
            p = dict(model.named_parameters())[name]
            w2 = self._channel_view(p.data, self.granularity, name)
            s_, z_, lo, hi = self.grid(w2, self._bits(name), self.symmetric)
            p.data.copy_(self._from_channel_view(self.fake_quant(w2, s_, z_, lo, hi), p.shape, name))

        n_layers = model.cfg.n_layers
        for l in range(n_layers):
            blk = model.blocks[l]
            # stage 1: W_Q, W_K, W_V share input resid_pre
            X = acts(f"blocks.{l}.hook_resid_pre")
            H = self._hessian(X)
            for wname in ("W_Q", "W_K", "W_V"):
                w = getattr(blk.attn, wname)                        # (head, d_model, d_head)
                W2 = w.data.permute(0, 2, 1).reshape(-1, w.shape[1])  # (head*d_head, d_model)
                W2 = self._gptq_matrix(W2, H, self._bits(f"blocks.{l}.attn.{wname}"))
                w.data.copy_(W2.reshape(w.shape[0], w.shape[2], w.shape[1]).permute(0, 2, 1))
            # stage 2: W_O, input z (recomputed with quantized Q/K/V)
            Z = acts(f"blocks.{l}.attn.hook_z")                     # (b, p, head, d_head)
            H = self._hessian(Z.reshape(-1, Z.shape[-2] * Z.shape[-1]))
            w = blk.attn.W_O                                        # (head, d_head, d_model)
            W2 = w.data.reshape(-1, w.shape[-1]).T                  # (d_model, head*d_head)
            w.data.copy_(self._gptq_matrix(W2, H, self._bits(f"blocks.{l}.attn.W_O")).T.reshape(w.shape))
            # stage 3: W_in, input resid_mid
            H = self._hessian(acts(f"blocks.{l}.hook_resid_mid"))
            w = blk.mlp.W_in                                        # (d_model, d_mlp)
            w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits(f"blocks.{l}.mlp.W_in")).T)
            # stage 4: W_out, input mlp post-activation
            H = self._hessian(acts(f"blocks.{l}.mlp.hook_post"))
            w = blk.mlp.W_out                                       # (d_mlp, d_model)
            w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits(f"blocks.{l}.mlp.W_out")).T)

        # unembedding: input = final residual stream
        H = self._hessian(acts(f"blocks.{n_layers - 1}.hook_resid_post"))
        w = model.unembed.W_U                                       # (d_model, d_vocab)
        w.data.copy_(self._gptq_matrix(w.data.T, H, self._bits("unembed.W_U")).T)
        return model
