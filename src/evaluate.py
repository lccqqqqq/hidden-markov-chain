"""
Common yardstick for every compression method.

  uv run python src/evaluate.py models/cylinder_graph/<run_dir>

Reports test cross-entropy in nats/token (total and per context position), parameter
count, fp32 bytes, and two reference lines computed from the generating HMM:
  * Bayes-optimal loss at this context length (exact, via the mixed-state presentation on
    the same test set) — the floor any ctx-16 predictor can reach;
  * the entropy rate h_X (long-sequence limit of the same quantity).
The late-position losses are the ones that approach h_X; early positions are dominated by
the finite-context floor and are largely unaffected by compression.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch as t
import yaml
from transformer_lens import HookedTransformer

from hmm import HMM
from utils import create_process_from_dict, initialize_transformer_from_yaml


def load_run(run_dir: str | Path, ckpt: str = "best_model.pt", device: str = "cpu"):
    """Rebuild the model from the run's config.yaml and load the checkpoint. Returns (model, cfg)."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = initialize_transformer_from_yaml(None, model_cfg=cfg["model"])
    state = t.load(run_dir / ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), cfg


def load_test_data(cfg: dict, split: str = "test") -> t.Tensor:
    data_dir = cfg.get("data", {}).get("data_dir", cfg["data_generator"]["save_dir"])
    return t.load(os.path.join(data_dir, split, "observations.pt"))


@t.no_grad()
def evaluate(model: HookedTransformer, data: t.Tensor, batch_size: int = 1024, device: str = "cpu") -> dict:
    """Mean next-token cross-entropy (nats) over the dataset, total and per position."""
    model.eval()
    pos_sum = None
    n = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        nll = t.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).reshape(targets.shape)                       # (batch, seq)
        s = nll.sum(dim=0)
        pos_sum = s if pos_sum is None else pos_sum + s
        n += len(batch)
    per_pos = (pos_sum / n).cpu().numpy()
    return {"loss": float(per_pos.mean()), "loss_per_pos": per_pos.tolist()}


def bayes_optimal(process: HMM, data: t.Tensor) -> dict:
    """
    Exact optimal predictor on the same test windows: P(x_{t+1} | x_{1..t}) from the belief
    state after t observations, starting from the stationary distribution (matching how the
    windows were cut from the stationary process).
    """
    obs = data.cpu().numpy()                         # (N, seq+1)
    E = process.emission_matrices                    # (V, S, S)
    E_next = E.sum(axis=2)                           # (V, S): P(token j | state i)
    beliefs = process.mixed_state_presentation(obs[:, :-1])   # (N, seq, S): belief after x_1..x_t
    pred = np.einsum("nts,vs->ntv", beliefs, E_next)           # P(x_{t+1} | x_{1..t})
    targets = obs[:, 1:]
    p = np.take_along_axis(pred, targets[..., None], axis=2)[..., 0]
    nll = -np.log(p + 1e-300)
    per_pos = nll.mean(axis=0)
    return {"loss": float(per_pos.mean()), "loss_per_pos": per_pos.tolist()}


def entropy_rate(process: HMM, length: int = 200_000, burn_in: int = 1000, seed: int = 0) -> float:
    """h_X via the empirical estimator in hmm.py (belief propagation on one long sample)."""
    np.random.seed(seed)
    return process.entropy_rate_empirical_estimate(length, burn_in=burn_in)


def generator_dof(process: HMM) -> int:
    """Number of non-zero entries of the HMM's joint emission tensor: the parameter count
    of the exact predictor (transition + emission), i.e. the compression floor."""
    return int(np.count_nonzero(process.emission_matrices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir, args.ckpt, args.device)
    data = load_test_data(cfg)
    process = create_process_from_dict(cfg["data_generator"]["process"])

    from compress.base import count_bytes, count_params
    res = evaluate(model, data, device=args.device)
    bayes = bayes_optimal(process, data)
    h = entropy_rate(process)
    print(json.dumps({
        "params": count_params(model), "bytes_fp32": count_bytes(model, None),
        "hmm_dof": generator_dof(process), "hidden_states": process.num_hidden_states,
        "test_loss": res["loss"], "bayes_optimal": bayes["loss"], "entropy_rate": h,
        "loss_per_pos": [round(x, 4) for x in res["loss_per_pos"]],
        "bayes_per_pos": [round(x, 4) for x in bayes["loss_per_pos"]],
    }, indent=1))


if __name__ == "__main__":
    main()
