"""
CLI entry point for iterative linear probing and counterfactual importance analysis.

Phase 1: Iterative probing — fit probes, extract subspaces, project out, repeat.
Phase 2: Counterfactual — mean-ablate each subspace via run_with_hooks, measure KL + ΔCE.

Usage:
    # Full pipeline
    python src/run_iterative_probe.py --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/

    # Probing only (skip counterfactual)
    python src/run_iterative_probe.py --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --skip-counterfactual

    # Custom parameters
    python src/run_iterative_probe.py --model-dir ... \
        --seq-length 10 --batch-size 10000 --max-iterations 10 --r2-threshold 0.05 --device cpu
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from activation_extraction import (
    get_device,
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from iterative_probe import (
    iterative_probe_all_layers,
    IterativeProbeResult,
)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def ablate_subspace(
    actvs: t.Tensor,
    subspace: t.Tensor,
    method: str = "mean",
    position: int | None = -1,
) -> t.Tensor:
    """
    Mean- or zero-ablate a subspace from activations.

    Generalisation of archive/old_src/ablation.py:ablate_direction to
    multi-dimensional subspaces.

    Args:
        actvs: (batch, seq_len, d_model)
        subspace: (rank, d_model) orthonormal rows
        method: "mean" or "zero"
        position: sequence position to ablate (-1 = last, None = all)

    Returns:
        Modified activations (in-place for the selected position).
    """
    # subspace: (rank, d_model)
    if position is None:
        # Project onto subspace: (batch, seq_len, rank)
        proj_coeffs = t.einsum("bsd,rd->bsr", actvs, subspace)
        # Reconstruction in subspace: (batch, seq_len, d_model)
        proj = t.einsum("bsr,rd->bsd", proj_coeffs, subspace)
        if method == "mean":
            proj_mean = proj.mean(dim=0, keepdim=True)  # (1, seq_len, d_model)
            actvs = actvs - proj + proj_mean
        elif method == "zero":
            actvs = actvs - proj
        else:
            raise ValueError(f"Unknown ablation method: {method}")
    else:
        pos_actvs = actvs[:, position, :]  # (batch, d_model)
        proj_coeffs = pos_actvs @ subspace.T  # (batch, rank)
        proj = proj_coeffs @ subspace  # (batch, d_model)
        if method == "mean":
            proj_mean = proj.mean(dim=0, keepdim=True)  # (1, d_model)
            actvs[:, position, :] = pos_actvs - proj + proj_mean
        elif method == "zero":
            actvs[:, position, :] = pos_actvs - proj
        else:
            raise ValueError(f"Unknown ablation method: {method}")

    return actvs


def compute_counterfactual(
    model: HookedTransformer,
    sequences: t.Tensor,
    clean_log_probs: t.Tensor,
    clean_ce: float,
    layer: int,
    subspace: np.ndarray,
    device: str,
    method: str = "mean",
    position: int = -1,
    inference_batch_size: int | None = None,
) -> dict:
    """
    Measure the causal importance of a subspace via mean ablation.

    Args:
        model: HookedTransformer in eval mode
        sequences: (batch, seq_len) token sequences
        clean_log_probs: (batch, vocab) log-softmax of clean model output at last position
        clean_ce: scalar, mean cross-entropy of clean model on these sequences
        layer: which layer's resid_post to intervene on
        subspace: (rank, d_model) numpy array, orthonormal rows
        device: device string
        method: "mean" or "zero"
        position: position to ablate (-1 = last)
        inference_batch_size: process in chunks if set

    Returns:
        dict with mean_kl, std_kl, mean_delta_ce, std_delta_ce
    """
    subspace_t = t.from_numpy(subspace).float().to(device)

    def hook_fn(resid, hook):
        return ablate_subspace(resid, subspace_t, method=method, position=position)

    hook_spec = [(f"blocks.{layer}.hook_resid_post", hook_fn)]

    if inference_batch_size is None or inference_batch_size >= sequences.shape[0]:
        chunk_ranges = [(0, sequences.shape[0])]
    else:
        chunk_ranges = [
            (i, min(i + inference_batch_size, sequences.shape[0]))
            for i in range(0, sequences.shape[0], inference_batch_size)
        ]

    all_kl = []
    all_ce = []

    for start, end in chunk_ranges:
        batch_seqs = sequences[start:end].to(device)
        batch_clean_lp = clean_log_probs[start:end].to(device)

        with t.no_grad():
            ablated_logits = model.run_with_hooks(batch_seqs, fwd_hooks=hook_spec)

        # Last position logits
        abl_log_probs = F.log_softmax(ablated_logits[:, -1, :], dim=-1)

        # KL(clean || ablated) = sum_x p_clean(x) * [log p_clean(x) - log p_abl(x)]
        kl = F.kl_div(
            abl_log_probs, batch_clean_lp,
            reduction="none", log_target=True,
        ).sum(dim=-1)  # (batch,)

        # CE of ablated model w.r.t. clean distribution
        # CE = -sum_x p_clean(x) * log p_abl(x)
        ce_abl = -(batch_clean_lp.exp() * abl_log_probs).sum(dim=-1)  # (batch,)

        all_kl.append(kl.cpu())
        all_ce.append(ce_abl.cpu())

    all_kl = t.cat(all_kl)
    all_ce = t.cat(all_ce)

    return {
        "mean_kl": float(all_kl.mean()),
        "std_kl": float(all_kl.std()),
        "mean_delta_ce": float(all_ce.mean() - clean_ce),
        "std_delta_ce": float(all_ce.std()),
    }


# ---------------------------------------------------------------------------
# Phase 1: Iterative probing
# ---------------------------------------------------------------------------

def run_phase1(
    model_dir: str,
    batch_size: int,
    seq_length: int,
    max_iterations: int,
    r2_threshold: float,
    test_fraction: float,
    device: str,
    inference_batch_size: int | None,
    output_dir: str,
) -> tuple[list[IterativeProbeResult], HookedTransformer, t.Tensor, np.ndarray]:
    """
    Run iterative probing (Phase 1).

    Returns:
        (results, model, sequences, belief_states) — model kept alive for Phase 2.
    """
    # Load config
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_n_ctx = config["model"]["n_ctx"]

    # Generate data
    process = get_process_from_config(config)
    print(f"Generating {batch_size} sequences of length {seq_length} "
          f"from {process.__class__.__name__}")
    sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
    belief_states = process.mixed_state_presentation(sequences.numpy())

    # Truncate to model context window
    if model_n_ctx < seq_length:
        print(f"Truncating sequences from {seq_length} to {model_n_ctx} (model n_ctx)")
        sequences = sequences[:, :model_n_ctx]
        belief_states = belief_states[:, :model_n_ctx, :]

    # Load model and extract activations
    print(f"Loading model from {model_dir}")
    model = load_model_from_dir(model_dir, device=device)

    print("Extracting residual stream activations...")
    raw_actvs = extract_residual_stream(
        model, sequences, device=device, batch_size=inference_batch_size,
    )

    # Prepare probe data (last position)
    layer_actvs, targets = prepare_probe_data(raw_actvs, belief_states, use_last_pos=True)
    del raw_actvs

    # Run iterative probing
    print(f"\nIterative probing (max_iter={max_iterations}, R² threshold={r2_threshold}):")
    results = iterative_probe_all_layers(
        layer_actvs, targets,
        max_iterations=max_iterations,
        r2_threshold=r2_threshold,
        test_fraction=test_fraction,
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    _save_iterative_metrics(results, output_dir)
    _save_subspaces(results, output_dir)

    print(f"\nPhase 1 results saved to {output_dir}")
    return results, model, sequences, belief_states


def _save_iterative_metrics(
    results: list[IterativeProbeResult],
    output_dir: str,
) -> None:
    """Save iterative probing metrics to CSV."""
    rows = []
    for lr in results:
        cum_dims = 0
        for i, it in enumerate(lr.iterations):
            r = it.probe_result
            rank = it.subspace.shape[0]
            cum_dims += rank
            rows.append({
                "layer": lr.layer,
                "iteration": i,
                "train_r2": r.train_r2,
                "test_r2": r.test_r2,
                "train_mse": r.train_mse,
                "test_mse": r.test_mse,
                "subspace_rank": rank,
                "cumulative_dims": cum_dims,
                "singular_values": ",".join(f"{s:.6f}" for s in it.singular_values),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "iterative_metrics.csv"), index=False)


def _save_subspaces(
    results: list[IterativeProbeResult],
    output_dir: str,
) -> None:
    """Save subspaces to npz."""
    data = {}
    for lr in results:
        for i, it in enumerate(lr.iterations):
            data[f"layer{lr.layer}_iter{i}"] = it.subspace
            data[f"layer{lr.layer}_iter{i}_S"] = it.singular_values
    np.savez_compressed(os.path.join(output_dir, "subspaces.npz"), **data)


# ---------------------------------------------------------------------------
# Phase 2: Counterfactual importance
# ---------------------------------------------------------------------------

def run_phase2(
    results: list[IterativeProbeResult],
    model: HookedTransformer,
    sequences: t.Tensor,
    device: str,
    output_dir: str,
    inference_batch_size: int | None = None,
) -> None:
    """
    Run counterfactual ablation experiments (Phase 2).

    For each layer and iteration subspace, mean-ablate and measure KL + ΔCE.
    Also ablate individual directions within each subspace.
    """
    sequences_device = sequences.to(device)

    # Get clean model outputs
    print("\nComputing clean model outputs...")
    with t.no_grad():
        clean_logits = model(sequences_device)
    clean_log_probs = F.log_softmax(clean_logits[:, -1, :], dim=-1).cpu()

    # Clean CE (entropy of clean distribution — serves as baseline)
    clean_ce = float(-(clean_log_probs.exp() * clean_log_probs).sum(dim=-1).mean())

    del clean_logits
    if t.cuda.is_available():
        t.cuda.empty_cache()

    print(f"Clean model entropy at last position: {clean_ce:.4f} nats")

    # Counterfactual experiments
    rows = []
    for lr in results:
        if lr.n_significant == 0:
            continue

        print(f"\nLayer {lr.layer} ({lr.n_significant} iterations, "
              f"{lr.cumulative_dims} total dims):")

        for i, it in enumerate(lr.iterations):
            # Ablate full subspace for this iteration
            print(f"  Iter {i} (rank {it.subspace.shape[0]}): ", end="", flush=True)
            cf = compute_counterfactual(
                model, sequences, clean_log_probs, clean_ce,
                layer=lr.layer, subspace=it.subspace, device=device,
                inference_batch_size=inference_batch_size,
            )
            print(f"KL={cf['mean_kl']:.4f} ± {cf['std_kl']:.4f}, "
                  f"ΔCE={cf['mean_delta_ce']:.4f}")
            rows.append({
                "layer": lr.layer,
                "iteration": i,
                "ablation_type": "subspace",
                "direction_idx": -1,
                **cf,
            })

            # Ablate individual directions within this subspace
            if it.subspace.shape[0] > 1:
                for d_idx in range(it.subspace.shape[0]):
                    direction = it.subspace[d_idx:d_idx + 1, :]  # (1, d_model)
                    cf_dir = compute_counterfactual(
                        model, sequences, clean_log_probs, clean_ce,
                        layer=lr.layer, subspace=direction, device=device,
                        inference_batch_size=inference_batch_size,
                    )
                    print(f"    dir {d_idx}: KL={cf_dir['mean_kl']:.4f}, "
                          f"ΔCE={cf_dir['mean_delta_ce']:.4f}")
                    rows.append({
                        "layer": lr.layer,
                        "iteration": i,
                        "ablation_type": f"direction_{d_idx}",
                        "direction_idx": d_idx,
                        **cf_dir,
                    })

        # Also ablate cumulative subspace (all iterations combined)
        full_sub = lr.full_subspace
        if full_sub.shape[0] > 0 and lr.n_significant > 1:
            print(f"  Cumulative ({full_sub.shape[0]} dims): ", end="", flush=True)
            cf_cum = compute_counterfactual(
                model, sequences, clean_log_probs, clean_ce,
                layer=lr.layer, subspace=full_sub, device=device,
                inference_batch_size=inference_batch_size,
            )
            print(f"KL={cf_cum['mean_kl']:.4f} ± {cf_cum['std_kl']:.4f}, "
                  f"ΔCE={cf_cum['mean_delta_ce']:.4f}")
            rows.append({
                "layer": lr.layer,
                "iteration": -1,
                "ablation_type": "cumulative",
                "direction_idx": -1,
                **cf_cum,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "counterfactual_metrics.csv"), index=False)
    print(f"\nPhase 2 results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Iterative linear probing and counterfactual importance analysis"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Model directory with config.yaml and checkpoint")
    parser.add_argument("--seq-length", type=int, default=10,
                        help="Sequence length for generated data")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Number of sequences to generate")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max probe-project iterations per layer")
    parser.add_argument("--r2-threshold", type=float, default=0.05,
                        help="Minimum test R² to continue iterating")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction held out for probe testing")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/mps); auto-detected if omitted")
    parser.add_argument("--inference-batch-size", type=int, default=None,
                        help="Batch size for model inference (memory control)")
    parser.add_argument("--skip-counterfactual", action="store_true",
                        help="Run only Phase 1 (iterative probing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory; defaults to {model_dir}/iterative_probe_results/")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")

    args = parser.parse_args()

    if args.device is None:
        args.device = get_device()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "iterative_probe_results")

    # Check skip-if-done
    metrics_path = os.path.join(args.output_dir, "iterative_metrics.csv")
    if os.path.exists(metrics_path) and not args.force:
        print(f"Results already exist at {args.output_dir}. Use --force to re-run.")
        sys.exit(0)

    print(f"Device: {args.device}")
    print(f"Model: {args.model_dir}")

    # Phase 1: Iterative probing
    results, model, sequences, belief_states = run_phase1(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_iterations=args.max_iterations,
        r2_threshold=args.r2_threshold,
        test_fraction=args.test_fraction,
        device=args.device,
        inference_batch_size=args.inference_batch_size,
        output_dir=args.output_dir,
    )

    # Summary
    print("\n=== Summary ===")
    for lr in results:
        print(f"Layer {lr.layer}: {lr.n_significant} copies, "
              f"{lr.cumulative_dims} dims used")

    # Phase 2: Counterfactual importance
    if args.skip_counterfactual:
        print("\nSkipping counterfactual phase (--skip-counterfactual).")
    else:
        run_phase2(
            results=results,
            model=model,
            sequences=sequences,
            device=args.device,
            output_dir=args.output_dir,
            inference_batch_size=args.inference_batch_size,
        )

    # Cleanup
    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
