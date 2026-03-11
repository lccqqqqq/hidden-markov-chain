"""
CLI pipeline for Direct Logit Attribution (DLA) analysis.

Decomposes logits into per-component contributions to understand which
network components produce the nonlinear correction beyond M_eff @ b.

Usage:
    python src/run_logit_attribution.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --batch-size 10000 --device cpu
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F

from activation_extraction import (
    get_device,
    load_model_from_dir,
    get_process_from_config,
)
from unembed_analysis import compute_observation_matrix
from logit_attribution import (
    component_names_for_n_layers,
    extract_components,
    verify_additivity,
    direct_logit_attribution,
    ln_corrected_logit_attribution,
    verify_ln_corrected_sum,
    component_statistics,
    logit_lens,
)


# ---------------------------------------------------------------------------
# Phase 0: Data generation and Bayes-optimal computation
# ---------------------------------------------------------------------------

def setup_data(config, model, batch_size, seq_length, device):
    """Generate data, compute beliefs, Bayes-optimal logits, and delta."""
    process = get_process_from_config(config)
    print(f"Generating {batch_size} sequences of length {seq_length} "
          f"from {process.__class__.__name__}")

    sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
    belief_states = process.mixed_state_presentation(sequences.numpy())

    # Truncate to model context window
    model_n_ctx = config["model"]["n_ctx"]
    if model_n_ctx < seq_length:
        print(f"Truncating sequences from {seq_length} to {model_n_ctx} (model n_ctx)")
        sequences = sequences[:, :model_n_ctx]
        belief_states = belief_states[:, :model_n_ctx, :]

    # Bayes-optimal logits at last position
    O = compute_observation_matrix(process)  # (d_vocab, n_states)
    targets = belief_states[:, -1, :]  # (batch, n_states)
    bayes_probs = targets @ O.T  # (batch, d_vocab)
    bayes_logits = np.log(bayes_probs + 1e-12)  # (batch, d_vocab)

    # Best linear approximation M_best: fit M such that M @ b ≈ log(O^T b)
    # Use least-squares: bayes_logits ≈ targets @ M_best^T
    # M_best^T = pinv(targets) @ bayes_logits
    M_best_T = np.linalg.lstsq(targets, bayes_logits, rcond=None)[0]  # (n_states, d_vocab)
    linear_approx = targets @ M_best_T  # (batch, d_vocab)
    delta = bayes_logits - linear_approx  # (batch, d_vocab) nonlinear correction

    print(f"  Bayes logits shape: {bayes_logits.shape}")
    print(f"  delta (nonlinear correction) std: {delta.std():.6f}")
    print(f"  delta / bayes_logits variance ratio: {delta.var() / bayes_logits.var():.4f}")

    return process, sequences, belief_states, targets, bayes_logits, delta


# ---------------------------------------------------------------------------
# Phase 1: Extract components and verify additivity
# ---------------------------------------------------------------------------

def run_phase1(model, sequences, device, n_layers, output_dir):
    """Extract additive components from cache."""
    print("\n=== Phase 1: Extract components ===")

    sequences_device = sequences.to(device)
    with t.no_grad():
        logits, cache = model.run_with_cache(sequences_device)

    # Extract components at last position
    components = extract_components(cache, n_layers, position=-1)

    # Get h_final for verification and LN correction
    h_final = cache["resid_post", n_layers - 1][:, -1, :].cpu().numpy()

    # Verify additivity
    err = verify_additivity(components, h_final)
    print(f"  Additivity max error: {err:.2e}")

    # True logits at last position
    true_logits = logits[:, -1, :].cpu().numpy()  # (batch, d_vocab)

    return components, h_final, true_logits, cache


# ---------------------------------------------------------------------------
# Phase 2: Raw DLA
# ---------------------------------------------------------------------------

def run_phase2(components, W_U, bayes_logits, delta, comp_names, output_dir):
    """Raw DLA: component @ W_U."""
    print("\n=== Phase 2: Raw DLA ===")

    raw_contribs = direct_logit_attribution(components, W_U)
    stats = component_statistics(raw_contribs, bayes_logits, delta, comp_names)

    print("  Component statistics (raw DLA):")
    for _, row in stats.iterrows():
        print(f"    {row['component']:12s}: "
              f"overlap_bayes={row['overlap_bayes']:+.4f}, "
              f"overlap_delta={row['overlap_delta']:+.4f}")

    stats.to_csv(os.path.join(output_dir, "dla_raw.csv"), index=False)
    return raw_contribs, stats


# ---------------------------------------------------------------------------
# Phase 3: LN-corrected DLA
# ---------------------------------------------------------------------------

def run_phase3(components, h_final, model, true_logits, bayes_logits, delta,
               comp_names, output_dir):
    """LN-corrected DLA via Jacobian."""
    print("\n=== Phase 3: LN-corrected DLA ===")

    ln_weight = model.ln_final.w.data.cpu().numpy()
    ln_bias = model.ln_final.b.data.cpu().numpy()
    W_U = model.W_U.data.cpu().numpy()
    b_U = model.b_U.data.cpu().numpy() if hasattr(model, 'b_U') and model.b_U is not None else None

    corrected_contribs = ln_corrected_logit_attribution(
        components, h_final, ln_weight, ln_bias, W_U, b_U
    )

    # Verify sum matches true logits
    err = verify_ln_corrected_sum(corrected_contribs, true_logits)
    print(f"  LN-corrected sum max error: {err:.2e}")

    # Statistics only for network components (not ln_bias/b_U constants)
    stats = component_statistics(corrected_contribs, bayes_logits, delta, comp_names)

    print("  Component statistics (LN-corrected DLA):")
    for _, row in stats.iterrows():
        print(f"    {row['component']:12s}: "
              f"overlap_bayes={row['overlap_bayes']:+.4f}, "
              f"overlap_delta={row['overlap_delta']:+.4f}")

    stats.to_csv(os.path.join(output_dir, "dla_corrected.csv"), index=False)

    # Also save constant contributions info
    const_info = {}
    for key in ["ln_residual", "b_U"]:
        if key in corrected_contribs:
            const_info[key] = corrected_contribs[key][0]  # constant across batch
    if const_info:
        np.savez_compressed(os.path.join(output_dir, "dla_constants.npz"), **const_info)

    return corrected_contribs, stats


# ---------------------------------------------------------------------------
# Phase 4: Logit lens
# ---------------------------------------------------------------------------

def run_phase4(cache, model, n_layers, bayes_logits, output_dir):
    """Logit lens: KL to Bayes-optimal at each layer."""
    print("\n=== Phase 4: Logit lens ===")

    lens_results = logit_lens(cache, model, n_layers, position=-1)

    # Compute KL to Bayes-optimal at each layer
    bayes_log_probs = bayes_logits - np.log(np.exp(bayes_logits).sum(axis=1, keepdims=True))
    bayes_probs = np.exp(bayes_log_probs)

    rows = []
    layer_labels = ["embed"] + [f"block_{i}" for i in range(n_layers)]
    for idx, (label, log_probs) in enumerate(zip(layer_labels, lens_results)):
        # KL(Bayes || model_layer) = sum_x p_bayes * (log p_bayes - log p_model)
        kl = np.sum(bayes_probs * (bayes_log_probs - log_probs), axis=1)
        rows.append({
            "layer_idx": idx,
            "layer_label": label,
            "mean_kl": float(kl.mean()),
            "std_kl": float(kl.std()),
            "median_kl": float(np.median(kl)),
        })
        print(f"  {label:10s}: KL = {kl.mean():.4f} +/- {kl.std():.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "logit_lens.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Phase 5: Component ablation
# ---------------------------------------------------------------------------

def run_phase5(model, sequences, n_layers, comp_names, device, output_dir,
               inference_batch_size=None):
    """Zero-ablate each component, measure KL shift."""
    print("\n=== Phase 5: Component ablation ===")

    sequences_device = sequences.to(device)

    # Clean forward pass
    with t.no_grad():
        clean_logits = model(sequences_device)
    clean_log_probs = F.log_softmax(clean_logits[:, -1, :], dim=-1)

    # Hook names for each component
    hook_map = {
        "embed": "hook_embed",
        "pos_embed": "hook_pos_embed",
    }
    for i in range(n_layers):
        hook_map[f"attn_{i}"] = f"blocks.{i}.hook_attn_out"
        hook_map[f"mlp_{i}"] = f"blocks.{i}.hook_mlp_out"

    rows = []
    for name in comp_names:
        hook_name = hook_map[name]

        def make_zero_hook(hn):
            def hook_fn(tensor, hook):
                tensor[:, -1, :] = 0.0
                return tensor
            return hook_fn

        hook_spec = [(hook_name, make_zero_hook(hook_name))]

        with t.no_grad():
            abl_logits = model.run_with_hooks(sequences_device, fwd_hooks=hook_spec)

        abl_log_probs = F.log_softmax(abl_logits[:, -1, :], dim=-1)

        # KL(clean || ablated)
        kl = F.kl_div(abl_log_probs, clean_log_probs,
                       reduction="none", log_target=True).sum(dim=-1)

        row = {
            "component": name,
            "mean_kl": float(kl.mean()),
            "std_kl": float(kl.std()),
            "median_kl": float(kl.median()),
        }
        rows.append(row)
        print(f"  {name:12s}: KL = {row['mean_kl']:.6f} +/- {row['std_kl']:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "component_ablation.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct Logit Attribution: residual stream decomposition"
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.device is None:
        args.device = get_device()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "logit_attribution")

    # Check skip-if-done
    marker = os.path.join(args.output_dir, "dla_corrected.csv")
    if os.path.exists(marker) and not args.force:
        print(f"Results already exist at {args.output_dir}. Use --force to re-run.")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Model: {args.model_dir}")

    # Load config
    config_path = os.path.join(args.model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_layers = config["model"]["n_layer"]
    comp_names = component_names_for_n_layers(n_layers)

    # Load model
    print(f"Loading model from {args.model_dir}")
    model = load_model_from_dir(args.model_dir, device=args.device)

    # Phase 0: Data and Bayes-optimal
    process, sequences, belief_states, targets, bayes_logits, delta = setup_data(
        config, model, args.batch_size, args.seq_length, args.device
    )

    # Phase 1: Extract components
    components, h_final, true_logits, cache = run_phase1(
        model, sequences, args.device, n_layers, args.output_dir
    )

    W_U = model.W_U.data.cpu().numpy()

    # Phase 2: Raw DLA
    run_phase2(components, W_U, bayes_logits, delta, comp_names, args.output_dir)

    # Phase 3: LN-corrected DLA
    run_phase3(components, h_final, model, true_logits, bayes_logits, delta,
               comp_names, args.output_dir)

    # Phase 4: Logit lens
    run_phase4(cache, model, n_layers, bayes_logits, args.output_dir)

    # Free cache memory before ablation
    del cache

    # Phase 5: Component ablation
    run_phase5(model, sequences, n_layers, comp_names, args.device, args.output_dir)

    # Cleanup
    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()

    print(f"\n=== Done ===")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
