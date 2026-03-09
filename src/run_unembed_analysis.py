"""
CLI pipeline for unembed decomposition analysis.

Analyzes how W_U reads belief-state subspaces found by iterative probing
on post-LayerNorm activations.

Usage:
    python src/run_unembed_analysis.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --seq-length 10 --batch-size 10000 --device cpu
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
from iterative_probe import (
    iterative_probe_single_layer,
)
from unembed_analysis import (
    compute_observation_matrix,
    wu_subspace_overlaps,
    wu_per_column_overlaps,
    effective_emission_matrix,
    principal_angles,
    wu_read_subspace,
    logit_subspace_contributions,
)


# ---------------------------------------------------------------------------
# Data generation and activation extraction
# ---------------------------------------------------------------------------

def generate_data_and_beliefs(config, batch_size, seq_length):
    """Generate sequences and compute belief states."""
    process = get_process_from_config(config)
    print(f"Generating {batch_size} sequences of length {seq_length} "
          f"from {process.__class__.__name__}")
    sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
    belief_states = process.mixed_state_presentation(sequences.numpy())
    return process, sequences, belief_states


def extract_postln_activations(model, sequences, device, inference_batch_size=None):
    """Extract post-LayerNorm activations at the last position.

    Returns:
        (batch, d_model) numpy array of post-LN activations.
    """
    if inference_batch_size is None or inference_batch_size >= sequences.shape[0]:
        chunk_ranges = [(0, sequences.shape[0])]
    else:
        chunk_ranges = [
            (i, min(i + inference_batch_size, sequences.shape[0]))
            for i in range(0, sequences.shape[0], inference_batch_size)
        ]

    all_actvs = []
    for start, end in chunk_ranges:
        batch_seqs = sequences[start:end].to(device)
        with t.no_grad():
            _, cache = model.run_with_cache(batch_seqs)
        # Post-LN activations at last position
        postln = cache["ln_final.hook_normalized"][:, -1, :].cpu().numpy()
        all_actvs.append(postln)

    return np.concatenate(all_actvs, axis=0)


# ---------------------------------------------------------------------------
# Phase 1: Iterative probing on post-LN activations
# ---------------------------------------------------------------------------

def run_postln_probing(postln_actvs, targets, max_iterations, r2_threshold,
                       test_fraction, output_dir):
    """Run iterative probing on post-LN activations (single 'layer')."""
    print(f"\nIterative probing on post-LN activations "
          f"(max_iter={max_iterations}, R² threshold={r2_threshold}):")

    result = iterative_probe_single_layer(
        postln_actvs, targets,
        max_iterations=max_iterations,
        r2_threshold=r2_threshold,
        test_fraction=test_fraction,
    )
    result.layer = 0  # single "layer"

    for i, it in enumerate(result.iterations):
        r = it.probe_result
        print(f"  Iter {i}: test R²={r.test_r2:.4f}, "
              f"rank={it.subspace.shape[0]}, "
              f"S={it.singular_values}")

    if result.n_significant == 0:
        print(f"  No significant probe found (R² < {r2_threshold})")
    else:
        print(f"  Total: {result.n_significant} copies, "
              f"{result.cumulative_dims} dims")

    # Save metrics
    rows = []
    cum_dims = 0
    for i, it in enumerate(result.iterations):
        r = it.probe_result
        rank = it.subspace.shape[0]
        cum_dims += rank
        rows.append({
            "iteration": i,
            "train_r2": r.train_r2,
            "test_r2": r.test_r2,
            "train_mse": r.train_mse,
            "test_mse": r.test_mse,
            "subspace_rank": rank,
            "cumulative_dims": cum_dims,
            "singular_values": ",".join(f"{s:.6f}" for s in it.singular_values),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "postln_iterative_metrics.csv"), index=False
    )

    # Save subspaces
    data = {}
    for i, it in enumerate(result.iterations):
        data[f"iter{i}"] = it.subspace
        data[f"iter{i}_S"] = it.singular_values
    np.savez_compressed(os.path.join(output_dir, "postln_subspaces.npz"), **data)

    return result


# ---------------------------------------------------------------------------
# Part 1: W_U variance decomposition
# ---------------------------------------------------------------------------

def run_part1(W_U, subspaces, output_dir):
    """W_U overlap with each probe subspace."""
    print("\n=== Part 1: W_U variance decomposition ===")

    overlaps = wu_subspace_overlaps(W_U, subspaces)
    per_col = wu_per_column_overlaps(W_U, subspaces)

    print(f"  Total overlap (Parseval check): {overlaps.sum():.6f}")
    print(f"  Top-3 subspaces: {np.argsort(overlaps)[::-1][:3]} "
          f"with overlaps {np.sort(overlaps)[::-1][:3]}")

    # Principal angles between W_U read subspace and top probe subspaces
    wu_read = wu_read_subspace(W_U)
    print(f"  W_U read subspace rank: {wu_read.shape[0]}")

    angles_list = []
    for k, V in enumerate(subspaces[:min(5, len(subspaces))]):
        angles = principal_angles(wu_read, V)
        angles_list.append({
            "subspace": k,
            "angles_deg": ",".join(f"{np.degrees(a):.2f}" for a in angles),
            "min_angle_deg": float(np.degrees(angles.min())),
        })
        print(f"  Principal angles (W_U read ↔ subspace {k}): "
              f"{np.degrees(angles).round(1)}°")

    # Save
    rows = []
    for k in range(len(subspaces)):
        row = {"subspace": k, "overlap_fraction": overlaps[k]}
        for j in range(per_col.shape[1]):
            row[f"overlap_token{j}"] = per_col[k, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "wu_overlap.csv"), index=False
    )

    if angles_list:
        pd.DataFrame(angles_list).to_csv(
            os.path.join(output_dir, "wu_principal_angles.csv"), index=False
        )

    return overlaps, per_col


# ---------------------------------------------------------------------------
# Part 2: Effective emission matrix recovery
# ---------------------------------------------------------------------------

def run_part2(W_probe_coef, W_U, subspaces, process, output_dir):
    """Recover effective emission matrix through dominant subspace."""
    print("\n=== Part 2: Effective emission matrix recovery ===")

    O = compute_observation_matrix(process)
    log_O_T = np.log(O.T + 1e-12)  # (n_states, d_vocab), avoid log(0)

    M_effs = {}
    for k in range(min(3, len(subspaces))):
        M_eff = effective_emission_matrix(W_probe_coef, W_U, subspaces[k])
        M_effs[f"subspace{k}"] = M_eff
        print(f"  Subspace {k} effective map (n_states × d_vocab):")
        print(f"    {M_eff.round(3)}")

    # Cosine similarity between M_eff rows and log(O^T) rows
    if len(subspaces) > 0:
        M0 = M_effs["subspace0"]
        # Row-wise cosine similarity
        for s in range(M0.shape[0]):
            cos_sim = (np.dot(M0[s], log_O_T[s]) /
                       (np.linalg.norm(M0[s]) * np.linalg.norm(log_O_T[s]) + 1e-12))
            print(f"  Cosine sim (state {s}): {cos_sim:.4f}")

        # Overall Frobenius cosine similarity (after centering columns)
        M0_c = M0 - M0.mean(axis=0, keepdims=True)
        log_c = log_O_T - log_O_T.mean(axis=0, keepdims=True)
        overall_cos = (np.sum(M0_c * log_c) /
                       (np.linalg.norm(M0_c, 'fro') * np.linalg.norm(log_c, 'fro') + 1e-12))
        print(f"  Overall cosine similarity (centered): {overall_cos:.4f}")

    # Save
    save_data = {"log_O_T": log_O_T, "O": O}
    for name, M in M_effs.items():
        save_data[name] = M
    np.savez_compressed(os.path.join(output_dir, "effective_emission.npz"), **save_data)

    return M_effs, O, log_O_T


# ---------------------------------------------------------------------------
# Part 3: W_U ablation
# ---------------------------------------------------------------------------

def run_part3(model, sequences, subspaces, device, output_dir,
              inference_batch_size=None):
    """Causal test: ablate W_U projections onto each subspace."""
    print("\n=== Part 3: Causal W_U ablation ===")

    sequences_device = sequences.to(device)

    # Clean outputs
    with t.no_grad():
        clean_logits = model(sequences_device)
    clean_log_probs = F.log_softmax(clean_logits[:, -1, :], dim=-1)
    clean_ce = float(-(clean_log_probs.exp() * clean_log_probs).sum(dim=-1).mean())
    print(f"  Clean model entropy: {clean_ce:.4f} nats")
    del clean_logits

    W_U_orig = model.W_U.data.clone()  # (d_model, d_vocab)
    rows = []

    # Per-subspace ablation
    for k, V in enumerate(subspaces):
        V_t = t.from_numpy(V).float().to(device)
        # W_U' = W_U - V^T @ (V @ W_U)
        projection = V_t.T @ (V_t @ W_U_orig)
        with t.no_grad():
            model.W_U.data = W_U_orig - projection
            abl_logits = model(sequences_device)
        abl_log_probs = F.log_softmax(abl_logits[:, -1, :], dim=-1)

        kl = F.kl_div(abl_log_probs, clean_log_probs,
                       reduction="none", log_target=True).sum(dim=-1)
        abl_ce = float(-(clean_log_probs.exp() * abl_log_probs).sum(dim=-1).mean())

        row = {
            "subspace": k,
            "ablation_type": "per_subspace",
            "mean_kl": float(kl.mean()),
            "std_kl": float(kl.std()),
            "mean_ce": abl_ce,
            "delta_ce": abl_ce - clean_ce,
        }
        rows.append(row)
        if k < 5 or row["mean_kl"] > 0.001:
            print(f"  Subspace {k}: KL={row['mean_kl']:.6f}, ΔCE={row['delta_ce']:.6f}")

    # Cumulative ablation
    print("  Cumulative ablation:")
    for k in range(len(subspaces)):
        cum_V = np.vstack(subspaces[:k + 1])
        V_t = t.from_numpy(cum_V).float().to(device)
        projection = V_t.T @ (V_t @ W_U_orig)
        with t.no_grad():
            model.W_U.data = W_U_orig - projection
            abl_logits = model(sequences_device)
        abl_log_probs = F.log_softmax(abl_logits[:, -1, :], dim=-1)

        kl = F.kl_div(abl_log_probs, clean_log_probs,
                       reduction="none", log_target=True).sum(dim=-1)
        abl_ce = float(-(clean_log_probs.exp() * abl_log_probs).sum(dim=-1).mean())

        row = {
            "subspace": k,
            "ablation_type": "cumulative",
            "mean_kl": float(kl.mean()),
            "std_kl": float(kl.std()),
            "mean_ce": abl_ce,
            "delta_ce": abl_ce - clean_ce,
        }
        rows.append(row)
        cum_dims = cum_V.shape[0]
        if k < 5 or cum_dims <= 8:
            print(f"    0..{k} ({cum_dims} dims): KL={row['mean_kl']:.6f}, "
                  f"ΔCE={row['delta_ce']:.6f}")

    # Restore original W_U
    model.W_U.data = W_U_orig

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "wu_ablation.csv"), index=False)
    print(f"  Saved {len(rows)} ablation results")

    return df


# ---------------------------------------------------------------------------
# Part 4: Logit contribution profile
# ---------------------------------------------------------------------------

def run_part4(postln_actvs, W_U, subspaces, belief_states, process, output_dir):
    """Decompose logits into per-subspace contributions and measure variance/correlation."""
    print("\n=== Part 4: Logit contribution profile ===")

    contributions = logit_subspace_contributions(postln_actvs, W_U, subspaces)
    # contributions: (n_subspaces, n_samples, d_vocab)

    # Bayes-optimal logits
    O = compute_observation_matrix(process)  # (d_vocab, n_states)
    bayes_probs = belief_states @ O.T  # (n_samples, d_vocab)
    bayes_logits = np.log(bayes_probs + 1e-12)  # (n_samples, d_vocab)

    rows = []
    for k in range(len(subspaces)):
        logit_k = contributions[k]  # (n_samples, d_vocab)
        # Variance of logit contributions across samples
        var_k = logit_k.var(axis=0).mean()  # mean over tokens
        # Correlation with Bayes-optimal logits
        # Flatten and compute Pearson correlation
        flat_k = logit_k.flatten()
        flat_bayes = bayes_logits.flatten()
        corr = np.corrcoef(flat_k, flat_bayes)[0, 1]
        # Also per-token correlation
        per_token_corr = []
        for j in range(logit_k.shape[1]):
            c = np.corrcoef(logit_k[:, j], bayes_logits[:, j])[0, 1]
            per_token_corr.append(c)

        row = {
            "subspace": k,
            "mean_variance": float(var_k),
            "bayes_correlation": float(corr),
        }
        for j, c in enumerate(per_token_corr):
            row[f"corr_token{j}"] = float(c)
        rows.append(row)

        if k < 5:
            print(f"  Subspace {k}: var={var_k:.6f}, corr(Bayes)={corr:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "logit_contributions.csv"), index=False)

    # Also save the full contributions for the top few subspaces (for plotting)
    save_data = {"bayes_logits": bayes_logits}
    for k in range(min(5, len(subspaces))):
        save_data[f"subspace{k}"] = contributions[k]
    np.savez_compressed(os.path.join(output_dir, "logit_contributions.npz"), **save_data)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unembed decomposition: how W_U reads belief-state subspaces"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Model directory with config.yaml and checkpoint")
    parser.add_argument("--seq-length", type=int, default=10,
                        help="Sequence length for generated data")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Number of sequences to generate")
    parser.add_argument("--max-iterations", type=int, default=32,
                        help="Max probe-project iterations on post-LN activations")
    parser.add_argument("--r2-threshold", type=float, default=0.05,
                        help="Minimum test R² to continue iterating")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction held out for probe testing")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/mps); auto-detected if omitted")
    parser.add_argument("--inference-batch-size", type=int, default=None,
                        help="Batch size for model inference (memory control)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory; defaults to {model_dir}/unembed_analysis/")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")

    args = parser.parse_args()

    if args.device is None:
        args.device = get_device()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "unembed_analysis")

    # Check skip-if-done
    overlap_path = os.path.join(args.output_dir, "wu_overlap.csv")
    if os.path.exists(overlap_path) and not args.force:
        print(f"Results already exist at {args.output_dir}. Use --force to re-run.")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Model: {args.model_dir}")

    # Load config
    config_path = os.path.join(args.model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_n_ctx = config["model"]["n_ctx"]

    # Generate data
    process, sequences, belief_states = generate_data_and_beliefs(
        config, args.batch_size, args.seq_length
    )

    # Truncate to model context window
    if model_n_ctx < args.seq_length:
        print(f"Truncating sequences from {args.seq_length} to {model_n_ctx} (model n_ctx)")
        sequences = sequences[:, :model_n_ctx]
        belief_states = belief_states[:, :model_n_ctx, :]

    # Load model
    print(f"Loading model from {args.model_dir}")
    model = load_model_from_dir(args.model_dir, device=args.device)

    # Extract post-LN activations
    print("Extracting post-LN activations...")
    postln_actvs = extract_postln_activations(
        model, sequences, args.device,
        inference_batch_size=args.inference_batch_size,
    )
    targets = belief_states[:, -1, :]  # last position belief states

    print(f"  Post-LN activations shape: {postln_actvs.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Phase 0: Iterative probing on post-LN activations
    probe_result = run_postln_probing(
        postln_actvs, targets,
        max_iterations=args.max_iterations,
        r2_threshold=args.r2_threshold,
        test_fraction=args.test_fraction,
        output_dir=args.output_dir,
    )

    if probe_result.n_significant == 0:
        print("\nNo significant subspaces found in post-LN activations. Exiting.")
        sys.exit(1)

    # Extract subspaces and probe weights
    subspaces = [it.subspace for it in probe_result.iterations]
    W_probe_coef = probe_result.iterations[0].probe_result.regressor.coef_

    # Extract W_U
    W_U = model.W_U.data.cpu().numpy()  # (d_model, d_vocab)
    b_U = model.b_U.data.cpu().numpy() if hasattr(model, 'b_U') else None
    print(f"\nW_U shape: {W_U.shape}, rank: {np.linalg.matrix_rank(W_U)}")
    if b_U is not None:
        print(f"b_U: {b_U}")

    # Part 1: W_U variance decomposition
    overlaps, per_col = run_part1(W_U, subspaces, args.output_dir)

    # Part 2: Effective emission matrix recovery
    M_effs, O, log_O_T = run_part2(
        W_probe_coef, W_U, subspaces, process, args.output_dir
    )

    # Part 3: Causal W_U ablation
    run_part3(model, sequences, subspaces, args.device, args.output_dir,
              inference_batch_size=args.inference_batch_size)

    # Part 4: Logit contribution profile
    run_part4(postln_actvs, W_U, subspaces, targets, process, args.output_dir)

    # Cleanup
    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()

    print("\n=== Done ===")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
