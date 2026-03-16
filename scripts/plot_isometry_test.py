"""
Isometry test: compare W_U blocks across probe subspaces via Procrustes.

Usage:
    python scripts/plot_isometry_test.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from activation_extraction import load_model_from_dir
from unembed_analysis import wu_block_isometry_analysis

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "unembed_analysis")


def load_subspaces(results_dir):
    """Load post-LN subspaces from npz file."""
    data = np.load(os.path.join(results_dir, "postln_subspaces.npz"))
    subspaces = []
    k = 0
    while f"iter{k}" in data:
        subspaces.append(data[f"iter{k}"])
        k += 1
    return subspaces


def main():
    parser = argparse.ArgumentParser(
        description="Isometry test for W_U projections across probe subspaces"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Model directory with config.yaml and checkpoints")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory for PDFs; defaults to {FIGURES_DIR}")
    args = parser.parse_args()

    results_dir = os.path.join(args.model_dir, "unembed_analysis")
    output_dir = args.output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading subspaces...")
    subspaces = load_subspaces(results_dir)
    n_sub = len(subspaces)
    print(f"  {n_sub} subspaces loaded")

    print("Loading model for W_U...")
    model = load_model_from_dir(args.model_dir, device="cpu")
    W_U = model.W_U.detach().numpy()  # (d_model, d_vocab)
    print(f"  W_U shape: {W_U.shape}")
    del model

    # Load auxiliary data for coloring / context
    probe_df = pd.read_csv(os.path.join(results_dir, "postln_iterative_metrics.csv"))
    overlap_df = pd.read_csv(os.path.join(results_dir, "wu_overlap.csv"))

    # Run isometry analysis
    print("Running Procrustes isometry analysis...")
    results = wu_block_isometry_analysis(W_U, subspaces)

    residuals = results["procrustes_residuals"]
    frob_norms = results["frob_norms"]
    rand_mean = results["random_baseline_mean"]
    rand_std = results["random_baseline_std"]

    # Probe R^2 for coloring (match length to n_sub)
    r2_values = probe_df["test_r2"].values[:n_sub]

    # W_U overlap fractions
    overlap_fracs = overlap_df["overlap_fraction"].values[:n_sub]

    # --- Figure: 2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Procrustes residual vs subspace index
    ax = axes[0]
    valid = ~np.isnan(residuals)
    idx = np.arange(n_sub)
    sc = ax.scatter(idx[valid], residuals[valid], c=r2_values[valid],
                    cmap="viridis", s=30, edgecolors="k", linewidths=0.3, zorder=3)
    ax.axhline(rand_mean, color="gray", linestyle="--", alpha=0.7,
               label=f"Random baseline ({rand_mean:.3f})")
    ax.fill_between([idx[0], idx[-1]],
                    rand_mean - rand_std, rand_mean + rand_std,
                    color="gray", alpha=0.15)
    ax.set_xlabel("Subspace index $k$")
    ax.set_ylabel(r"Procrustes residual $\|R_k \hat{B}_0 - \hat{B}_k\|_F$")
    ax.set_title("(A) Geometric alignment with subspace 0")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Probe $R^2$", fontsize=9)

    # Panel B: Frobenius norm (scale) + overlap fraction on secondary axis
    ax = axes[1]
    color1 = "steelblue"
    color2 = "firebrick"
    ax.bar(idx, frob_norms, color=color1, alpha=0.7, label=r"$\|B_k\|_F$")
    ax.set_xlabel("Subspace index $k$")
    ax.set_ylabel(r"$\|B_k\|_F = \|V_k W_U\|_F$", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_title(r"(B) $W_U$ block scale per subspace")
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = ax.twinx()
    ax2.plot(idx[:len(overlap_fracs)], overlap_fracs, "o-", color=color2,
             markersize=3, linewidth=1.2, label="Overlap fraction")
    ax2.set_ylabel(r"$\|V_k W_U\|_F^2 / \|W_U\|_F^2$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "isometry_test.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # Save metrics CSV
    metrics_df = pd.DataFrame({
        "subspace": idx,
        "frob_norm": frob_norms,
        "procrustes_residual": residuals,
        "aligned_cosine": results["aligned_cosines"],
        "det_R": results["det_R"],
        "probe_r2": r2_values,
    })
    csv_path = os.path.join(results_dir, "isometry_test.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Print summary
    print("\n--- Isometry test summary ---")
    print(f"Random baseline: {rand_mean:.4f} +/- {rand_std:.4f}")
    n_good = np.sum(residuals[valid] < rand_mean - 2 * rand_std)
    print(f"Subspaces significantly below random: {n_good} / {valid.sum()}")
    print(f"Subspace 0 residual: {residuals[0]:.6f}")
    for k in [1, 2, 3, 5, 10, 20]:
        if k < n_sub and valid[k]:
            print(f"Subspace {k}: residual={residuals[k]:.4f}, "
                  f"R²={r2_values[k]:.3f}, det(R)={results['det_R'][k]:+.2f}")


if __name__ == "__main__":
    main()
