"""
Generate PDF figures for unembed decomposition analysis.

Usage:
    python scripts/plot_unembed_analysis.py \
        --results-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/unembed_analysis
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "unembed_analysis")


def plot_postln_probing(probe_df, output_path):
    """Figure 0: Post-LN iterative probing R² and MSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(probe_df["iteration"], probe_df["test_r2"], "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Iteration (probe-project cycle)")
    ax.set_ylabel("Test R²")
    ax.set_title("Post-LN iterative probing: R² after successive projections")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="R² threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(probe_df["cumulative_dims"], probe_df["test_mse"], "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Cumulative dimensions projected out")
    ax.set_ylabel("Test MSE")
    ax.set_title("Post-LN probe MSE vs. dimensions removed")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_wu_overlap(overlap_df, output_path):
    """Figure 1: W_U overlap bar chart with per-column breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    n_subspaces = len(overlap_df)

    # Left: overall overlap fraction
    ax = axes[0]
    n_show = min(n_subspaces, 20)
    ax.bar(range(n_show), overlap_df["overlap_fraction"].values[:n_show],
           color="steelblue", alpha=0.8)
    ax.set_xlabel("Subspace index (probe iteration)")
    ax.set_ylabel(r"$\|V_k W_U\|_F^2 / \|W_U\|_F^2$")
    ax.set_title(r"$W_U$ overlap with each probe subspace")
    ax.axhline(1.0 / n_subspaces, color="gray", linestyle="--", alpha=0.5,
               label=f"Uniform (1/{n_subspaces})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: per-token breakdown
    ax = axes[1]
    token_cols = [c for c in overlap_df.columns if c.startswith("overlap_token")]
    d_vocab = len(token_cols)
    n_show_sub = min(8, n_subspaces)
    x = np.arange(n_show_sub)
    width = 0.8 / d_vocab
    colors = plt.cm.Set2(np.linspace(0, 1, d_vocab))

    for j, col in enumerate(token_cols):
        ax.bar(x + j * width, overlap_df[col].values[:n_show_sub], width,
               color=colors[j], label=f"Token {j}", alpha=0.85)

    ax.set_xlabel("Subspace index")
    ax.set_ylabel(r"$\|V_k w_j\|^2 / \|w_j\|^2$")
    ax.set_title("Per-token overlap breakdown")
    ax.set_xticks(x + width * (d_vocab - 1) / 2)
    ax.set_xticklabels(range(n_show_sub))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_effective_emission(emission_data, output_path):
    """Figure 2: Recovered M_eff vs theoretical log(O^T)."""
    M_eff = emission_data["subspace0"]
    log_O_T = emission_data["log_O_T"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.imshow(M_eff, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Token j")
    ax.set_ylabel("State i")
    ax.set_title(r"Recovered $M_{\mathrm{eff}}$ (subspace 0)")
    for i in range(M_eff.shape[0]):
        for j in range(M_eff.shape[1]):
            ax.text(j, i, f"{M_eff[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(log_O_T, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Token j")
    ax.set_ylabel("State i")
    ax.set_title(r"Theoretical $\log(O^T)$")
    for i in range(log_O_T.shape[0]):
        for j in range(log_O_T.shape[1]):
            ax.text(j, i, f"{log_O_T[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(r"Effective emission matrix vs. Bayes-optimal", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_wu_ablation(ablation_df, results_dir, output_path):
    """Figure 3: Per-subspace KL bar chart + cumulative KL line chart."""
    per_sub = ablation_df[ablation_df["ablation_type"] == "per_subspace"].copy()
    cumul = ablation_df[ablation_df["ablation_type"] == "cumulative"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-subspace KL
    ax = axes[0]
    n_show = min(20, len(per_sub))
    ax.bar(per_sub["subspace"].values[:n_show], per_sub["mean_kl"].values[:n_show],
           color="firebrick", alpha=0.8,
           yerr=per_sub["std_kl"].values[:n_show], capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xlabel("Subspace index")
    ax.set_ylabel("KL(clean || ablated)")
    ax.set_title(r"Per-subspace $W_U$ ablation")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Right: cumulative KL
    ax = axes[1]
    try:
        probe_df = pd.read_csv(os.path.join(results_dir, "postln_iterative_metrics.csv"))
        cum_dims = probe_df["cumulative_dims"].values[:len(cumul)]
    except Exception:
        cum_dims = np.arange(1, len(cumul) + 1) * 2

    ax.plot(cum_dims, cumul["mean_kl"].values, "o-",
            color="firebrick", markersize=4)
    ax.fill_between(cum_dims,
                    cumul["mean_kl"].values - cumul["std_kl"].values,
                    cumul["mean_kl"].values + cumul["std_kl"].values,
                    color="firebrick", alpha=0.15)
    ax.set_xlabel(r"Cumulative dimensions removed from $W_U$")
    ax.set_ylabel("KL(clean || ablated)")
    ax.set_title(r"Cumulative $W_U$ ablation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_logit_contributions(logit_df, output_path):
    """Figure 4: Logit contribution variance + Bayes-optimal correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_show = min(20, len(logit_df))

    ax = axes[0]
    ax.bar(logit_df["subspace"].values[:n_show], logit_df["mean_variance"].values[:n_show],
           color="teal", alpha=0.8)
    ax.set_xlabel("Subspace index")
    ax.set_ylabel("Mean logit variance")
    ax.set_title("Logit contribution variance per subspace")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(logit_df["subspace"].values[:n_show], logit_df["bayes_correlation"].values[:n_show],
           color="darkorange", alpha=0.8)
    ax.set_xlabel("Subspace index")
    ax.set_ylabel("Correlation with Bayes-optimal logits")
    ax.set_title("Logit contribution: correlation with Bayes-optimal")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF figures for unembed decomposition analysis"
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with unembed analysis results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory for PDFs; defaults to {FIGURES_DIR}")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = FIGURES_DIR

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}")

    overlap_df = pd.read_csv(os.path.join(args.results_dir, "wu_overlap.csv"))
    ablation_df = pd.read_csv(os.path.join(args.results_dir, "wu_ablation.csv"))
    logit_df = pd.read_csv(os.path.join(args.results_dir, "logit_contributions.csv"))
    emission_data = np.load(os.path.join(args.results_dir, "effective_emission.npz"))
    probe_df = pd.read_csv(os.path.join(args.results_dir, "postln_iterative_metrics.csv"))

    print(f"Saving figures to {args.output_dir}")

    plot_postln_probing(probe_df, os.path.join(args.output_dir, "postln_probing.pdf"))
    plot_wu_overlap(overlap_df, os.path.join(args.output_dir, "wu_overlap.pdf"))
    plot_effective_emission(emission_data, os.path.join(args.output_dir, "effective_emission.pdf"))
    plot_wu_ablation(ablation_df, args.results_dir,
                     os.path.join(args.output_dir, "wu_ablation.pdf"))
    plot_logit_contributions(logit_df, os.path.join(args.output_dir, "logit_contributions.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
