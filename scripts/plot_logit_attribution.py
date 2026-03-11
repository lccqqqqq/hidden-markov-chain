"""
Generate PDF figures for Direct Logit Attribution analysis.

Usage:
    python scripts/plot_logit_attribution.py \
        --results-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/logit_attribution
"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "logit_attribution")

COMPONENT_COLORS = {
    "embed": "gray", "pos": "gray",
    "attn": "steelblue",
    "mlp": "firebrick",
}

LEGEND_ELEMENTS = [
    Patch(facecolor="gray", alpha=0.85, label="Embed"),
    Patch(facecolor="steelblue", alpha=0.85, label="Attention"),
    Patch(facecolor="firebrick", alpha=0.85, label="MLP"),
]


def _get_colors(names):
    colors = []
    for name in names:
        if name.startswith("embed") or name.startswith("pos"):
            colors.append("gray")
        elif name.startswith("attn"):
            colors.append("steelblue")
        elif name.startswith("mlp"):
            colors.append("firebrick")
        else:
            colors.append("gray")
    return colors


def plot_component_overlaps(df, output_path):
    """Figure 1: 1x2 panel of overlap with Bayes-optimal and delta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = df["component"].values
    x = range(len(names))
    colors = _get_colors(names)

    ax = axes[0]
    ax.bar(x, df["overlap_bayes"].values, color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"$\langle c_i,\, \log O^T b \rangle \,/\, \|\log O^T b\|^2$")
    ax.set_title(r"Overlap with Bayes-optimal logits $\log(O^T b)$")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(handles=LEGEND_ELEMENTS, loc="best", fontsize=8)

    ax = axes[1]
    ax.bar(x, df["overlap_delta"].values, color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"$\langle c_i,\, \delta \rangle \,/\, \|\delta\|^2$")
    ax.set_title(r"Overlap with nonlinear correction $\delta$")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_logit_lens(df, output_path):
    """Figure 2: KL to Bayes-optimal vs network depth."""
    fig, ax = plt.subplots(figsize=(8, 4))

    x = df["layer_idx"].values
    labels = df["layer_label"].values
    kl = df["mean_kl"].values
    kl_std = df["std_kl"].values

    ax.plot(x, kl, "o-", color="teal", markersize=6, linewidth=2)
    ax.fill_between(x, kl - kl_std, np.maximum(kl + kl_std, 0), color="teal", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("KL(Bayes || model) [nats]")
    ax.set_title("Logit lens: prediction quality through the network")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_component_ablation(df, output_path):
    """Figure 3: Bar chart of ablation KL per component (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 4))

    names = df["component"].values
    kl = df["mean_kl"].values
    kl_std = df["std_kl"].values
    colors = _get_colors(names)

    ax.bar(range(len(names)), kl, color=colors, alpha=0.85,
           yerr=kl_std, capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("KL(clean || ablated) [nats]")
    ax.set_title("Component ablation: causal importance (zero-ablation at last position)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(handles=LEGEND_ELEMENTS, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_dla_comparison(raw_df, corrected_df, output_path):
    """Figure 4: Raw vs LN-corrected DLA overlap with Bayes-optimal side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, df, title in [
        (axes[0], raw_df, "Raw DLA (no LN correction)"),
        (axes[1], corrected_df, "LN-corrected DLA"),
    ]:
        names = df["component"].values
        vals = df["overlap_bayes"].values
        colors = _get_colors(names)

        ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(r"Overlap with Bayes-optimal logits")

    fig.suptitle("DLA comparison: effect of LayerNorm correction", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF figures for logit attribution analysis"
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = FIGURES_DIR
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}")

    corrected_df = pd.read_csv(os.path.join(args.results_dir, "dla_corrected.csv"))
    raw_df = pd.read_csv(os.path.join(args.results_dir, "dla_raw.csv"))
    lens_df = pd.read_csv(os.path.join(args.results_dir, "logit_lens.csv"))
    ablation_df = pd.read_csv(os.path.join(args.results_dir, "component_ablation.csv"))

    print(f"Saving figures to {args.output_dir}")

    plot_component_overlaps(corrected_df,
                            os.path.join(args.output_dir, "component_overlaps.pdf"))
    plot_logit_lens(lens_df,
                    os.path.join(args.output_dir, "logit_lens.pdf"))
    plot_component_ablation(ablation_df,
                            os.path.join(args.output_dir, "component_ablation.pdf"))
    plot_dla_comparison(raw_df, corrected_df,
                        os.path.join(args.output_dir, "dla_comparison.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
