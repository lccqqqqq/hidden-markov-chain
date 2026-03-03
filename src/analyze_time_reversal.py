"""
Analysis of forward vs. reversed training sweep results.

Downloads runs from W&B, computes forward/reverse val_loss per architecture,
tests H3 (reverse > forward consistently) and H4 (gap depends on capacity),
and saves plots to figures/time_reversal/.

Usage:
    python src/analyze_time_reversal.py
    python src/analyze_time_reversal.py --entity my-entity --project my-project
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "chuqiao-lin-university-of-oxford"
PROJECT = "hidden-markov-chain-src"
OUT_DIR = "figures/time_reversal"
DATA_OUT = "out/time_reversal_analysis.json"


def fetch_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=300)

    fwd, rev = [], []
    for r in runs:
        if r.state != "finished":
            continue
        val_loss = r.summary.get("val_loss", None)
        if val_loss is None:
            continue

        cfg = r.config
        n_layer = cfg.get("n_layer", cfg.get("model", {}).get("n_layer", None))
        n_embd = cfg.get("n_embd", cfg.get("model", {}).get("n_embd", None))
        attn_only = cfg.get("attn_only", cfg.get("model", {}).get("attn_only", None))
        norm = cfg.get("normalization_type", cfg.get("model", {}).get("normalization_type", None))

        if any(x is None for x in [n_layer, n_embd, attn_only, norm]):
            continue

        entry = {
            "name": r.name,
            "val_loss": float(val_loss),
            "n_layer": int(n_layer),
            "n_embd": int(n_embd),
            "attn_only": bool(attn_only),
            "normalization_type": str(norm),
        }

        tags = r.tags
        if "reversed" in tags or "time-reversal" in tags:
            rev.append(entry)
        else:
            fwd.append(entry)

    return fwd, rev


def build_dataframe(runs, label):
    df = pd.DataFrame(runs)
    df["direction"] = label
    return df


def aggregate_best(df):
    """For each unique architecture config, keep the run with lowest val_loss."""
    key_cols = ["n_layer", "n_embd", "attn_only", "normalization_type"]
    return df.loc[df.groupby(key_cols)["val_loss"].idxmin()].reset_index(drop=True)


def num_params_approx(n_layer, n_embd, attn_only, vocab_size=48):
    """Approximate parameter count for CylinderGraph transformer."""
    d = n_embd
    V = vocab_size
    # Embedding: V * d (tied, so count once)
    embed = V * d
    # Each layer: attention (4 * d^2 for Q,K,V,O) + optionally MLP (2 * d * 4d)
    attn_per_layer = 4 * d * d
    mlp_per_layer = 0 if attn_only else 2 * d * 4 * d
    layers = n_layer * (attn_per_layer + mlp_per_layer)
    return embed + layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Fetching W&B runs...")
    fwd_raw, rev_raw = fetch_runs(args.entity, args.project)
    print(f"  Forward: {len(fwd_raw)} finished runs with val_loss")
    print(f"  Reversed: {len(rev_raw)} finished runs with val_loss")

    df_fwd = build_dataframe(fwd_raw, "forward")
    df_rev = build_dataframe(rev_raw, "reversed")

    # Aggregate: best val_loss per config
    df_fwd_best = aggregate_best(df_fwd)
    df_rev_best = aggregate_best(df_rev)
    print(f"  Unique forward configs: {len(df_fwd_best)}")
    print(f"  Unique reversed configs: {len(df_rev_best)}")

    # Merge on architecture
    key_cols = ["n_layer", "n_embd", "attn_only", "normalization_type"]
    df_merged = pd.merge(
        df_fwd_best[key_cols + ["val_loss"]].rename(columns={"val_loss": "loss_fwd"}),
        df_rev_best[key_cols + ["val_loss"]].rename(columns={"val_loss": "loss_rev"}),
        on=key_cols,
        how="inner",
    )
    df_merged["delta"] = df_merged["loss_rev"] - df_merged["loss_fwd"]
    df_merged["n_params"] = df_merged.apply(
        lambda row: num_params_approx(row.n_layer, row.n_embd, row.attn_only), axis=1
    )

    print(f"\n=== Matched configs: {len(df_merged)} ===")
    print(df_merged.sort_values("delta", ascending=False).to_string(index=False))

    # --- H3: Is Δ > 0 consistently? ---
    n_pos = (df_merged["delta"] > 0).sum()
    n_total = len(df_merged)
    print(f"\n=== H3: Δ > 0 in {n_pos}/{n_total} configs ({100*n_pos/n_total:.1f}%) ===")
    print(f"Mean Δ = {df_merged['delta'].mean():.4f} ± {df_merged['delta'].std():.4f}")
    print(f"Min Δ = {df_merged['delta'].min():.4f}, Max Δ = {df_merged['delta'].max():.4f}")

    # --- H4: Does Δ depend on capacity? ---
    corr_params = df_merged["n_params"].corr(df_merged["delta"])
    corr_nembd = df_merged["n_embd"].corr(df_merged["delta"])
    corr_nlayer = df_merged["n_layer"].corr(df_merged["delta"])
    print(f"\n=== H4: Correlations with Δ ===")
    print(f"  Corr(n_params, Δ) = {corr_params:.3f}")
    print(f"  Corr(n_embd,  Δ) = {corr_nembd:.3f}")
    print(f"  Corr(n_layer, Δ) = {corr_nlayer:.3f}")

    # ========== PLOTS ==========

    # 1. Scatter: forward vs reverse val_loss, colored by n_embd
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter_colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}
    markers = {1: "o", 2: "s", 4: "^"}
    for _, row in df_merged.iterrows():
        c = scatter_colors.get(row.n_embd, "gray")
        m = markers.get(row.n_layer, "D")
        ax.scatter(row.loss_fwd, row.loss_rev, color=c, marker=m, s=60, alpha=0.8)
    # Legend for color
    for n_embd, c in scatter_colors.items():
        ax.scatter([], [], color=c, label=f"n_embd={n_embd}", s=60)
    for n_layer, m in markers.items():
        ax.scatter([], [], color="gray", marker=m, label=f"n_layer={n_layer}", s=60)
    lims = [
        min(df_merged["loss_fwd"].min(), df_merged["loss_rev"].min()) - 0.02,
        max(df_merged["loss_fwd"].max(), df_merged["loss_rev"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="forward = reversed")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Forward val_loss")
    ax.set_ylabel("Reversed val_loss")
    ax.set_title("Forward vs Reversed val_loss by architecture")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fwd_vs_rev_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}/fwd_vs_rev_scatter.png")

    # 2. Bar chart of Δ sorted by architecture
    df_plot = df_merged.sort_values(["n_layer", "n_embd", "attn_only", "normalization_type"])
    labels = [
        f"L{row.n_layer}_d{row.n_embd}_{'attn' if row.attn_only else 'full'}_{row.normalization_type}"
        for _, row in df_plot.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), 5))
    bar_colors = ["#d62728" if d > 0 else "#2ca02c" for d in df_plot["delta"]]
    ax.bar(range(len(labels)), df_plot["delta"], color=bar_colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Δ = loss_rev − loss_fwd")
    ax.set_title("Forward–Reverse loss gap by architecture")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_by_arch.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/delta_by_arch.png")

    # 3. Δ vs n_params (log scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    for attn_only, marker, label in [(True, "^", "attn_only"), (False, "o", "full")]:
        sub = df_merged[df_merged["attn_only"] == attn_only]
        ax.scatter(sub["n_params"], sub["delta"], marker=marker, alpha=0.7, label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Approx. parameter count (log scale)")
    ax.set_ylabel("Δ = loss_rev − loss_fwd")
    ax.set_title(f"Loss gap vs model size (corr={corr_params:.2f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_vs_params.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/delta_vs_params.png")

    # 4. Grouped bar: mean val_loss by n_layer (fwd vs rev)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, n_embd in zip(axes, [4, 8, 16]):
        sub = df_merged[df_merged["n_embd"] == n_embd]
        layers = sorted(sub["n_layer"].unique())
        x = np.arange(len(layers))
        w = 0.35
        fwd_vals = [sub[sub["n_layer"] == l]["loss_fwd"].mean() for l in layers]
        rev_vals = [sub[sub["n_layer"] == l]["loss_rev"].mean() for l in layers]
        ax.bar(x - w/2, fwd_vals, w, label="Forward", color="#1f77b4", alpha=0.8)
        ax.bar(x + w/2, rev_vals, w, label="Reversed", color="#d62728", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L={l}" for l in layers])
        ax.set_title(f"n_embd={n_embd}")
        ax.set_xlabel("n_layer")
        if ax == axes[0]:
            ax.set_ylabel("val_loss")
        ax.legend(fontsize=8)
    fig.suptitle("Mean val_loss by depth and embedding size")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grouped_loss_by_arch.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/grouped_loss_by_arch.png")

    # 5. AttnOnly vs Full: delta distributions
    fig, ax = plt.subplots(figsize=(5, 4))
    attn_deltas = df_merged[df_merged["attn_only"] == True]["delta"].values
    full_deltas = df_merged[df_merged["attn_only"] == False]["delta"].values
    ax.hist(attn_deltas, bins=15, alpha=0.6, label=f"attn_only (n={len(attn_deltas)})", color="#ff7f0e")
    ax.hist(full_deltas, bins=15, alpha=0.6, label=f"full (n={len(full_deltas)})", color="#1f77b4")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Δ = loss_rev − loss_fwd")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of loss gap: attn-only vs full")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_hist_by_archtype.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/delta_hist_by_archtype.png")

    # --- Save summary JSON ---
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    summary = {
        "n_forward_runs": len(fwd_raw),
        "n_reversed_runs": len(rev_raw),
        "n_matched_configs": int(n_total),
        "n_delta_positive": int(n_pos),
        "frac_delta_positive": float(n_pos / n_total),
        "mean_delta": float(df_merged["delta"].mean()),
        "std_delta": float(df_merged["delta"].std()),
        "min_delta": float(df_merged["delta"].min()),
        "max_delta": float(df_merged["delta"].max()),
        "corr_nparams_delta": float(corr_params),
        "corr_nembd_delta": float(corr_nembd),
        "corr_nlayer_delta": float(corr_nlayer),
        "mean_attn_only_delta": float(np.mean(attn_deltas)),
        "mean_full_delta": float(np.mean(full_deltas)),
        "records": df_merged.to_dict(orient="records"),
    }
    with open(DATA_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {DATA_OUT}")


if __name__ == "__main__":
    main()
