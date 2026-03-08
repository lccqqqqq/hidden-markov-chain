"""
Analyze training dynamics: forward vs reversed CylinderGraph HMM.

Tests H5 (transient dynamics asymmetry), H6 (learning phases),
and H7 (architecture-dependent convergence gap).

Reads CSVs from data/training_dynamics/ (produced by extract_training_curves.py),
computes convergence metrics, and generates figures.

Usage:
    python src/analyze_training_dynamics.py
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import stats


DATA_DIR = "data/training_dynamics"
FIG_DIR = "figures/training_dynamics"
OUT_PATH = "out/training_dynamics_summary.json"

ENTROPY_RATE = 2.44  # nats, approximate for this CylinderGraphHMM

# Colors matching analyze_time_reversal.py
COLOR_FWD = "#1f77b4"
COLOR_REV = "#d62728"
SCATTER_COLORS = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}
MARKERS = {1: "o", 2: "s", 4: "^"}


def num_params_approx(n_layer, n_embd, attn_only, vocab_size=48):
    """Approximate parameter count for CylinderGraph transformer."""
    d = n_embd
    V = vocab_size
    embed = V * d
    attn_per_layer = 4 * d * d
    mlp_per_layer = 0 if attn_only else 2 * d * 4 * d
    layers = n_layer * (attn_per_layer + mlp_per_layer)
    return embed + layers


def compute_run_metrics(steps, losses):
    """Compute convergence metrics for a single run's val_loss curve.

    Args:
        steps: array of step numbers
        losses: array of val_loss values

    Returns:
        dict with tau_50, tau_90, AUC, epoch1_loss, curvature_ratio
    """
    steps = np.array(steps)
    losses = np.array(losses)

    L_0 = losses[0]
    L_final = losses[-1]
    improvement = L_0 - L_final

    if improvement <= 0:
        # No improvement — return NaN metrics
        return {
            "tau_50": np.nan, "tau_90": np.nan, "AUC": np.nan,
            "epoch1_loss": np.nan, "curvature_ratio": np.nan,
            "L_0": L_0, "L_final": L_final,
        }

    # tau_50: first step where loss <= L_0 - 0.5 * improvement
    threshold_50 = L_0 - 0.5 * improvement
    idx_50 = np.where(losses <= threshold_50)[0]
    tau_50 = steps[idx_50[0]] if len(idx_50) > 0 else steps[-1]

    # tau_90: first step where loss <= L_0 - 0.9 * improvement
    threshold_90 = L_0 - 0.9 * improvement
    idx_90 = np.where(losses <= threshold_90)[0]
    tau_90 = steps[idx_90[0]] if len(idx_90) > 0 else steps[-1]

    # AUC: area under excess loss curve (trapezoidal)
    excess = losses - L_final
    AUC = np.trapezoid(excess, steps)

    # Epoch-1 loss: val_loss at step closest to 4834
    epoch1_idx = np.argmin(np.abs(steps - 4834))
    epoch1_loss = losses[epoch1_idx]

    # Curvature ratio
    curvature_ratio = tau_90 / tau_50 if tau_50 > 0 else np.nan

    return {
        "tau_50": float(tau_50),
        "tau_90": float(tau_90),
        "AUC": float(AUC),
        "epoch1_loss": float(epoch1_loss),
        "curvature_ratio": float(curvature_ratio),
        "L_0": float(L_0),
        "L_final": float(L_final),
    }


def load_data(data_dir):
    """Load extracted CSVs."""
    df_loss = pd.read_csv(os.path.join(data_dir, "loss_curves.csv"))
    df_epoch = pd.read_csv(os.path.join(data_dir, "epoch_train_loss.csv"))
    df_meta = pd.read_csv(os.path.join(data_dir, "run_metadata.csv"))
    return df_loss, df_epoch, df_meta


def compute_all_metrics(df_loss):
    """Compute convergence metrics for each run."""
    rows = []
    for run_id, grp in df_loss.groupby("run_id"):
        grp = grp.sort_values("step")
        metrics = compute_run_metrics(grp["step"].values, grp["val_loss"].values)
        info = grp.iloc[0]
        metrics.update({
            "run_id": run_id,
            "n_layer": info["n_layer"],
            "n_embd": info["n_embd"],
            "attn_only": info["attn_only"],
            "norm": info["norm"],
            "direction": info["direction"],
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def aggregate_by_config(df_metrics):
    """Average metrics across seeds for each (arch, direction) config.

    Returns one row per (arch, direction) with mean and std of each metric.
    """
    key_cols = ["n_layer", "n_embd", "attn_only", "norm", "direction"]
    metric_cols = ["tau_50", "tau_90", "AUC", "epoch1_loss", "curvature_ratio", "L_final"]

    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = "mean"
        agg_dict[f"{col}_std"] = (col, "std")

    # Manual aggregation to get both mean and std
    rows = []
    for keys, grp in df_metrics.groupby(key_cols):
        row = dict(zip(key_cols, keys))
        row["n_runs"] = len(grp)
        for col in metric_cols:
            row[col] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std() if len(grp) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def pair_configs(df_agg):
    """Pair forward/reversed configs by architecture, compute deltas."""
    key_cols = ["n_layer", "n_embd", "attn_only", "norm"]
    metric_cols = ["tau_50", "tau_90", "AUC", "epoch1_loss", "curvature_ratio", "L_final"]

    df_fwd = df_agg[df_agg["direction"] == "forward"].copy()
    df_rev = df_agg[df_agg["direction"] == "reversed"].copy()

    df_paired = pd.merge(
        df_fwd[key_cols + metric_cols + ["n_runs"]].rename(
            columns={c: f"{c}_fwd" for c in metric_cols + ["n_runs"]}
        ),
        df_rev[key_cols + metric_cols + ["n_runs"]].rename(
            columns={c: f"{c}_rev" for c in metric_cols + ["n_runs"]}
        ),
        on=key_cols,
        how="inner",
    )

    for col in metric_cols:
        df_paired[f"delta_{col}"] = df_paired[f"{col}_rev"] - df_paired[f"{col}_fwd"]

    df_paired["n_params"] = df_paired.apply(
        lambda r: num_params_approx(r.n_layer, r.n_embd, r.attn_only), axis=1
    )

    return df_paired


def aggregate_loss_curves(df_loss):
    """Compute mean loss curve per (arch, direction) config.

    Returns DataFrame with step, mean_val_loss, std_val_loss for each config.
    """
    key_cols = ["n_layer", "n_embd", "attn_only", "norm", "direction"]
    grouped = df_loss.groupby(key_cols + ["step"]).agg(
        mean_val_loss=("val_loss", "mean"),
        std_val_loss=("val_loss", "std"),
        n_runs=("val_loss", "count"),
    ).reset_index()
    # Fill NaN std (single run) with 0
    grouped["std_val_loss"] = grouped["std_val_loss"].fillna(0)
    return grouped


# ============================================================
# Plotting functions
# ============================================================

def plot_loss_curves_grid(df_curves, fig_dir):
    """Figure 1: Grid of loss curves, n_layer x n_embd, fwd vs rev overlay."""
    layers = sorted(df_curves["n_layer"].unique())
    embds = sorted(df_curves["n_embd"].unique())
    n_rows = len(layers)
    n_cols = len(embds)

    # Separate panels for attn_only=True and attn_only=False
    for attn_only, attn_label in [(False, "full"), (True, "attn_only")]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True)
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, n_layer in enumerate(layers):
            for j, n_embd in enumerate(embds):
                ax = axes[i, j]

                for direction, color, label in [
                    ("forward", COLOR_FWD, "Forward"),
                    ("reversed", COLOR_REV, "Reversed"),
                ]:
                    # Average over norm variants for cleaner plot
                    mask = (
                        (df_curves["n_layer"] == n_layer) &
                        (df_curves["n_embd"] == n_embd) &
                        (df_curves["attn_only"] == attn_only) &
                        (df_curves["direction"] == direction)
                    )
                    sub = df_curves[mask].sort_values("step")
                    if len(sub) == 0:
                        continue

                    # If multiple norm variants, average them
                    curve = sub.groupby("step").agg(
                        val_loss=("mean_val_loss", "mean"),
                    ).reset_index()

                    ax.plot(curve["step"], curve["val_loss"],
                            color=color, label=label, linewidth=1.2, alpha=0.85)

                    # Shade std if available (from multi-seed)
                    if sub["n_runs"].max() > 1:
                        curve_std = sub.groupby("step").agg(
                            val_loss=("mean_val_loss", "mean"),
                            std=("std_val_loss", "mean"),
                        ).reset_index()
                        ax.fill_between(
                            curve_std["step"],
                            curve_std["val_loss"] - curve_std["std"],
                            curve_std["val_loss"] + curve_std["std"],
                            color=color, alpha=0.12,
                        )

                ax.axhline(ENTROPY_RATE, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.6)
                ax.set_title(f"L={n_layer}, d={n_embd}", fontsize=10)
                ax.grid(alpha=0.3)

                if i == n_rows - 1:
                    ax.set_xlabel("Step")
                if j == 0:
                    ax.set_ylabel("Val Loss")
                if i == 0 and j == n_cols - 1:
                    ax.legend(fontsize=8)

        fig.suptitle(f"Training Dynamics: Forward vs Reversed ({attn_label})", fontsize=13)
        plt.tight_layout()
        path = os.path.join(fig_dir, f"loss_curves_grid_{attn_label}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_loss_curves_by_norm(df_curves, fig_dir):
    """Figure 1b: Grid with norm variants shown separately (not averaged)."""
    layers = sorted(df_curves["n_layer"].unique())
    embds = sorted(df_curves["n_embd"].unique())
    n_rows = len(layers)
    n_cols = len(embds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, n_layer in enumerate(layers):
        for j, n_embd in enumerate(embds):
            ax = axes[i, j]

            for direction, color in [("forward", COLOR_FWD), ("reversed", COLOR_REV)]:
                for attn_only in [False, True]:
                    for norm in ["none", "LN"]:
                        mask = (
                            (df_curves["n_layer"] == n_layer) &
                            (df_curves["n_embd"] == n_embd) &
                            (df_curves["attn_only"] == attn_only) &
                            (df_curves["norm"] == norm) &
                            (df_curves["direction"] == direction)
                        )
                        sub = df_curves[mask].sort_values("step")
                        if len(sub) == 0:
                            continue

                        ls = "-" if not attn_only else "--"
                        alpha = 0.9 if norm == "none" else 0.5
                        ax.plot(sub["step"], sub["mean_val_loss"],
                                color=color, linestyle=ls, linewidth=0.8, alpha=alpha)

            ax.axhline(ENTROPY_RATE, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_title(f"L={n_layer}, d={n_embd}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("Step")
            if j == 0:
                ax.set_ylabel("Val Loss")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_FWD, label="Forward"),
        Line2D([0], [0], color=COLOR_REV, label="Reversed"),
        Line2D([0], [0], color="gray", linestyle="-", label="Full"),
        Line2D([0], [0], color="gray", linestyle="--", label="Attn-only"),
    ]
    axes[0, -1].legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.suptitle("Training Dynamics: All Configs", fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, "loss_curves_grid_all.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_convergence_scatter(df_paired, fig_dir):
    """Figure 2: tau_50(fwd) vs tau_50(rev) scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "tau_50", r"$\tau_{50}$ (steps to 50% improvement)"),
        (axes[1], "tau_90", r"$\tau_{90}$ (steps to 90% improvement)"),
    ]:
        for _, row in df_paired.iterrows():
            c = SCATTER_COLORS.get(row.n_embd, "gray")
            m = MARKERS.get(row.n_layer, "D")
            ax.scatter(row[f"{metric}_fwd"], row[f"{metric}_rev"],
                       color=c, marker=m, s=60, alpha=0.8)

        # Diagonal
        all_vals = pd.concat([df_paired[f"{metric}_fwd"], df_paired[f"{metric}_rev"]])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.4, label="fwd = rev")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Forward {metric}")
        ax.set_ylabel(f"Reversed {metric}")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # Legend for colors/markers
    for n_embd, c in SCATTER_COLORS.items():
        axes[0].scatter([], [], color=c, label=f"d={n_embd}", s=60)
    for n_layer, m in MARKERS.items():
        axes[0].scatter([], [], color="gray", marker=m, label=f"L={n_layer}", s=60)
    axes[0].legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    path = os.path.join(fig_dir, "convergence_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_delta_epoch1_vs_final(df_paired, fig_dir):
    """Figure 3: Delta_epoch1 vs Delta_final — do early gaps close?"""
    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in df_paired.iterrows():
        c = SCATTER_COLORS.get(row.n_embd, "gray")
        m = MARKERS.get(row.n_layer, "D")
        ax.scatter(row["delta_epoch1_loss"], row["delta_L_final"],
                   color=c, marker=m, s=60, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\Delta$ epoch-1 loss (rev $-$ fwd)")
    ax.set_ylabel(r"$\Delta$ final loss (rev $-$ fwd)")
    ax.set_title("Early-training gap vs final gap")
    ax.grid(alpha=0.3)

    # Legend
    for n_embd, c in SCATTER_COLORS.items():
        ax.scatter([], [], color=c, label=f"d={n_embd}", s=60)
    for n_layer, m in MARKERS.items():
        ax.scatter([], [], color="gray", marker=m, label=f"L={n_layer}", s=60)
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    path = os.path.join(fig_dir, "delta_epoch1_vs_final.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_delta_by_arch(df_paired, fig_dir):
    """Figure 4: Delta_tau_50 by architecture (bar chart)."""
    df_plot = df_paired.sort_values(["n_layer", "n_embd", "attn_only", "norm"])
    labels = [
        f"L{int(row.n_layer)}_d{int(row.n_embd)}_"
        f"{'attn' if row.attn_only else 'full'}_{row.norm}"
        for _, row in df_plot.iterrows()
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(labels) * 0.35), 12), sharex=True)

    for ax, metric, ylabel, title in [
        (axes[0], "delta_tau_50", r"$\Delta\tau_{50}$ (steps)", "Convergence speed gap"),
        (axes[1], "delta_AUC", r"$\Delta$ AUC", "Excess training cost gap"),
        (axes[2], "delta_epoch1_loss", r"$\Delta$ epoch-1 loss", "Early-training loss gap"),
    ]:
        vals = df_plot[metric].values
        bar_colors = [COLOR_REV if d > 0 else COLOR_FWD for d in vals]
        ax.bar(range(len(labels)), vals, color=bar_colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)

    fig.suptitle(r"Forward–Reversed gaps by architecture (red = reversed slower/higher)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "delta_by_arch.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_delta_vs_architecture(df_paired, fig_dir):
    """Figure 5: Delta_tau_50 vs n_layer and n_params (H7)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Delta_tau_50 vs n_layer
    ax = axes[0]
    for attn_only, marker, label in [(True, "^", "attn_only"), (False, "o", "full")]:
        sub = df_paired[df_paired["attn_only"] == attn_only]
        for n_embd, color in SCATTER_COLORS.items():
            ss = sub[sub["n_embd"] == n_embd]
            if len(ss) > 0:
                ax.scatter(ss["n_layer"], ss["delta_tau_50"],
                           color=color, marker=marker, s=60, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("n_layer")
    ax.set_ylabel(r"$\Delta\tau_{50}$ (rev $-$ fwd)")
    ax.set_title(r"H7: $\Delta\tau_{50}$ vs depth")
    ax.grid(alpha=0.3)

    # Delta_tau_50 vs n_params
    ax = axes[1]
    for attn_only, marker, label in [(True, "^", "attn_only"), (False, "o", "full")]:
        sub = df_paired[df_paired["attn_only"] == attn_only]
        ax.scatter(sub["n_params"], sub["delta_tau_50"],
                   marker=marker, alpha=0.7, label=label)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Approx. parameter count (log scale)")
    ax.set_ylabel(r"$\Delta\tau_{50}$ (rev $-$ fwd)")
    ax.set_title(r"H7: $\Delta\tau_{50}$ vs model size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Delta_AUC vs n_params
    ax = axes[2]
    for attn_only, marker, label in [(True, "^", "attn_only"), (False, "o", "full")]:
        sub = df_paired[df_paired["attn_only"] == attn_only]
        ax.scatter(sub["n_params"], sub["delta_AUC"],
                   marker=marker, alpha=0.7, label=label)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Approx. parameter count (log scale)")
    ax.set_ylabel(r"$\Delta$ AUC (rev $-$ fwd)")
    ax.set_title(r"$\Delta$ AUC vs model size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "delta_vs_architecture.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_early_dynamics_zoom(df_curves, fig_dir):
    """Figure 6: Zoom into first 1000 steps to check Phase I (unigram)."""
    layers = sorted(df_curves["n_layer"].unique())
    embds = sorted(df_curves["n_embd"].unique())

    fig, axes = plt.subplots(len(layers), len(embds),
                             figsize=(4 * len(embds), 3.5 * len(layers)),
                             sharex=True, sharey=True)
    if len(layers) == 1:
        axes = axes[np.newaxis, :]
    if len(embds) == 1:
        axes = axes[:, np.newaxis]

    early = df_curves[df_curves["step"] <= 1000]

    for i, n_layer in enumerate(layers):
        for j, n_embd in enumerate(embds):
            ax = axes[i, j]
            for direction, color, label in [
                ("forward", COLOR_FWD, "Forward"),
                ("reversed", COLOR_REV, "Reversed"),
            ]:
                mask = (
                    (early["n_layer"] == n_layer) &
                    (early["n_embd"] == n_embd) &
                    (early["direction"] == direction)
                )
                sub = early[mask].sort_values("step")
                if len(sub) == 0:
                    continue
                curve = sub.groupby("step").agg(
                    val_loss=("mean_val_loss", "mean"),
                ).reset_index()
                ax.plot(curve["step"], curve["val_loss"],
                        color=color, label=label, linewidth=1.2)

            ax.set_title(f"L={n_layer}, d={n_embd}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(layers) - 1:
                ax.set_xlabel("Step")
            if j == 0:
                ax.set_ylabel("Val Loss")
            if i == 0 and j == len(embds) - 1:
                ax.legend(fontsize=8)

    fig.suptitle("Early Training Dynamics (first 1000 steps)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, "early_dynamics_zoom.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_statistical_tests(df_paired):
    """Run statistical tests on paired deltas."""
    results = {}

    for metric in ["delta_tau_50", "delta_tau_90", "delta_AUC", "delta_epoch1_loss"]:
        vals = df_paired[metric].dropna().values
        if len(vals) < 3:
            continue

        # Paired t-test (vs 0)
        t_stat, p_ttest = stats.ttest_1samp(vals, 0)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(vals)
        except ValueError:
            w_stat, p_wilcoxon = np.nan, np.nan

        n_pos = (vals > 0).sum()

        results[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "n_positive": int(n_pos),
            "n_total": int(len(vals)),
            "frac_positive": float(n_pos / len(vals)),
            "t_statistic": float(t_stat),
            "p_ttest": float(p_ttest),
            "w_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "p_wilcoxon": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
        }

    # Correlations (H7)
    corr_results = {}
    for delta_metric in ["delta_tau_50", "delta_AUC"]:
        for arch_metric in ["n_params", "n_layer", "n_embd"]:
            vals_x = df_paired[arch_metric].values
            vals_y = df_paired[delta_metric].dropna().values
            mask = ~np.isnan(vals_y)
            if mask.sum() > 3:
                r_pearson, p_pearson = stats.pearsonr(vals_x[mask], vals_y[mask])
                r_spearman, p_spearman = stats.spearmanr(vals_x[mask], vals_y[mask])
                corr_results[f"{delta_metric}_vs_{arch_metric}"] = {
                    "pearson_r": float(r_pearson),
                    "pearson_p": float(p_pearson),
                    "spearman_r": float(r_spearman),
                    "spearman_p": float(p_spearman),
                }

    results["correlations"] = corr_results
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze training dynamics fwd vs rev")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--fig_dir", default=FIG_DIR)
    parser.add_argument("--out_path", default=OUT_PATH)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Load data
    print("Loading data...")
    df_loss, df_epoch, df_meta = load_data(args.data_dir)
    print(f"  {len(df_meta)} runs, {len(df_loss)} loss curve points")

    # Compute per-run metrics
    print("\nComputing convergence metrics...")
    df_metrics = compute_all_metrics(df_loss)
    print(f"  Computed metrics for {len(df_metrics)} runs")

    # Aggregate by config (mean over seeds)
    df_agg = aggregate_by_config(df_metrics)
    print(f"  Aggregated to {len(df_agg)} config-direction combos")

    # Pair forward/reversed
    df_paired = pair_configs(df_agg)
    print(f"  Matched {len(df_paired)} architecture pairs")

    # Aggregate loss curves for plotting
    df_curves = aggregate_loss_curves(df_loss)

    # Print summary
    print(f"\n{'='*60}")
    print("H5: Transient Dynamics Asymmetry")
    print(f"{'='*60}")
    for metric in ["delta_tau_50", "delta_tau_90", "delta_AUC", "delta_epoch1_loss"]:
        vals = df_paired[metric].dropna()
        n_pos = (vals > 0).sum()
        print(f"  {metric:>20}: mean = {vals.mean():+.1f}, "
              f"std = {vals.std():.1f}, "
              f"positive in {n_pos}/{len(vals)} configs")

    print(f"\n{'='*60}")
    print("H7: Architecture Dependence")
    print(f"{'='*60}")
    for metric in ["delta_tau_50", "delta_AUC"]:
        for arch in ["n_layer", "n_embd", "n_params"]:
            mask = ~df_paired[metric].isna()
            if mask.sum() > 3:
                r, p = stats.pearsonr(
                    df_paired.loc[mask, arch].values,
                    df_paired.loc[mask, metric].values,
                )
                print(f"  Corr({metric}, {arch}) = {r:.3f} (p={p:.3f})")

    # Statistical tests
    print(f"\n{'='*60}")
    print("Statistical Tests")
    print(f"{'='*60}")
    test_results = run_statistical_tests(df_paired)
    for metric, res in test_results.items():
        if metric == "correlations":
            continue
        print(f"\n  {metric}:")
        print(f"    mean = {res['mean']:+.2f}, median = {res['median']:+.2f}")
        print(f"    positive: {res['n_positive']}/{res['n_total']} ({res['frac_positive']:.1%})")
        print(f"    t-test: t={res['t_statistic']:.2f}, p={res['p_ttest']:.4f}")
        if res.get("p_wilcoxon") is not None:
            print(f"    Wilcoxon: W={res['w_statistic']:.1f}, p={res['p_wilcoxon']:.4f}")

    # Sanity check: compare final val_loss with existing analysis
    # Use best-of selection (like existing analysis) for apples-to-apples comparison
    print(f"\n{'='*60}")
    print("Sanity Check: Final Val Loss Cross-Validation")
    print(f"{'='*60}")
    key_cols = ["n_layer", "n_embd", "attn_only", "norm"]
    fwd_best = df_meta[df_meta["direction"] == "forward"]
    fwd_best = fwd_best.loc[fwd_best.groupby(key_cols)["best_val_loss"].idxmin()]
    rev_best = df_meta[df_meta["direction"] == "reversed"]
    rev_best = rev_best.loc[rev_best.groupby(key_cols)["best_val_loss"].idxmin()]
    sanity_merged = pd.merge(
        fwd_best[key_cols + ["best_val_loss"]].rename(columns={"best_val_loss": "loss_fwd"}),
        rev_best[key_cols + ["best_val_loss"]].rename(columns={"best_val_loss": "loss_rev"}),
        on=key_cols, how="inner",
    )
    sanity_merged["delta"] = sanity_merged["loss_rev"] - sanity_merged["loss_fwd"]
    print(f"  This analysis (best-of): {len(sanity_merged)} matched, "
          f"mean Δ = {sanity_merged['delta'].mean():.4f}")

    existing_path = "out/time_reversal_analysis.json"
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        # Filter existing to same n_embd range
        existing_df = pd.DataFrame(existing["records"])
        existing_filtered = existing_df[existing_df["n_embd"].isin(df_meta["n_embd"].unique())]
        print(f"  Existing analysis (same n_embd): {len(existing_filtered)} matched, "
              f"mean Δ = {existing_filtered['delta'].mean():.4f}")
    else:
        print(f"  {existing_path} not found — skipping cross-validation")

    print(f"\n  Note: seed-averaged Δ(L_final) = {df_paired['delta_L_final'].mean():.4f}")
    print(f"  (differs from best-of due to seed count asymmetry: "
          f"fwd has more seeds for large configs)")

    # Generate plots
    print(f"\n{'='*60}")
    print("Generating Figures")
    print(f"{'='*60}")
    plot_loss_curves_grid(df_curves, args.fig_dir)
    plot_loss_curves_by_norm(df_curves, args.fig_dir)
    plot_convergence_scatter(df_paired, args.fig_dir)
    plot_delta_epoch1_vs_final(df_paired, args.fig_dir)
    plot_delta_by_arch(df_paired, args.fig_dir)
    plot_delta_vs_architecture(df_paired, args.fig_dir)
    plot_early_dynamics_zoom(df_curves, args.fig_dir)

    # Save summary JSON
    summary = {
        "n_runs": int(len(df_meta)),
        "n_forward": int((df_meta["direction"] == "forward").sum()),
        "n_reversed": int((df_meta["direction"] == "reversed").sum()),
        "n_matched_configs": int(len(df_paired)),
        "statistical_tests": test_results,
        "paired_records": df_paired.to_dict(orient="records"),
        "per_run_metrics": df_metrics.to_dict(orient="records"),
    }
    with open(args.out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {args.out_path}")


if __name__ == "__main__":
    main()
