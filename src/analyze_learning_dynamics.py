"""
Analyze multi-seed learning dynamics: forward vs reversed CylinderGraph HMM.

Reads the consolidated loss curves from learning_dynamics_experiment.py
and produces:
1. Mean +/- std loss curves for forward vs reverse
2. Convergence speed metrics (steps to threshold)
3. Variance analysis across seeds
4. Statistical tests for dynamics asymmetry
5. Connection to theoretical predictions (crypticity → slower convergence)

Usage:
    python src/analyze_learning_dynamics.py
    python src/analyze_learning_dynamics.py --curves out/learning_dynamics/all_curves.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))


CURVES_PATH = "out/learning_dynamics/all_curves.json"
FIG_DIR = "figures/learning_dynamics"
OUT_PATH = "out/learning_dynamics_analysis.json"

COLOR_FWD = "#1f77b4"
COLOR_REV = "#d62728"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze multi-seed learning dynamics")
    parser.add_argument("--curves", type=str, default=CURVES_PATH,
                        help="Path to consolidated curves JSON")
    parser.add_argument("--fig_dir", type=str, default=FIG_DIR)
    parser.add_argument("--output", type=str, default=OUT_PATH)
    parser.add_argument("--entropy_rate", type=float, default=None,
                        help="Entropy rate h_X (auto-detected if not set)")
    parser.add_argument("--threshold_margin", type=float, default=0.01,
                        help="Threshold margin above entropy rate for convergence metric")
    return parser.parse_args()


def load_curves(path):
    """Load consolidated curves and align to common step grid."""
    with open(path) as f:
        data = json.load(f)

    result = {}
    for direction in ["forward", "reversed"]:
        curves = data[direction]
        if not curves:
            result[direction] = {"steps": np.array([]), "losses": np.array([])}
            continue

        # Find common step grid (intersection of all runs' steps)
        all_steps = [np.array(c["steps"]) for c in curves.values()]
        all_losses = [np.array(c["val_losses"]) for c in curves.values()]

        # Use the step grid from the first run (they should all be the same
        # since they use the same config and dataset size)
        ref_steps = all_steps[0]

        # Interpolate all curves to the reference grid
        aligned_losses = []
        for steps, losses in zip(all_steps, all_losses):
            if np.array_equal(steps, ref_steps):
                aligned_losses.append(losses)
            else:
                # Linear interpolation for misaligned grids
                interp_losses = np.interp(ref_steps, steps, losses)
                aligned_losses.append(interp_losses)

        result[direction] = {
            "steps": ref_steps,
            "losses": np.array(aligned_losses),  # (n_seeds, n_steps)
            "n_seeds": len(aligned_losses),
        }

    return result


def compute_convergence_metrics(steps, losses, entropy_rate, margin):
    """
    Compute convergence metrics for each seed's loss curve.

    Args:
        steps: (n_steps,) step numbers
        losses: (n_seeds, n_steps) val_loss curves
        entropy_rate: h_X in nats
        margin: threshold = entropy_rate + margin

    Returns:
        dict with per-seed and aggregate metrics
    """
    threshold = entropy_rate + margin
    n_seeds = losses.shape[0]

    steps_to_threshold = []
    auc_excess = []
    early_losses = []  # loss at step 100

    for i in range(n_seeds):
        curve = losses[i]

        # Steps to reach threshold
        below = np.where(curve <= threshold)[0]
        if len(below) > 0:
            steps_to_threshold.append(float(steps[below[0]]))
        else:
            steps_to_threshold.append(float(steps[-1]))

        # AUC of excess loss above entropy rate
        excess = np.maximum(curve - entropy_rate, 0)
        auc = float(np.trapezoid(excess, steps))
        auc_excess.append(auc)

        # Early loss (at step closest to 100)
        idx_100 = np.argmin(np.abs(steps - 100))
        early_losses.append(float(curve[idx_100]))

    return {
        "steps_to_threshold": np.array(steps_to_threshold),
        "auc_excess": np.array(auc_excess),
        "early_losses": np.array(early_losses),
        "threshold": threshold,
    }


def run_statistical_tests(metrics_fwd, metrics_rev):
    """Run paired statistical tests on convergence metrics."""
    results = {}

    for name in ["steps_to_threshold", "auc_excess", "early_losses"]:
        vals_fwd = metrics_fwd[name]
        vals_rev = metrics_rev[name]

        n_fwd = len(vals_fwd)
        n_rev = len(vals_rev)

        # Two-sample t-test (not paired since seeds may differ)
        t_stat, p_ttest = stats.ttest_ind(vals_fwd, vals_rev)

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_mann_whitney = stats.mannwhitneyu(vals_fwd, vals_rev, alternative="two-sided")
        except ValueError:
            u_stat, p_mann_whitney = np.nan, np.nan

        # Bootstrap test for difference of means
        n_bootstrap = 10000
        combined = np.concatenate([vals_fwd, vals_rev])
        observed_diff = np.mean(vals_rev) - np.mean(vals_fwd)
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            perm = np.random.permutation(combined)
            d = np.mean(perm[:n_fwd]) - np.mean(perm[n_fwd:n_fwd + n_rev])
            bootstrap_diffs.append(d)
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_bootstrap = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))

        results[name] = {
            "mean_fwd": float(np.mean(vals_fwd)),
            "std_fwd": float(np.std(vals_fwd)),
            "mean_rev": float(np.mean(vals_rev)),
            "std_rev": float(np.std(vals_rev)),
            "diff_mean": float(observed_diff),
            "n_fwd": n_fwd,
            "n_rev": n_rev,
            "t_statistic": float(t_stat),
            "p_ttest": float(p_ttest),
            "u_statistic": float(u_stat) if not np.isnan(u_stat) else None,
            "p_mann_whitney": float(p_mann_whitney) if not np.isnan(p_mann_whitney) else None,
            "p_bootstrap": p_bootstrap,
        }

    return results


def plot_mean_loss_curves(data, entropy_rate, fig_dir):
    """Plot mean +/- std loss curves for forward vs reverse."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Full training curves
    ax = axes[0]
    for direction, color, label in [
        ("forward", COLOR_FWD, "Forward"),
        ("reversed", COLOR_REV, "Reversed"),
    ]:
        d = data[direction]
        if d["losses"].size == 0:
            continue
        steps = d["steps"]
        mean = d["losses"].mean(axis=0)
        std = d["losses"].std(axis=0)

        ax.plot(steps, mean, color=color, label=f"{label} (n={d['n_seeds']})", linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

    if entropy_rate is not None:
        ax.axhline(entropy_rate, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"$h_X = {entropy_rate:.3f}$")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Val loss (nats)")
    ax.set_title("(a) Mean training curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (b): Early dynamics zoom (first 20% of steps)
    ax = axes[1]
    for direction, color, label in [
        ("forward", COLOR_FWD, "Forward"),
        ("reversed", COLOR_REV, "Reversed"),
    ]:
        d = data[direction]
        if d["losses"].size == 0:
            continue
        steps = d["steps"]
        mean = d["losses"].mean(axis=0)
        std = d["losses"].std(axis=0)

        max_step = steps[-1] * 0.2
        mask = steps <= max_step

        ax.plot(steps[mask], mean[mask], color=color, label=label, linewidth=1.5)
        ax.fill_between(steps[mask], (mean - std)[mask], (mean + std)[mask],
                        color=color, alpha=0.15)

    if entropy_rate is not None:
        ax.axhline(entropy_rate, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Val loss (nats)")
    ax.set_title("(b) Early training dynamics (first 20%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"mean_loss_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/mean_loss_curves.{{pdf,png}}")


def plot_excess_loss_curves(data, entropy_rate, fig_dir):
    """Plot excess loss (val_loss - h_X) on log scale."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for direction, color, label in [
        ("forward", COLOR_FWD, "Forward"),
        ("reversed", COLOR_REV, "Reversed"),
    ]:
        d = data[direction]
        if d["losses"].size == 0:
            continue
        steps = d["steps"]
        excess = d["losses"] - entropy_rate  # (n_seeds, n_steps)
        mean_excess = excess.mean(axis=0)

        # Only plot where mean excess is positive
        mask = mean_excess > 1e-4
        ax.plot(steps[mask], mean_excess[mask], color=color, label=label, linewidth=1.5)

        # Show individual seeds as thin lines
        for i in range(min(d["n_seeds"], 5)):
            seed_excess = excess[i]
            seed_mask = seed_excess > 1e-4
            ax.plot(steps[seed_mask], seed_excess[seed_mask],
                    color=color, alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Excess loss (val_loss - $h_X$) (nats)")
    ax.set_title("Excess loss convergence")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"excess_loss_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/excess_loss_curves.{{pdf,png}}")


def plot_convergence_histograms(metrics_fwd, metrics_rev, fig_dir):
    """Plot histograms of convergence metrics across seeds."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, name, xlabel, title in [
        (axes[0], "steps_to_threshold", "Steps", "Steps to convergence"),
        (axes[1], "auc_excess", "AUC (nats * steps)", "Excess training cost"),
        (axes[2], "early_losses", "Loss (nats)", "Loss at step 100"),
    ]:
        vals_fwd = metrics_fwd[name]
        vals_rev = metrics_rev[name]

        bins = np.linspace(
            min(vals_fwd.min(), vals_rev.min()),
            max(vals_fwd.max(), vals_rev.max()),
            15,
        )

        ax.hist(vals_fwd, bins=bins, alpha=0.5, color=COLOR_FWD,
                label=f"Forward (n={len(vals_fwd)})")
        ax.hist(vals_rev, bins=bins, alpha=0.5, color=COLOR_REV,
                label=f"Reversed (n={len(vals_rev)})")

        # Mark means
        ax.axvline(vals_fwd.mean(), color=COLOR_FWD, linestyle="--", linewidth=1.5)
        ax.axvline(vals_rev.mean(), color=COLOR_REV, linestyle="--", linewidth=1.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"convergence_histograms.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/convergence_histograms.{{pdf,png}}")


def plot_variance_evolution(data, fig_dir):
    """Plot how loss variance across seeds evolves during training."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for direction, color, label in [
        ("forward", COLOR_FWD, "Forward"),
        ("reversed", COLOR_REV, "Reversed"),
    ]:
        d = data[direction]
        if d["losses"].size == 0 or d["n_seeds"] < 2:
            continue
        steps = d["steps"]
        variance = d["losses"].var(axis=0)
        ax.plot(steps, variance, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Var(val_loss) across seeds")
    ax.set_title("Loss variance evolution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"variance_evolution.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/variance_evolution.{{pdf,png}}")


def plot_all_seeds(data, entropy_rate, fig_dir):
    """Plot all individual seed curves for visual inspection."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, direction, color, title in [
        (axes[0], "forward", COLOR_FWD, "Forward (all seeds)"),
        (axes[1], "reversed", COLOR_REV, "Reversed (all seeds)"),
    ]:
        d = data[direction]
        if d["losses"].size == 0:
            continue

        for i in range(d["n_seeds"]):
            ax.plot(d["steps"], d["losses"][i], color=color, alpha=0.3, linewidth=0.7)

        # Mean in black
        mean = d["losses"].mean(axis=0)
        ax.plot(d["steps"], mean, color="black", linewidth=2, label="Mean")

        if entropy_rate is not None:
            ax.axhline(entropy_rate, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Val loss (nats)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"all_seeds.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir}/all_seeds.{{pdf,png}}")


def estimate_entropy_rate():
    """Estimate entropy rate from the HMM config."""
    try:
        import yaml
        from utils import create_process_from_dict
        with open("config/base_config.yaml") as f:
            cfg = yaml.safe_load(f)
        proc = create_process_from_dict(cfg["data_generator"]["process"])
        h = proc.entropy_rate_empirical_estimate(100000, burn_in=1000)
        print(f"Estimated entropy rate h_X = {h:.6f} nats")
        return h
    except Exception as e:
        print(f"Could not estimate entropy rate: {e}")
        return None


def main():
    args = parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load curves
    print(f"Loading curves from {args.curves}")
    data = load_curves(args.curves)

    n_fwd = data["forward"].get("n_seeds", 0) if data["forward"]["losses"].size > 0 else 0
    n_rev = data["reversed"].get("n_seeds", 0) if data["reversed"]["losses"].size > 0 else 0
    print(f"  Forward: {n_fwd} seeds, Reversed: {n_rev} seeds")

    if n_fwd == 0 and n_rev == 0:
        print("ERROR: No curves found. Run learning_dynamics_experiment.py first.")
        return

    # Entropy rate
    entropy_rate = args.entropy_rate
    if entropy_rate is None:
        entropy_rate = estimate_entropy_rate()

    # === Convergence metrics ===
    if n_fwd > 0 and entropy_rate is not None:
        metrics_fwd = compute_convergence_metrics(
            data["forward"]["steps"], data["forward"]["losses"],
            entropy_rate, args.threshold_margin)
    else:
        metrics_fwd = None

    if n_rev > 0 and entropy_rate is not None:
        metrics_rev = compute_convergence_metrics(
            data["reversed"]["steps"], data["reversed"]["losses"],
            entropy_rate, args.threshold_margin)
    else:
        metrics_rev = None

    # === Statistical tests ===
    test_results = None
    if metrics_fwd is not None and metrics_rev is not None:
        print("\n" + "=" * 60)
        print("Statistical Tests: Forward vs Reversed Learning Dynamics")
        print("=" * 60)

        test_results = run_statistical_tests(metrics_fwd, metrics_rev)

        for name, res in test_results.items():
            print(f"\n  {name}:")
            print(f"    Forward:  {res['mean_fwd']:.4f} +/- {res['std_fwd']:.4f} (n={res['n_fwd']})")
            print(f"    Reversed: {res['mean_rev']:.4f} +/- {res['std_rev']:.4f} (n={res['n_rev']})")
            print(f"    Diff (rev-fwd): {res['diff_mean']:+.4f}")
            print(f"    t-test: t={res['t_statistic']:.3f}, p={res['p_ttest']:.4f}")
            if res.get("p_mann_whitney") is not None:
                print(f"    Mann-Whitney: p={res['p_mann_whitney']:.4f}")
            print(f"    Bootstrap: p={res['p_bootstrap']:.4f}")

    # === Null check: final val_loss ===
    print("\n" + "=" * 60)
    print("Null Check: Final Val Loss (should be ~ equal)")
    print("=" * 60)
    if n_fwd > 0:
        final_fwd = data["forward"]["losses"][:, -1]
        print(f"  Forward:  {final_fwd.mean():.6f} +/- {final_fwd.std():.6f}")
    if n_rev > 0:
        final_rev = data["reversed"]["losses"][:, -1]
        print(f"  Reversed: {final_rev.mean():.6f} +/- {final_rev.std():.6f}")
    if n_fwd > 0 and n_rev > 0:
        diff = final_rev.mean() - final_fwd.mean()
        _, p = stats.ttest_ind(final_fwd, final_rev)
        print(f"  Diff: {diff:+.6f}, p={p:.4f}")

    # === Generate plots ===
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    plot_mean_loss_curves(data, entropy_rate, args.fig_dir)
    if entropy_rate is not None:
        plot_excess_loss_curves(data, entropy_rate, args.fig_dir)
    if metrics_fwd is not None and metrics_rev is not None:
        plot_convergence_histograms(metrics_fwd, metrics_rev, args.fig_dir)
    plot_variance_evolution(data, args.fig_dir)
    plot_all_seeds(data, entropy_rate, args.fig_dir)

    # === Save results ===
    summary = {
        "n_seeds_forward": n_fwd,
        "n_seeds_reversed": n_rev,
        "entropy_rate": entropy_rate,
        "threshold_margin": args.threshold_margin,
    }

    if test_results is not None:
        summary["statistical_tests"] = test_results

    if n_fwd > 0:
        summary["forward_final_loss"] = {
            "mean": float(data["forward"]["losses"][:, -1].mean()),
            "std": float(data["forward"]["losses"][:, -1].std()),
        }
    if n_rev > 0:
        summary["reversed_final_loss"] = {
            "mean": float(data["reversed"]["losses"][:, -1].mean()),
            "std": float(data["reversed"]["losses"][:, -1].std()),
        }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # === Theory connection ===
    if test_results is not None:
        print("\n" + "=" * 60)
        print("Theory Connection Summary")
        print("=" * 60)
        stt = test_results.get("steps_to_threshold", {})
        if stt.get("diff_mean", 0) > 0:
            print("  Reversed converges SLOWER than forward.")
            print("  --> Consistent with higher reverse crypticity (H2).")
        elif stt.get("diff_mean", 0) < 0:
            print("  Forward converges SLOWER than reversed.")
            print("  --> Consistent with higher forward crypticity (H2).")
        else:
            print("  No significant difference in convergence speed.")

        auc = test_results.get("auc_excess", {})
        if auc.get("p_ttest", 1) < 0.05:
            print(f"  AUC difference is statistically significant (p={auc['p_ttest']:.4f}).")
        else:
            print(f"  AUC difference is NOT statistically significant (p={auc.get('p_ttest', 1):.4f}).")

        print("\n  To complete the theory-experiment connection:")
        print("  Compare the direction with higher crypticity (from epsilon_machine_analysis.py)")
        print("  to the direction with slower convergence (from this analysis).")


if __name__ == "__main__":
    main()
