"""
Plot Bayes-optimal loss and excess loss for forward vs reverse HMM.

Reads the output of reverse_hmm_analysis.py (k=16, 50k samples) and generates
a two-panel figure showing:
  (a) Bayes-optimal loss L*(k) for both directions
  (b) Excess Bayes-optimal loss L*(k) - h_X (convergence asymmetry)

Usage:
    python src/plot_bayes_optimal_gap.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

INPUT_PATH = "out/bayes_optimal_k16/reverse_hmm_analysis_k16.json"
FIG_DIR = "figures/time_reversal"

COLOR_FWD = "#1f77b4"
COLOR_REV = "#d62728"


def estimate_empirical_entropy_rate():
    """Compute the empirical observation entropy rate h_X via the forward algorithm."""
    import yaml
    from utils import create_process_from_dict

    with open("config/base_config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    return proc.entropy_rate_empirical_estimate(100000, burn_in=1000)


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    os.makedirs(FIG_DIR, exist_ok=True)

    k = np.arange(1, data["max_context"] + 1)
    loss_fwd = np.array(data["bayes_optimal_loss_forward"])
    loss_rev = np.array(data["bayes_optimal_loss_reverse"])

    # Empirical observation entropy rate (not the joint H(X,S'|S))
    print("Estimating empirical entropy rate h_X...")
    h_X = estimate_empirical_entropy_rate()
    print(f"  h_X = {h_X:.6f} nats")
    print(f"  (cf. joint H(X,S'|S) = {data['entropy_rate_forward']:.6f} from JSON)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): Bayes-optimal loss
    ax = axes[0]
    ax.plot(k, loss_fwd, "o-", color=COLOR_FWD, label="Forward", markersize=4)
    ax.plot(k, loss_rev, "s-", color=COLOR_REV, label="Reversed", markersize=4)
    ax.axhline(h_X, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$h_X = {h_X:.3f}$")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel("Bayes-optimal loss $L^*(k)$ (nats)")
    ax.set_title("(a) Bayes-optimal loss by context length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): Excess loss = L*(k) - h_X
    # At large k, Monte Carlo noise causes L*(k) to fluctuate below h_X.
    # Truncate to k where both excess values are above a noise floor.
    ax = axes[1]
    excess_fwd = loss_fwd - h_X
    excess_rev = loss_rev - h_X
    noise_floor = 5e-3  # below this, MC noise dominates
    k_valid = np.where((excess_fwd > noise_floor) & (excess_rev > noise_floor))[0]
    k_max = k_valid[-1] + 1 if len(k_valid) > 0 else len(k)
    k_plot = k[:k_max]
    ax.plot(k_plot, excess_fwd[:k_max], "o-", color=COLOR_FWD, label="Forward", markersize=4)
    ax.plot(k_plot, excess_rev[:k_max], "s-", color=COLOR_REV, label="Reversed", markersize=4)
    ax.axhspan(0, noise_floor, color="gray", alpha=0.08)
    ax.axhline(noise_floor, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(k_plot[-1] + 0.3, noise_floor * 1.3, "MC noise floor", fontsize=7,
            color="gray", va="bottom")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel(r"Excess loss $L^*(k) - h_X$ (nats)")
    ax.set_title("(b) Convergence to entropy rate")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "bayes_optimal_gap_k16.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
