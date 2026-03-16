"""
Theoretical verification of DirectedCycleHMM forward/reverse asymmetry.

Sweeps over bias values and computes Bayes-optimal loss curves for forward
and reverse HMMs. Verifies entropy rate invariance and plots the theoretical
gap as a function of bias.

Usage:
    python scripts/verify_directed_cycle.py
    python scripts/verify_directed_cycle.py --num_samples 100000 --max_context 20
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmm import DirectedCycleHMM
from reverse_hmm_analysis import (
    build_reverse_emission_matrices,
    compute_bayes_optimal_loss_by_context,
    compute_entropy_rate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify DirectedCycleHMM forward/reverse asymmetry")
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--emission_noise", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--max_context", type=int, default=17)
    parser.add_argument("--output_dir", type=str, default="figures/directed_cycle")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bias_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = {}

    for bias in bias_values:
        print(f"\n{'='*60}")
        print(f"bias = {bias}")
        print(f"{'='*60}")

        hmm = DirectedCycleHMM(num_states=args.num_states, bias=bias, emission_noise=args.emission_noise)
        E_fwd = hmm.emission_matrices
        pi = hmm.get_stationary_distribution()

        # Sanity: emission matrices sum to 1 per source state
        row_sums = E_fwd.sum(axis=(0, 2))
        assert np.allclose(row_sums, 1.0, atol=1e-10), f"E_fwd row sums: {row_sums}"
        print(f"  E_fwd row sums: OK (all 1.0)")
        print(f"  Stationary dist: {pi}")

        # Build reverse
        E_rev = build_reverse_emission_matrices(E_fwd, pi)
        rev_row_sums = E_rev.sum(axis=(0, 2))
        assert np.allclose(rev_row_sums, 1.0, atol=1e-10), f"E_rev row sums: {rev_row_sums}"
        print(f"  E_rev row sums: OK (all 1.0)")

        # Entropy rates
        h_fwd = compute_entropy_rate(E_fwd, pi)
        h_rev = compute_entropy_rate(E_rev, pi)
        print(f"  Entropy rate (fwd): {h_fwd:.8f} nats")
        print(f"  Entropy rate (rev): {h_rev:.8f} nats")
        print(f"  |Diff|: {abs(h_fwd - h_rev):.2e}")
        assert abs(h_fwd - h_rev) < 1e-6, f"Entropy rate mismatch: {h_fwd} vs {h_rev}"

        # Bayes-optimal loss
        loss_fwd = compute_bayes_optimal_loss_by_context(
            E_fwd, pi, args.num_samples, args.max_context, desc=f"fwd bias={bias}")
        loss_rev = compute_bayes_optimal_loss_by_context(
            E_rev, pi, args.num_samples, args.max_context, desc=f"rev bias={bias}")

        # Print table
        print(f"\n  {'k':>3} | {'L*_fwd':>10} | {'L*_rev':>10} | {'Diff':>10}")
        print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for k in range(args.max_context):
            diff = loss_rev[k] - loss_fwd[k]
            print(f"  {k:>3} | {loss_fwd[k]:>10.6f} | {loss_rev[k]:>10.6f} | {diff:>+10.6f}")

        results[bias] = {
            "loss_fwd": loss_fwd,
            "loss_rev": loss_rev,
            "h_fwd": h_fwd,
            "h_rev": h_rev,
        }

    # === Plot 1: Bayes-optimal excess loss curves for each bias ===
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    ks = np.arange(args.max_context)

    for idx, bias in enumerate(bias_values):
        ax = axes[idx]
        r = results[bias]
        h = r["h_fwd"]
        ax.plot(ks, r["loss_fwd"] - h, "o-", label="Forward", markersize=3)
        ax.plot(ks, r["loss_rev"] - h, "s-", label="Reverse", markersize=3)
        ax.set_title(f"bias = {bias}")
        ax.set_ylabel("Excess loss (nats)")
        ax.set_xlabel("Context length k")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=-0.01)

    fig.suptitle(
        f"DirectedCycleHMM (N={args.num_states}, noise={args.emission_noise}): "
        "Bayes-optimal excess loss",
        fontsize=12,
    )
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "bayes_optimal_excess_loss.pdf")
    fig.savefig(path1, bbox_inches="tight")
    print(f"\nSaved: {path1}")

    # Also save PNG
    path1_png = os.path.join(args.output_dir, "bayes_optimal_excess_loss.png")
    fig.savefig(path1_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1_png}")
    plt.close(fig)

    # === Plot 2: Integrated gap vs bias ===
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    integrated_gaps = []
    for bias in bias_values:
        r = results[bias]
        gap = np.sum(r["loss_rev"] - r["loss_fwd"])
        integrated_gaps.append(gap)

    ax2.plot(bias_values, integrated_gaps, "ko-", markersize=6)
    ax2.set_xlabel("Bias (forward transition probability)")
    ax2.set_ylabel(r"$\sum_k [L^*_{\mathrm{rev}}(k) - L^*_{\mathrm{fwd}}(k)]$ (nats)")
    ax2.set_title(
        f"DirectedCycleHMM (N={args.num_states}, noise={args.emission_noise}): "
        "Integrated asymmetry gap"
    )
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    path2 = os.path.join(args.output_dir, "integrated_gap_vs_bias.pdf")
    fig2.savefig(path2, bbox_inches="tight")
    print(f"Saved: {path2}")

    path2_png = os.path.join(args.output_dir, "integrated_gap_vs_bias.png")
    fig2.savefig(path2_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2_png}")
    plt.close(fig2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Integrated gap (sum of L*_rev - L*_fwd over k=0..{})".format(args.max_context - 1))
    print(f"{'='*60}")
    for bias, gap in zip(bias_values, integrated_gaps):
        print(f"  bias={bias:.2f}:  gap = {gap:+.4f} nats")


if __name__ == "__main__":
    main()
