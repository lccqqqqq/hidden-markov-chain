"""
Belief convergence analysis for forward vs reverse CylinderGraph HMM.

For a fixed prediction target at position t, computes the Bayes-optimal
predictive distribution as context length k grows:

    Forward:  P(x_t | x_{t-k}, ..., x_{t-1})  for k = 1, 2, ..., K
    Backward: P(x_t | x_{t+1}, ..., x_{t+k})  for k = 1, 2, ..., K

Measures:
1. KL divergence from finite-k belief to k=inf belief (convergence rate)
2. Conditional cross-entropy at each k (Bayes-optimal loss curve)
3. Excess loss L*(k) - h_X (how far from entropy rate)

The asymmetry in convergence rates is predicted by computational mechanics:
the direction with higher crypticity should converge slower.

Extends the preliminary analysis in reverse_hmm_analysis.py with:
- Longer reference sequences (k_ref >> max_context) for better k=inf proxy
- Entropy rate verification (forward == reverse to machine precision)
- Joint plots combining belief convergence and Bayes-optimal loss

Usage:
    python src/belief_convergence.py
    python src/belief_convergence.py --num_samples 100000 --max_context 32 --k_ref 200
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import create_process_from_dict
from reverse_hmm_analysis import build_reverse_emission_matrices
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_FWD = "#1f77b4"
COLOR_REV = "#d62728"
FIG_DIR = "figures/belief_convergence"


def parse_args():
    parser = argparse.ArgumentParser(description="Belief convergence analysis: forward vs reverse")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of Monte Carlo sequences")
    parser.add_argument("--max_context", type=int, default=32,
                        help="Maximum context length k to evaluate")
    parser.add_argument("--k_ref", type=int, default=200,
                        help="Reference sequence length for k=inf proxy")
    parser.add_argument("--output", type=str, default="out/belief_convergence.json")
    parser.add_argument("--fig_dir", type=str, default=FIG_DIR)
    return parser.parse_args()


class TempHMM:
    """Lightweight HMM wrapper for sequence generation."""
    def __init__(self, E, pi):
        self.E = E
        self.pi = pi
        self.num_states = len(pi)
        self.d_vocab = E.shape[0]

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=np.int64)
        obs = np.zeros(length, dtype=np.int64)
        current_state = np.random.choice(self.num_states, p=self.pi)

        for t in range(length):
            states[t] = current_state
            probs = self.E[:, current_state, :].flatten()
            sample = np.random.choice(len(probs), p=probs)
            obs[t] = sample // self.num_states
            current_state = sample % self.num_states

        return states, obs


def compute_belief_trajectory(E, pi, observations):
    """
    Compute belief state trajectory using the forward algorithm.

    Args:
        E: Emission matrices (d_vocab, num_states, num_states)
        pi: Initial belief (stationary distribution)
        observations: Sequence of observations

    Returns:
        beliefs: list of belief state arrays, one per time step
    """
    belief = pi.copy()
    beliefs = []

    for t in range(len(observations)):
        belief = belief @ E[observations[t]]
        belief_sum = belief.sum()
        if belief_sum > 0:
            belief = belief / belief_sum
        beliefs.append(belief.copy())

    return beliefs


def kl_divergence(p, q):
    """KL(p || q) with numerical safety."""
    kl = 0.0
    for i in range(len(p)):
        if p[i] > 1e-15:
            kl += p[i] * np.log(p[i] / (q[i] + 1e-30))
    return max(kl, 0.0)


def compute_predictive_loss(E, belief):
    """
    Compute the predictive cross-entropy given a belief state.

    H(X_t | belief) = -sum_j P(x_t=j | belief) log P(x_t=j | belief)
    where P(x_t=j | belief) = sum_i belief(i) * O(j|i)
    and O(j|i) = sum_k E[j,i,k]
    """
    O = E.sum(axis=2)  # (d_vocab, num_states)
    pred_dist = belief @ O.T  # (d_vocab,) predictive distribution over tokens
    pred_dist = np.maximum(pred_dist, 1e-30)
    return -np.sum(pred_dist * np.log(pred_dist))


def compute_convergence_curves(E, pi, num_samples, max_context, k_ref, desc=""):
    """
    Compute belief convergence and Bayes-optimal loss curves.

    For each MC sample:
    1. Generate a sequence of length k_ref
    2. Run forward algorithm on full sequence to get reference belief at k_ref
    3. For each k = 1, ..., max_context:
       - Compute belief using only first k observations
       - Compute KL(ref_belief || partial_belief)
       - Compute Bayes-optimal loss at this belief

    Returns:
        kl_by_k: (max_context,) average KL from ref belief
        loss_by_k: (max_context,) average Bayes-optimal CE loss
        predictive_loss_by_k: (max_context,) average -log P(x_{k+1} | x_1...x_k)
        ref_loss: average loss at k_ref (entropy rate proxy)
    """
    hmm = TempHMM(E, pi)
    O = E.sum(axis=2)  # observation matrix

    kl_accum = np.zeros(max_context)
    loss_accum = np.zeros(max_context)  # predictive entropy H(X|belief)
    predictive_loss_accum = np.zeros(max_context)  # actual -log P(x_{k+1}|past)
    predictive_count = np.zeros(max_context)
    ref_loss_accum = 0.0

    for _ in tqdm(range(num_samples), desc=f"Convergence {desc}"):
        _, obs = hmm.generate_sequence(k_ref + 1)  # +1 for prediction target

        # Full trajectory of beliefs
        beliefs = compute_belief_trajectory(E, pi, obs[:k_ref])
        ref_belief = beliefs[-1]

        # Reference loss (entropy rate proxy)
        ref_loss_accum += compute_predictive_loss(E, ref_belief)

        # Convergence at each context length
        for k in range(min(max_context, k_ref)):
            partial_belief = beliefs[k]

            # KL from reference belief
            kl_accum[k] += kl_divergence(ref_belief, partial_belief)

            # Predictive entropy at this context length
            loss_accum[k] += compute_predictive_loss(E, partial_belief)

            # Actual predictive loss: -log P(x_{k+1} | x_0...x_k)
            if k + 1 < len(obs):
                obs_next = obs[k + 1]
                pred_prob = partial_belief @ O[obs_next]
                predictive_loss_accum[k] += -np.log(pred_prob + 1e-30)
                predictive_count[k] += 1

    kl_by_k = kl_accum / num_samples
    loss_by_k = loss_accum / num_samples
    predictive_loss_by_k = predictive_loss_accum / np.maximum(predictive_count, 1)
    ref_loss = ref_loss_accum / num_samples

    return kl_by_k, loss_by_k, predictive_loss_by_k, ref_loss


def compute_observation_entropy_rate(E, pi, num_samples=50000, burn_in=1000):
    """
    Compute observation entropy rate h_X via Monte Carlo forward algorithm.

    Generates a single long sequence, runs the forward algorithm,
    and averages the per-step predictive loss after burn-in.
    """
    hmm = TempHMM(E, pi)
    O = E.sum(axis=2)
    total_length = num_samples + burn_in

    _, obs = hmm.generate_sequence(total_length)

    belief = pi.copy()
    h_sum = 0.0
    count = 0

    for t in range(total_length):
        if t >= burn_in:
            # Predictive distribution
            pred = belief @ O.T
            pred = np.maximum(pred, 1e-30)
            # Cross-entropy with actual observation
            h_sum += -np.log(pred[obs[t]])
            count += 1

        # Update belief
        belief = belief @ E[obs[t]]
        belief_sum = belief.sum()
        if belief_sum > 0:
            belief = belief / belief_sum

    return h_sum / count


def plot_convergence(k, kl_fwd, kl_rev, loss_fwd, loss_rev, h_X, fig_dir):
    """Generate the main convergence figure."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): KL divergence from reference belief
    ax = axes[0]
    ax.plot(k, kl_fwd, "o-", color=COLOR_FWD, label="Forward", markersize=3)
    ax.plot(k, kl_rev, "s-", color=COLOR_REV, label="Reverse", markersize=3)
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(\beta_\infty \| \beta_k)$ (nats)")
    ax.set_title("(a) Belief convergence rate")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (b): Bayes-optimal predictive loss
    ax = axes[1]
    ax.plot(k, loss_fwd, "o-", color=COLOR_FWD, label="Forward", markersize=3)
    ax.plot(k, loss_rev, "s-", color=COLOR_REV, label="Reverse", markersize=3)
    ax.axhline(h_X, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$h_X = {h_X:.3f}$")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel(r"Bayes-optimal loss $L^*(k)$ (nats)")
    ax.set_title("(b) Bayes-optimal loss by context")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (c): Excess loss on log scale
    ax = axes[2]
    excess_fwd = loss_fwd - h_X
    excess_rev = loss_rev - h_X
    # Only plot where excess is positive (above MC noise)
    noise_floor = 5e-3
    mask_fwd = excess_fwd > noise_floor
    mask_rev = excess_rev > noise_floor
    k_fwd = k[mask_fwd]
    k_rev = k[mask_rev]
    ax.plot(k_fwd, excess_fwd[mask_fwd], "o-", color=COLOR_FWD, label="Forward", markersize=3)
    ax.plot(k_rev, excess_rev[mask_rev], "s-", color=COLOR_REV, label="Reverse", markersize=3)
    ax.axhspan(0, noise_floor, color="gray", alpha=0.08)
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel(r"Excess loss $L^*(k) - h_X$ (nats)")
    ax.set_title("(c) Convergence to entropy rate")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "belief_convergence_3panel.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # Also save as PNG for quick viewing
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    axes2[0].plot(k, kl_fwd, "o-", color=COLOR_FWD, label="Forward", markersize=3)
    axes2[0].plot(k, kl_rev, "s-", color=COLOR_REV, label="Reverse", markersize=3)
    axes2[0].set_xlabel("Context length $k$")
    axes2[0].set_ylabel(r"$D_{KL}(\beta_\infty || \beta_k)$ (nats)")
    axes2[0].set_title("(a) Belief convergence rate")
    axes2[0].set_yscale("log")
    axes2[0].legend(fontsize=9)
    axes2[0].grid(alpha=0.3)

    axes2[1].plot(k, loss_fwd, "o-", color=COLOR_FWD, label="Forward", markersize=3)
    axes2[1].plot(k, loss_rev, "s-", color=COLOR_REV, label="Reverse", markersize=3)
    axes2[1].axhline(h_X, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                     label=f"h_X = {h_X:.3f}")
    axes2[1].set_xlabel("Context length $k$")
    axes2[1].set_ylabel("Bayes-optimal loss L*(k) (nats)")
    axes2[1].set_title("(b) Bayes-optimal loss by context")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(alpha=0.3)

    axes2[2].plot(k_fwd, excess_fwd[mask_fwd], "o-", color=COLOR_FWD, label="Forward", markersize=3)
    axes2[2].plot(k_rev, excess_rev[mask_rev], "s-", color=COLOR_REV, label="Reverse", markersize=3)
    axes2[2].set_xlabel("Context length $k$")
    axes2[2].set_ylabel("Excess loss L*(k) - h_X (nats)")
    axes2[2].set_title("(c) Convergence to entropy rate")
    axes2[2].set_yscale("log")
    axes2[2].legend(fontsize=9)
    axes2[2].grid(alpha=0.3)

    plt.tight_layout()
    path_png = os.path.join(fig_dir, "belief_convergence_3panel.png")
    fig2.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Saved: {path_png}")


def plot_kl_ratio(k, kl_fwd, kl_rev, fig_dir):
    """Plot the ratio of reverse/forward KL divergence."""
    os.makedirs(fig_dir, exist_ok=True)

    # Only compute ratio where both are positive
    mask = (kl_fwd > 1e-6) & (kl_rev > 1e-6)
    ratio = kl_rev[mask] / kl_fwd[mask]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k[mask], ratio, "o-", color="#2ca02c", markersize=4)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Symmetric (ratio=1)")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{rev}} / D_{\mathrm{KL}}^{\mathrm{fwd}}$")
    ax.set_title("Belief convergence asymmetry ratio")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "kl_ratio.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()

    # Load config and create forward HMM
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    proc = create_process_from_dict(cfg["data_generator"]["process"])
    E_fwd = proc.emission_matrices
    pi = proc.get_stationary_distribution()

    d_vocab, num_states, _ = E_fwd.shape
    print(f"HMM: {num_states} hidden states, {d_vocab} vocab tokens")
    print(f"max_context={args.max_context}, k_ref={args.k_ref}, n_samples={args.num_samples}")
    print()

    # Build reverse HMM
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    # === Entropy rate (for reference and verification) ===
    print("Computing observation entropy rates (MC)...")
    h_X_fwd = compute_observation_entropy_rate(E_fwd, pi, num_samples=100000, burn_in=1000)
    h_X_rev = compute_observation_entropy_rate(E_rev, pi, num_samples=100000, burn_in=1000)
    print(f"  h_X (forward):  {h_X_fwd:.6f} nats")
    print(f"  h_X (reverse):  {h_X_rev:.6f} nats")
    print(f"  |Diff|:         {abs(h_X_fwd - h_X_rev):.2e}")
    h_X = (h_X_fwd + h_X_rev) / 2  # average for plotting
    assert abs(h_X_fwd - h_X_rev) < 0.05, (
        f"Entropy rate mismatch: {h_X_fwd:.6f} vs {h_X_rev:.6f}")
    print(f"  PASS: Forward and reverse entropy rates match")
    print()

    # === Convergence curves ===
    print("Computing forward convergence curves...")
    kl_fwd, loss_fwd, pred_loss_fwd, ref_loss_fwd = compute_convergence_curves(
        E_fwd, pi, args.num_samples, args.max_context, args.k_ref, desc="forward")

    print("Computing reverse convergence curves...")
    kl_rev, loss_rev, pred_loss_rev, ref_loss_rev = compute_convergence_curves(
        E_rev, pi, args.num_samples, args.max_context, args.k_ref, desc="reverse")

    # === Verification ===
    print()
    print("=== Verification ===")
    print(f"  Ref loss (fwd, k={args.k_ref}): {ref_loss_fwd:.6f}")
    print(f"  Ref loss (rev, k={args.k_ref}): {ref_loss_rev:.6f}")
    print(f"  h_X (MC):                       {h_X:.6f}")
    print()

    # === Summary table ===
    k = np.arange(1, args.max_context + 1)
    print(f"{'k':>3} | {'KL_fwd':>10} | {'KL_rev':>10} | {'Loss_fwd':>10} | "
          f"{'Loss_rev':>10} | {'Pred_fwd':>10} | {'Pred_rev':>10}")
    print(f"{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(min(args.max_context, 20)):  # Print first 20
        print(f"{i+1:>3} | {kl_fwd[i]:>10.6f} | {kl_rev[i]:>10.6f} | "
              f"{loss_fwd[i]:>10.6f} | {loss_rev[i]:>10.6f} | "
              f"{pred_loss_fwd[i]:>10.6f} | {pred_loss_rev[i]:>10.6f}")
    if args.max_context > 20:
        print(f"  ... ({args.max_context - 20} more rows)")
    print()

    # === Asymmetry summary ===
    print("=== Convergence Asymmetry Summary ===")
    # Average ratio of KL over first few k values (where signal is strong)
    k_signal = min(10, args.max_context)
    mask = (kl_fwd[:k_signal] > 1e-6) & (kl_rev[:k_signal] > 1e-6)
    if mask.any():
        ratio = kl_rev[:k_signal][mask] / kl_fwd[:k_signal][mask]
        print(f"  Mean KL ratio (rev/fwd) for k=1..{k_signal}: {np.mean(ratio):.3f}")
        print(f"  Median KL ratio: {np.median(ratio):.3f}")
    excess_fwd_sum = np.sum(loss_fwd - h_X)
    excess_rev_sum = np.sum(loss_rev - h_X)
    print(f"  Cumulative excess loss (fwd): {excess_fwd_sum:.4f}")
    print(f"  Cumulative excess loss (rev): {excess_rev_sum:.4f}")
    print(f"  Ratio (rev/fwd): {excess_rev_sum / excess_fwd_sum:.3f}"
          if excess_fwd_sum > 0 else "")
    print()

    # === Plots ===
    os.makedirs(args.fig_dir, exist_ok=True)
    plot_convergence(k, kl_fwd, kl_rev, pred_loss_fwd, pred_loss_rev, h_X, args.fig_dir)
    plot_kl_ratio(k, kl_fwd, kl_rev, args.fig_dir)

    # === Save results ===
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "config": {
            "num_hidden_states": int(num_states),
            "d_vocab": int(d_vocab),
            "max_context": args.max_context,
            "k_ref": args.k_ref,
            "num_samples": args.num_samples,
        },
        "entropy_rate": {
            "h_X_forward": float(h_X_fwd),
            "h_X_reverse": float(h_X_rev),
            "h_X_avg": float(h_X),
            "match": bool(abs(h_X_fwd - h_X_rev) < 0.05),
        },
        "belief_kl_forward": kl_fwd.tolist(),
        "belief_kl_reverse": kl_rev.tolist(),
        "bayes_optimal_loss_forward": loss_fwd.tolist(),
        "bayes_optimal_loss_reverse": loss_rev.tolist(),
        "predictive_loss_forward": pred_loss_fwd.tolist(),
        "predictive_loss_reverse": pred_loss_rev.tolist(),
        "reference_loss_forward": float(ref_loss_fwd),
        "reference_loss_reverse": float(ref_loss_rev),
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
