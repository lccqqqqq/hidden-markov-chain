"""
Compare Bayes-optimal prediction of a centre token from left vs right context.

Setup:
  Generate sequences of length 2k+1 from the stationary HMM.
  The target is x_{k+1} (the centre token, 1-indexed).

  Left predictor:  Run forward algorithm L->R on (x_1, ..., x_k) with the
                   FORWARD emission matrices, starting from the stationary
                   distribution.  Predict x_{k+1}.

  Right predictor: Run forward algorithm R->L on (x_{k+2}, ..., x_{2k+1}) with
                   the REVERSE emission matrices, starting from the stationary
                   distribution.  Predict x_{k+1}.

Both are Bayes-optimal given their respective k-token context windows.
Any difference in loss reflects the intrinsic asymmetry between forward
and reverse prediction at finite context.

Usage:
    python src/centre_token_prediction.py
    python src/centre_token_prediction.py --max_k 8 --num_samples 50000
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centre-token prediction: left context (fwd) vs right context (rev)")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--max_k", type=int, default=8,
                        help="Maximum context length k (each side)")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of Monte Carlo sequences")
    parser.add_argument("--output", type=str,
                        default="out/centre_token_prediction.json")
    parser.add_argument("--fig_dir", type=str,
                        default="figures/centre_token_prediction")
    return parser.parse_args()


def predict_centre_token(E_fwd, E_rev, pi, sequence, k):
    """
    Predict the centre token x_{k} (0-indexed) of a sequence of length 2k+1,
    using k tokens of left context (forward) and k tokens of right context
    (reverse).

    Args:
        E_fwd: Forward emission matrices (d_vocab, n_states, n_states)
        E_rev: Reverse emission matrices (d_vocab, n_states, n_states)
        pi: Stationary distribution (n_states,)
        sequence: Observation sequence of length 2k+1
        k: Context length (number of tokens on each side)

    Returns:
        loss_left:  -log P_fwd(x_k | x_0, ..., x_{k-1})
        loss_right: -log P_rev(x_k | x_{k+1}, ..., x_{2k})
        pred_left:  Full predictive distribution from left context (d_vocab,)
        pred_right: Full predictive distribution from right context (d_vocab,)
    """
    d_vocab = E_fwd.shape[0]
    target = sequence[k]  # centre token (0-indexed)

    # Observation matrices: O[j, i] = P(observe j | state i)
    O_fwd = E_fwd.sum(axis=2)  # (d_vocab, n_states)
    O_rev = E_rev.sum(axis=2)  # (d_vocab, n_states)

    # --- Left predictor: forward algorithm on x_0, ..., x_{k-1} ---
    belief_left = pi.copy()
    for t in range(k):
        obs = sequence[t]
        # Update: belief -> belief @ E_fwd[obs] then normalise
        belief_left = belief_left @ E_fwd[obs]
        s = belief_left.sum()
        if s > 0:
            belief_left /= s

    # Predictive distribution from left context
    pred_left = O_fwd.T @ np.zeros(d_vocab)  # placeholder
    # P(x_k = j | left context) = sum_i belief_left(i) * O_fwd(j, i)
    pred_left = belief_left @ O_fwd.T  # (n_states,) @ (n_states, d_vocab) -> (d_vocab,)

    # --- Right predictor: reverse forward algorithm on x_{2k}, ..., x_{k+1} ---
    # Read right context in reverse order
    belief_right = pi.copy()
    for t in range(k):
        obs = sequence[2 * k - t]  # x_{2k}, x_{2k-1}, ..., x_{k+1}
        belief_right = belief_right @ E_rev[obs]
        s = belief_right.sum()
        if s > 0:
            belief_right /= s

    # Predictive distribution from right context
    # The reverse HMM's "next token" prediction is in the reverse time direction,
    # which corresponds to predicting x_k given x_{k+1}, ..., x_{2k}.
    # P(x_k = j | right context) = sum_i belief_right(i) * O_rev(j, i)
    pred_right = belief_right @ O_rev.T  # (d_vocab,)

    # Losses
    loss_left = -np.log(pred_left[target] + 1e-30)
    loss_right = -np.log(pred_right[target] + 1e-30)

    return loss_left, loss_right, pred_left, pred_right


def generate_sequence_from_E(E, pi, length):
    """Generate a sequence of given length from emission matrices E."""
    n_states = len(pi)
    d_vocab = E.shape[0]

    state = np.random.choice(n_states, p=pi)
    obs = np.zeros(length, dtype=np.int64)

    for t in range(length):
        # Joint distribution over (observation, next_state)
        probs = E[:, state, :].flatten()  # (d_vocab * n_states,)
        sample = np.random.choice(len(probs), p=probs)
        obs[t] = sample // n_states
        state = sample % n_states

    return obs


def run_experiment(E_fwd, E_rev, pi, max_k, num_samples):
    """
    Run the centre-token prediction experiment for k = 1, ..., max_k.

    Returns:
        results: dict with arrays indexed by k
    """
    results = {
        "k_values": list(range(1, max_k + 1)),
        "mean_loss_left": [],
        "mean_loss_right": [],
        "std_loss_left": [],
        "std_loss_right": [],
        "mean_entropy_left": [],
        "mean_entropy_right": [],
    }

    for k in range(1, max_k + 1):
        seq_length = 2 * k + 1
        losses_left = np.zeros(num_samples)
        losses_right = np.zeros(num_samples)
        entropies_left = np.zeros(num_samples)
        entropies_right = np.zeros(num_samples)

        for i in tqdm(range(num_samples), desc=f"k={k}"):
            seq = generate_sequence_from_E(E_fwd, pi, seq_length)
            ll, lr, pl, pr = predict_centre_token(
                E_fwd, E_rev, pi, seq, k)

            losses_left[i] = ll
            losses_right[i] = lr
            entropies_left[i] = -np.sum(pl * np.log(pl + 1e-30))
            entropies_right[i] = -np.sum(pr * np.log(pr + 1e-30))

        results["mean_loss_left"].append(float(losses_left.mean()))
        results["mean_loss_right"].append(float(losses_right.mean()))
        results["std_loss_left"].append(float(losses_left.std()))
        results["std_loss_right"].append(float(losses_right.std()))
        results["mean_entropy_left"].append(float(entropies_left.mean()))
        results["mean_entropy_right"].append(float(entropies_right.mean()))

        print(f"  k={k}: L_left={losses_left.mean():.6f} +/- {losses_left.std():.4f}, "
              f"L_right={losses_right.mean():.6f} +/- {losses_right.std():.4f}, "
              f"delta={losses_right.mean() - losses_left.mean():+.6f}")

    return results


def plot_results(results, h_X, fig_dir):
    """Generate summary plots."""
    os.makedirs(fig_dir, exist_ok=True)

    k_vals = np.array(results["k_values"])
    ml = np.array(results["mean_loss_left"])
    mr = np.array(results["mean_loss_right"])
    sl = np.array(results["std_loss_left"])
    sr = np.array(results["std_loss_right"])
    el = np.array(results["mean_entropy_left"])
    er = np.array(results["mean_entropy_right"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Mean loss
    ax = axes[0]
    ax.errorbar(k_vals - 0.05, ml, yerr=sl / np.sqrt(len(k_vals)),
                fmt='o-', color="#1f77b4", label="Left (forward)", capsize=3)
    ax.errorbar(k_vals + 0.05, mr, yerr=sr / np.sqrt(len(k_vals)),
                fmt='s-', color="#d62728", label="Right (reverse)", capsize=3)
    if h_X is not None:
        ax.axhline(h_X, color="gray", ls="--", lw=0.8, label=f"$h_X = {h_X:.3f}$")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel("Mean loss (nats)")
    ax.set_title("(a) Bayes-optimal loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (b): Loss difference
    ax = axes[1]
    delta = mr - ml
    ax.bar(k_vals, delta, color="#2ca02c", alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel("$\\Delta L$ = right $-$ left (nats)")
    ax.set_title("(b) Loss gap (right $-$ left)")
    ax.grid(alpha=0.3)

    # Panel (c): Predictive entropy
    ax = axes[2]
    ax.plot(k_vals, el, 'o-', color="#1f77b4", label="Left (forward)")
    ax.plot(k_vals, er, 's-', color="#d62728", label="Right (reverse)")
    ax.set_xlabel("Context length $k$")
    ax.set_ylabel("Mean predictive entropy (nats)")
    ax.set_title("(c) Predictive entropy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "centre_token_prediction.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()

    # Load HMM
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    E_fwd = proc.emission_matrices
    pi = proc.get_stationary_distribution()
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    # Estimate entropy rate
    h_X = proc.entropy_rate_empirical_estimate(100000, burn_in=1000)
    print(f"Entropy rate h_X = {h_X:.6f} nats")

    # Run experiment
    print(f"\nRunning centre-token prediction: k = 1..{args.max_k}, "
          f"N = {args.num_samples}")
    results = run_experiment(E_fwd, E_rev, pi, args.max_k, args.num_samples)
    results["h_X"] = float(h_X)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Plot
    plot_results(results, h_X, args.fig_dir)


if __name__ == "__main__":
    main()
