"""
Theoretical analysis of forward vs. time-reversed CylinderGraph HMM.

Constructs the reverse HMM emission matrices, verifies entropy rate invariance,
and computes Bayes-optimal loss at each context length k for both directions.

The reverse HMM is defined by:
    E_rev[j, k, i] = O(j|k) * T(i, k) * pi(i) / pi(k)

where O(j|k) is the observation probability, T(i,k) is the forward transition
probability from state i to state k, and pi is the stationary distribution.

Key theoretical fact: The entropy rate is identical for forward and backward
processes because joint entropy H(X_1, ..., X_n) is permutation-invariant.

Usage:
    python src/reverse_hmm_analysis.py
    python src/reverse_hmm_analysis.py --num_samples 50000 --max_context 16
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import create_process_from_dict
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Forward vs reverse HMM theoretical analysis")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of Monte Carlo sequences for Bayes-optimal loss")
    parser.add_argument("--max_context", type=int, default=16,
                        help="Maximum context length k to evaluate")
    parser.add_argument("--output", type=str, default="out/reverse_hmm_analysis.json",
                        help="Output JSON path")
    return parser.parse_args()


def build_reverse_emission_matrices(E_fwd, pi):
    """
    Build reverse-time HMM emission matrices.

    Forward HMM: E[j, i, k] = P(observe j AND transition to state k | state i)
    This encodes: E[j, i, k] = O(j|i) * T(i->k | observed j)
    More precisely: E[j, i, k] = P(x_t = j, s_{t+1} = k | s_t = i)

    For the reversed process (reading observations in reverse order):
    E_rev[j, k, i] = P(x_t = j, s_{t-1} = i | s_t = k)  [in the reversed chain]

    Using detailed-balance-like relation:
    P(s_t = k, x_t = j, s_{t+1}=i) [reversed] = P(s_{t-1} = i, x_t = j, s_t = k) [forward]

    So: E_rev[j, k, i] = P(x_t = j, s_{t-1} = i | s_t = k)
                        = P(s_{t-1}=i) * P(x_t=j, s_t=k | s_{t-1}=i) / P(s_t=k)
                        = pi[i] * E_fwd[j, i, k] / pi[k]

    Args:
        E_fwd: Forward emission matrices, shape (d_vocab, num_states, num_states)
               E_fwd[j, i, k] = P(observe j, transition to k | in state i)
        pi: Stationary distribution, shape (num_states,)

    Returns:
        E_rev: Reverse emission matrices, shape (d_vocab, num_states, num_states)
               E_rev[j, k, i] = P(observe j, "transition" to i | in state k) [reversed time]
    """
    d_vocab, num_states, _ = E_fwd.shape
    E_rev = np.zeros_like(E_fwd)

    for j in range(d_vocab):
        for k in range(num_states):
            if pi[k] > 1e-15:
                for i in range(num_states):
                    E_rev[j, k, i] = pi[i] * E_fwd[j, i, k] / pi[k]

    return E_rev


def verify_reverse_hmm(E_fwd, E_rev, pi):
    """Run sanity checks on the reverse HMM construction."""
    num_states = E_fwd.shape[1]
    checks = {}

    # 1. Reverse emission matrices should sum to 1 for each source state
    row_sums = E_rev.sum(axis=(0, 2))  # sum over vocab and target state
    checks["rev_row_sums_close_to_1"] = bool(np.allclose(row_sums, 1.0, atol=1e-10))
    checks["rev_row_sums"] = row_sums.tolist()

    # 2. Reverse transition matrix rows should sum to 1
    T_rev = E_rev.sum(axis=0)  # sum over vocab -> T_rev[k, i]
    rev_row_sums = T_rev.sum(axis=1)
    checks["rev_transition_rows_sum_to_1"] = bool(np.allclose(rev_row_sums, 1.0, atol=1e-10))

    # 3. Reverse stationary distribution should match forward
    eigenvalues, eigenvectors = np.linalg.eig(T_rev.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_rev = np.real(eigenvectors[:, idx])
    pi_rev = pi_rev / pi_rev.sum()
    # Handle sign ambiguity
    if pi_rev[0] < 0:
        pi_rev = -pi_rev
    checks["stationary_dist_match"] = bool(np.allclose(pi, pi_rev, atol=1e-8))
    checks["pi_forward"] = pi.tolist()
    checks["pi_reverse"] = pi_rev.tolist()

    return checks


def compute_entropy_rate(E, pi):
    """
    Compute entropy rate of an HMM given emission matrices and stationary distribution.

    H = -sum_i pi(i) sum_{j,k} E[j,i,k] * log(E[j,i,k])
    """
    h = 0.0
    d_vocab, num_states, _ = E.shape
    for i in range(num_states):
        for j in range(d_vocab):
            for k in range(num_states):
                if E[j, i, k] > 1e-15:
                    h -= pi[i] * E[j, i, k] * np.log(E[j, i, k])
    return h


def forward_algorithm_predictive(E, pi, observations):
    """
    Run the forward algorithm and compute predictive log-loss at each position.

    For position t, given observations o_0, ..., o_{t-1}, compute:
    P(o_t | o_0, ..., o_{t-1}) = sum_k belief_t(k) * sum_j,k' E[o_t, k, k']
                                = sum_k belief_t(k) * O(o_t | k)

    where O(o_t | k) = sum_{k'} E[o_t, k, k'] is the observation probability.

    Returns:
        losses: array of -log P(o_t | o_0, ..., o_{t-1}) for t = 0, ..., len-1
    """
    length = len(observations)
    d_vocab, num_states, _ = E.shape

    # Observation probabilities: O[j, i] = P(observe j | state i) = sum_k E[j, i, k]
    O = E.sum(axis=2)  # (d_vocab, num_states)

    belief = pi.copy()  # P(s_0) = pi
    losses = np.zeros(length)

    for t in range(length):
        obs_t = observations[t]

        # Predictive probability: P(o_t | past) = sum_i belief(i) * O(o_t | i)
        pred_prob = belief @ O[obs_t]
        losses[t] = -np.log(pred_prob + 1e-30)

        # Update belief: incorporate observation and transition
        # P(s_{t+1} = k | o_0..o_t) ∝ sum_i belief(i) * E[o_t, i, k]
        belief = belief @ E[obs_t]  # (num_states,) @ (num_states, num_states) -> (num_states,)
        belief_sum = belief.sum()
        if belief_sum > 0:
            belief = belief / belief_sum

    return losses


def compute_bayes_optimal_loss_by_context(E, pi, num_samples, max_context, desc=""):
    """
    Compute Bayes-optimal loss at each context length k via Monte Carlo.

    Generates sequences from the HMM, runs the forward algorithm, and
    averages the loss at each position t (which uses context length t).

    Args:
        E: Emission matrices (d_vocab, num_states, num_states)
        pi: Stationary distribution
        num_samples: Number of sequences to generate
        max_context: Maximum sequence length

    Returns:
        losses_by_k: array of shape (max_context,), where losses_by_k[k] is the
                     average Bayes-optimal loss using context of length k
    """
    from hmm import HMM

    # Build a lightweight HMM wrapper for sequence generation
    class _TempHMM(HMM):
        def __init__(self, emission_matrices, stationary):
            self._E = emission_matrices
            self._pi = stationary

        @property
        def emission_matrices(self):
            return self._E

        def get_stationary_distribution(self):
            return self._pi

    hmm = _TempHMM(E, pi)

    loss_accum = np.zeros(max_context)
    count_accum = np.zeros(max_context)

    for _ in tqdm(range(num_samples), desc=f"Bayes-optimal loss {desc}"):
        _, obs = hmm.generate_sequence(max_context)
        losses = forward_algorithm_predictive(E, pi, obs)
        loss_accum += losses
        count_accum += 1.0

    return loss_accum / count_accum


def compute_belief_convergence(E, pi, num_samples, max_context, desc=""):
    """
    Compute KL divergence from partial-context belief to fully-converged belief.

    For each sequence, compute belief state at each position t using context
    0..t, and compare to the belief at the final position (using full context).

    Returns:
        kl_by_k: array of shape (max_context,), where kl_by_k[k] is the
                 average KL(belief_full || belief_k)
    """
    from hmm import HMM

    class _TempHMM(HMM):
        def __init__(self, emission_matrices, stationary):
            self._E = emission_matrices
            self._pi = stationary

        @property
        def emission_matrices(self):
            return self._E

        def get_stationary_distribution(self):
            return self._pi

    hmm = _TempHMM(E, pi)

    kl_accum = np.zeros(max_context)

    for _ in tqdm(range(num_samples), desc=f"Belief convergence {desc}"):
        _, obs = hmm.generate_sequence(max_context)

        # Compute beliefs at each position
        beliefs = []
        belief = pi.copy()
        for t in range(max_context):
            belief = belief @ E[obs[t]]
            belief = belief / (belief.sum() + 1e-30)
            beliefs.append(belief.copy())

        # Use final belief as "converged" reference
        belief_final = beliefs[-1]

        # KL(final || partial) at each position
        for t in range(max_context):
            # KL(P || Q) = sum P * log(P/Q), P=final, Q=partial
            kl = 0.0
            for s in range(len(pi)):
                if belief_final[s] > 1e-15:
                    kl += belief_final[s] * np.log(belief_final[s] / (beliefs[t][s] + 1e-30))
            kl_accum[t] += max(kl, 0.0)  # clamp numerical noise

    return kl_accum / num_samples


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
    print(f"Stationary distribution: {pi}")
    print()

    # Build reverse HMM
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    # Verify construction
    print("=== Verification Checks ===")
    checks = verify_reverse_hmm(E_fwd, E_rev, pi)
    for k, v in checks.items():
        if isinstance(v, bool):
            status = "PASS" if v else "FAIL"
            print(f"  {k}: {status}")
        elif isinstance(v, list) and len(v) <= 20:
            print(f"  {k}: {[f'{x:.6f}' for x in v]}")
    print()

    # Entropy rates
    h_fwd = compute_entropy_rate(E_fwd, pi)
    h_rev = compute_entropy_rate(E_rev, pi)
    print(f"=== Entropy Rates ===")
    print(f"  Forward:  {h_fwd:.8f} nats")
    print(f"  Reverse:  {h_rev:.8f} nats")
    print(f"  |Diff|:   {abs(h_fwd - h_rev):.2e}")
    entropy_match = abs(h_fwd - h_rev) < 1e-6
    print(f"  Match:    {'PASS' if entropy_match else 'FAIL'}")
    print()

    # Bayes-optimal loss by context length
    print(f"=== Bayes-Optimal Loss by Context (n={args.num_samples} samples) ===")
    loss_fwd = compute_bayes_optimal_loss_by_context(
        E_fwd, pi, args.num_samples, args.max_context, desc="forward")
    loss_rev = compute_bayes_optimal_loss_by_context(
        E_rev, pi, args.num_samples, args.max_context, desc="reverse")

    print(f"  {'k':>3} | {'Forward':>10} | {'Reverse':>10} | {'Diff':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for k in range(args.max_context):
        diff = loss_rev[k] - loss_fwd[k]
        print(f"  {k:>3} | {loss_fwd[k]:>10.6f} | {loss_rev[k]:>10.6f} | {diff:>+10.6f}")
    print()

    # Belief state convergence
    print(f"=== Belief State Convergence (KL from final belief) ===")
    kl_fwd = compute_belief_convergence(
        E_fwd, pi, min(args.num_samples, 10000), args.max_context, desc="forward")
    kl_rev = compute_belief_convergence(
        E_rev, pi, min(args.num_samples, 10000), args.max_context, desc="reverse")

    print(f"  {'k':>3} | {'KL_fwd':>10} | {'KL_rev':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}")
    for k in range(args.max_context):
        print(f"  {k:>3} | {kl_fwd[k]:>10.6f} | {kl_rev[k]:>10.6f}")
    print()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "num_hidden_states": int(num_states),
        "d_vocab": int(d_vocab),
        "entropy_rate_forward": float(h_fwd),
        "entropy_rate_reverse": float(h_rev),
        "entropy_rate_match": bool(entropy_match),
        "num_samples": args.num_samples,
        "max_context": args.max_context,
        "bayes_optimal_loss_forward": loss_fwd.tolist(),
        "bayes_optimal_loss_reverse": loss_rev.tolist(),
        "belief_kl_forward": kl_fwd.tolist(),
        "belief_kl_reverse": kl_rev.tolist(),
        "verification": {k: bool(v) for k, v in checks.items() if isinstance(v, (bool, np.bool_))},
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
