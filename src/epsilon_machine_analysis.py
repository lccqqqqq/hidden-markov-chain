"""
Epsilon-machine analysis of the CylinderGraph HMM (forward vs reverse).

Computes:
1. Forward and reverse emission matrices
2. Forward and reverse transition matrices and observation matrices
3. Number of effective causal states (via transition matrix structure)
4. Crypticity chi_fwd, chi_rev (excess entropy beyond entropy rate)
5. Causal irreversibility Xi = C_rev - C_fwd

The epsilon-machine is the minimal unifilar HMM for a process. For a
general HMM (like CylinderGraph), the epsilon-machine may have fewer
states than the HMM (state merging) or more (state splitting). This
analysis characterizes the structural asymmetry between forward and
reverse prediction.

Usage:
    python src/epsilon_machine_analysis.py
    python src/epsilon_machine_analysis.py --config config/base_config.yaml
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import create_process_from_dict
from reverse_hmm_analysis import build_reverse_emission_matrices
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Epsilon-machine analysis of CylinderGraph HMM")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--output", type=str, default="out/epsilon_machine_analysis.json",
                        help="Output JSON path")
    parser.add_argument("--num_samples", type=int, default=100000,
                        help="Number of Monte Carlo samples for crypticity estimation")
    parser.add_argument("--convergence_length", type=int, default=200,
                        help="Sequence length for belief convergence (proxy for k=inf)")
    return parser.parse_args()


def compute_transition_and_observation_matrices(E):
    """
    Decompose emission matrices E[j,i,k] into transition matrix T and
    observation matrix O.

    T[i,k] = sum_j E[j,i,k] = P(s_{t+1}=k | s_t=i)
    O[j,i] = sum_k E[j,i,k] = P(x_t=j | s_t=i)

    Returns:
        T: (num_states, num_states) transition matrix
        O: (d_vocab, num_states) observation probability matrix
    """
    T = E.sum(axis=0)  # (num_states, num_states)
    O = E.sum(axis=2)  # (d_vocab, num_states)
    return T, O


def compute_statistical_complexity(T, O, pi):
    """
    Compute the statistical complexity C_mu = -sum_i pi(i) log pi(i).

    This is the Shannon entropy of the stationary distribution over
    causal states. For the HMM states (which may not be minimal causal
    states), this gives an upper bound on the true statistical complexity.

    Args:
        T: Transition matrix (num_states, num_states)
        O: Observation matrix (d_vocab, num_states) [unused here but kept for API]
        pi: Stationary distribution (num_states,)

    Returns:
        C_mu: Statistical complexity in nats
    """
    C_mu = 0.0
    for i in range(len(pi)):
        if pi[i] > 1e-15:
            C_mu -= pi[i] * np.log(pi[i])
    return C_mu


def compute_entropy_rate(E, pi):
    """
    Compute entropy rate h_mu = -sum_i pi(i) sum_{j,k} E[j,i,k] log E[j,i,k].

    This is the entropy rate of the HMM (joint over observations and transitions).
    """
    h = 0.0
    d_vocab, num_states, _ = E.shape
    for i in range(num_states):
        for j in range(d_vocab):
            for k in range(num_states):
                if E[j, i, k] > 1e-15:
                    h -= pi[i] * E[j, i, k] * np.log(E[j, i, k])
    return h


def compute_observation_entropy_rate(E, pi, num_samples=100000, burn_in=1000):
    """
    Compute the observation entropy rate h_X via Monte Carlo.

    h_X = lim_{n->inf} H(X_n | X_1, ..., X_{n-1})

    This is the entropy rate of the observed process only (marginalizing
    out hidden states). It equals the Bayes-optimal prediction loss at
    infinite context.

    Uses the forward algorithm to compute exact predictive distributions.
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
    return hmm.entropy_rate_empirical_estimate(num_samples, burn_in=burn_in)


def compute_crypticity(E, pi, num_samples=100000, convergence_length=200):
    """
    Estimate crypticity chi = C_mu - E[I(S; past)] where past = observed history.

    Crypticity measures how much of the internal state information is
    "hidden" from the observer. Higher crypticity means the observer
    needs more context to infer the causal state.

    We estimate this via Monte Carlo:
    chi = C_mu - <H(S_0) - H(S_t | x_0, ..., x_{t-1})>
    where the average is over long sequences and H(S_t | past) is the
    entropy of the belief state after observing t tokens.

    More precisely:
    chi = C_mu - I_infty, where I_infty = lim_{t->inf} I(S_t; X_0, ..., X_{t-1})
        = C_mu - [H(S) - H(S | X_0, ..., X_{t-1})]  at convergence
        = H(S | X_0, ..., X_{inf})  [residual uncertainty about state given infinite past]

    Args:
        E: Emission matrices
        pi: Stationary distribution
        num_samples: Number of MC sequences
        convergence_length: Length of each sequence (proxy for infinite past)

    Returns:
        chi: Crypticity estimate in nats
        conditional_entropy: H(S | infinite past) - the residual state uncertainty
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
    num_states = len(pi)

    # H(S) = entropy of stationary distribution = C_mu
    C_mu = compute_statistical_complexity(None, None, pi)

    # Estimate H(S | past_inf) via Monte Carlo
    # Generate long sequences and compute belief entropy at the end
    belief_entropies = []

    for _ in range(num_samples):
        _, obs = hmm.generate_sequence(convergence_length)

        # Run forward algorithm to get final belief state
        belief = pi.copy()
        for t in range(convergence_length):
            belief = belief @ E[obs[t]]
            belief_sum = belief.sum()
            if belief_sum > 0:
                belief = belief / belief_sum

        # Entropy of the final belief state
        h_belief = 0.0
        for s in range(num_states):
            if belief[s] > 1e-15:
                h_belief -= belief[s] * np.log(belief[s])
        belief_entropies.append(h_belief)

    conditional_entropy = np.mean(belief_entropies)
    chi = conditional_entropy  # chi = H(S | past_inf) = C_mu - I_inf

    return chi, conditional_entropy


def count_effective_states(T, pi, threshold=1e-10):
    """
    Count the number of states with non-negligible stationary probability.

    Also identifies absorbing/transient structure.

    Returns:
        n_effective: Number of states with pi > threshold
        n_total: Total states
    """
    n_effective = int(np.sum(pi > threshold))
    return n_effective, len(pi)


def analyze_transition_structure(T, pi):
    """
    Analyze the transition matrix structure.

    Returns:
        info dict with eigenvalue spectrum, mixing time estimate, etc.
    """
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]

    # Spectral gap determines mixing time
    if len(eigenvalues) > 1:
        spectral_gap = 1.0 - eigenvalues[1]
        mixing_time = 1.0 / spectral_gap if spectral_gap > 1e-10 else float('inf')
    else:
        spectral_gap = 1.0
        mixing_time = 0.0

    return {
        "eigenvalue_magnitudes": eigenvalues.tolist(),
        "spectral_gap": float(spectral_gap),
        "mixing_time_estimate": float(mixing_time),
    }


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
    print(f"Stationary distribution entropy: {-np.sum(pi[pi>0] * np.log(pi[pi>0])):.4f} nats")
    print()

    # Build reverse HMM
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    # Decompose into transition and observation matrices
    T_fwd, O_fwd = compute_transition_and_observation_matrices(E_fwd)
    T_rev, O_rev = compute_transition_and_observation_matrices(E_rev)

    # === 1. State counts ===
    n_eff_fwd, n_tot_fwd = count_effective_states(T_fwd, pi)
    n_eff_rev, n_tot_rev = count_effective_states(T_rev, pi)
    print("=== State Counts ===")
    print(f"  Forward:  {n_eff_fwd}/{n_tot_fwd} effective states")
    print(f"  Reverse:  {n_eff_rev}/{n_tot_rev} effective states")
    print()

    # === 2. Statistical complexity (upper bound via HMM states) ===
    C_mu_fwd = compute_statistical_complexity(T_fwd, O_fwd, pi)
    C_mu_rev = compute_statistical_complexity(T_rev, O_rev, pi)
    print("=== Statistical Complexity (HMM upper bound) ===")
    print(f"  C_mu (forward):  {C_mu_fwd:.6f} nats")
    print(f"  C_mu (reverse):  {C_mu_rev:.6f} nats")
    print(f"  Note: Same because pi is shared. True C_mu may differ")
    print(f"  if epsilon-machines have different numbers of causal states.")
    print()

    # === 3. Entropy rates ===
    h_joint_fwd = compute_entropy_rate(E_fwd, pi)
    h_joint_rev = compute_entropy_rate(E_rev, pi)
    print("=== Joint Entropy Rates H(X,S'|S) ===")
    print(f"  Forward:  {h_joint_fwd:.8f} nats")
    print(f"  Reverse:  {h_joint_rev:.8f} nats")
    print(f"  |Diff|:   {abs(h_joint_fwd - h_joint_rev):.2e}")
    print()

    # Observation entropy rate (what the transformer actually tries to learn)
    print("=== Observation Entropy Rate h_X (Monte Carlo) ===")
    h_X_fwd = compute_observation_entropy_rate(E_fwd, pi, num_samples=args.num_samples)
    h_X_rev = compute_observation_entropy_rate(E_rev, pi, num_samples=args.num_samples)
    print(f"  Forward:  {h_X_fwd:.6f} nats")
    print(f"  Reverse:  {h_X_rev:.6f} nats")
    print(f"  |Diff|:   {abs(h_X_fwd - h_X_rev):.2e}")
    print()

    # === 4. Transition matrix structure ===
    print("=== Transition Matrix Analysis ===")
    struct_fwd = analyze_transition_structure(T_fwd, pi)
    struct_rev = analyze_transition_structure(T_rev, pi)
    print(f"  Forward spectral gap:  {struct_fwd['spectral_gap']:.6f}")
    print(f"  Reverse spectral gap:  {struct_rev['spectral_gap']:.6f}")
    print(f"  Forward mixing time:   {struct_fwd['mixing_time_estimate']:.2f}")
    print(f"  Reverse mixing time:   {struct_rev['mixing_time_estimate']:.2f}")
    print(f"  Forward top eigenvalues: {struct_fwd['eigenvalue_magnitudes'][:5]}")
    print(f"  Reverse top eigenvalues: {struct_rev['eigenvalue_magnitudes'][:5]}")
    print()

    # === 5. Crypticity ===
    print(f"=== Crypticity (MC estimate, n={args.num_samples}, L={args.convergence_length}) ===")
    chi_fwd, h_cond_fwd = compute_crypticity(
        E_fwd, pi, num_samples=args.num_samples,
        convergence_length=args.convergence_length)
    chi_rev, h_cond_rev = compute_crypticity(
        E_rev, pi, num_samples=args.num_samples,
        convergence_length=args.convergence_length)
    print(f"  chi_fwd (crypticity):   {chi_fwd:.6f} nats")
    print(f"  chi_rev (crypticity):   {chi_rev:.6f} nats")
    print(f"  H(S|past_inf) fwd:      {h_cond_fwd:.6f} nats")
    print(f"  H(S|past_inf) rev:      {h_cond_rev:.6f} nats")
    print()

    # === 6. Causal irreversibility ===
    # Xi = C_rev - C_fwd (for true epsilon-machines)
    # Here we can compute an analogous quantity from crypticity:
    # The direction with higher crypticity has slower belief convergence
    Xi_crypticity = chi_rev - chi_fwd
    print("=== Causal Irreversibility ===")
    print(f"  Xi (chi_rev - chi_fwd): {Xi_crypticity:+.6f} nats")
    if abs(Xi_crypticity) > 0.01:
        slower = "reverse" if Xi_crypticity > 0 else "forward"
        print(f"  --> {slower} direction has higher crypticity")
        print(f"  --> Expect {slower} prediction to converge slower")
    else:
        print(f"  --> Near-zero: forward and reverse are similarly complex")
    print()

    # === 7. Observation matrix analysis ===
    # Check if observation distributions distinguish states differently
    # in forward vs reverse
    print("=== Observation Matrix Analysis ===")
    # Average pairwise KL between observation distributions of different states
    def avg_pairwise_kl(O):
        n = O.shape[1]
        kl_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                kl = 0.0
                for v in range(O.shape[0]):
                    if O[v, i] > 1e-15 and O[v, j] > 1e-15:
                        kl += O[v, i] * np.log(O[v, i] / O[v, j])
                        kl += O[v, j] * np.log(O[v, j] / O[v, i])
                kl_sum += kl / 2  # symmetrized KL
                count += 1
        return kl_sum / count if count > 0 else 0.0

    avg_kl_fwd = avg_pairwise_kl(O_fwd)
    avg_kl_rev = avg_pairwise_kl(O_rev)
    print(f"  Avg pairwise sym-KL (forward obs):  {avg_kl_fwd:.6f}")
    print(f"  Avg pairwise sym-KL (reverse obs):  {avg_kl_rev:.6f}")
    print(f"  Ratio (rev/fwd): {avg_kl_rev / avg_kl_fwd:.3f}" if avg_kl_fwd > 0 else "")
    print()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "config": {
            "num_hidden_states": int(num_states),
            "d_vocab": int(d_vocab),
            "process_params": cfg["data_generator"]["process"]["params"],
        },
        "state_counts": {
            "forward_effective": n_eff_fwd,
            "forward_total": n_tot_fwd,
            "reverse_effective": n_eff_rev,
            "reverse_total": n_tot_rev,
        },
        "statistical_complexity": {
            "C_mu_forward": float(C_mu_fwd),
            "C_mu_reverse": float(C_mu_rev),
            "note": "Upper bound from HMM states; true C_mu requires epsilon-machine minimization",
        },
        "entropy_rates": {
            "h_joint_forward": float(h_joint_fwd),
            "h_joint_reverse": float(h_joint_rev),
            "h_observation_forward": float(h_X_fwd),
            "h_observation_reverse": float(h_X_rev),
            "joint_diff": float(abs(h_joint_fwd - h_joint_rev)),
            "observation_diff": float(abs(h_X_fwd - h_X_rev)),
        },
        "transition_structure": {
            "forward": struct_fwd,
            "reverse": struct_rev,
        },
        "crypticity": {
            "chi_forward": float(chi_fwd),
            "chi_reverse": float(chi_rev),
            "H_S_given_past_forward": float(h_cond_fwd),
            "H_S_given_past_reverse": float(h_cond_rev),
            "num_samples": args.num_samples,
            "convergence_length": args.convergence_length,
        },
        "causal_irreversibility": {
            "Xi_crypticity": float(Xi_crypticity),
            "interpretation": (
                f"{'Reverse' if Xi_crypticity > 0 else 'Forward'} has higher crypticity"
                if abs(Xi_crypticity) > 0.01
                else "Near-symmetric"
            ),
        },
        "observation_distinguishability": {
            "avg_pairwise_sym_kl_forward": float(avg_kl_fwd),
            "avg_pairwise_sym_kl_reverse": float(avg_kl_rev),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
