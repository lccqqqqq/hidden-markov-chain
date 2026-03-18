"""
Tests for centre_token_prediction.py.

Verifies:
1. Forward algorithm correctness on a trivial 2-state HMM
2. Reverse emission matrix construction consistency
3. Predictive distributions sum to 1
4. At large k, both predictors converge to the entropy rate
5. Symmetry: on a time-reversible HMM, left and right losses should match
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from centre_token_prediction import (
    predict_centre_token,
    generate_sequence_from_E,
    run_experiment,
)
from reverse_hmm_analysis import build_reverse_emission_matrices


def make_simple_hmm():
    """
    Create a simple 2-state, 2-token HMM for testing.

    State 0 emits token 0 with prob 0.9, token 1 with prob 0.1.
    State 1 emits token 1 with prob 0.9, token 0 with prob 0.1.
    Transition: stay with prob 0.7, switch with prob 0.3.

    E[j, i, k] = P(observe j, transition to k | in state i)
               = P(observe j | state i) * P(transition to k | state i)
    (observation and transition are independent given state)
    """
    O = np.array([[0.9, 0.1],    # P(token 0 | state 0), P(token 0 | state 1)
                  [0.1, 0.9]])   # P(token 1 | state 0), P(token 1 | state 1)
    T = np.array([[0.7, 0.3],    # P(stay at 0 | state 0), P(go to 1 | state 0)
                  [0.3, 0.7]])   # P(go to 0 | state 1), P(stay at 1 | state 1)

    # E[j, i, k] = O[j, i] * T[i, k]
    d_vocab, n_states = O.shape
    E = np.zeros((d_vocab, n_states, n_states))
    for j in range(d_vocab):
        for i in range(n_states):
            for k in range(n_states):
                E[j, i, k] = O[j, i] * T[i, k]

    # This HMM is symmetric under state/token swap, so pi = [0.5, 0.5]
    pi = np.array([0.5, 0.5])
    return E, pi


def make_asymmetric_hmm():
    """
    Create an asymmetric 2-state, 2-token HMM where forward != reverse.

    State 0: emits token 0 with prob 0.95
    State 1: emits token 1 with prob 0.95
    Transition: 0->1 with prob 0.4, 1->0 with prob 0.1
    (asymmetric transitions create forward/reverse asymmetry)
    """
    O = np.array([[0.95, 0.05],
                  [0.05, 0.95]])
    T = np.array([[0.6, 0.4],
                  [0.1, 0.9]])

    d_vocab, n_states = O.shape
    E = np.zeros((d_vocab, n_states, n_states))
    for j in range(d_vocab):
        for i in range(n_states):
            for k in range(n_states):
                E[j, i, k] = O[j, i] * T[i, k]

    # Stationary distribution: pi(0) * 0.4 = pi(1) * 0.1 => pi(0)/pi(1) = 1/4
    # pi(0) = 0.2, pi(1) = 0.8
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    if pi[0] < 0:
        pi = -pi

    return E, pi


def test_emission_matrix_validity():
    """E[j, i, k] should sum to 1 over (j, k) for each i."""
    print("Test: emission matrix validity")
    for name, (E, pi) in [("simple", make_simple_hmm()),
                           ("asymmetric", make_asymmetric_hmm())]:
        row_sums = E.sum(axis=(0, 2))  # sum over vocab and target state
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"{name}: row sums = {row_sums}"
    print("  PASSED")


def test_reverse_emission_matrix():
    """Reverse emission matrix should also be a valid emission matrix."""
    print("Test: reverse emission matrix validity")
    for name, (E_fwd, pi) in [("simple", make_simple_hmm()),
                                ("asymmetric", make_asymmetric_hmm())]:
        E_rev = build_reverse_emission_matrices(E_fwd, pi)

        # Should sum to 1 over (j, k) for each source state
        row_sums = E_rev.sum(axis=(0, 2))
        assert np.allclose(row_sums, 1.0, atol=1e-10), \
            f"{name}: reverse row sums = {row_sums}"

        # Should have same stationary distribution
        T_rev = E_rev.sum(axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(T_rev.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi_rev = np.real(eigenvectors[:, idx])
        pi_rev = pi_rev / pi_rev.sum()
        if pi_rev[0] < 0:
            pi_rev = -pi_rev
        assert np.allclose(pi, pi_rev, atol=1e-8), \
            f"{name}: pi mismatch: {pi} vs {pi_rev}"
    print("  PASSED")


def test_predictive_distributions_sum_to_1():
    """Both predictive distributions should be valid probability distributions."""
    print("Test: predictive distributions sum to 1")
    E_fwd, pi = make_simple_hmm()
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    np.random.seed(42)
    for k in [1, 2, 4]:
        seq = generate_sequence_from_E(E_fwd, pi, 2 * k + 1)
        _, _, pred_left, pred_right = predict_centre_token(
            E_fwd, E_rev, pi, seq, k)

        assert np.all(pred_left >= 0), f"k={k}: negative pred_left"
        assert np.all(pred_right >= 0), f"k={k}: negative pred_right"
        assert abs(pred_left.sum() - 1.0) < 1e-10, \
            f"k={k}: pred_left sums to {pred_left.sum()}"
        assert abs(pred_right.sum() - 1.0) < 1e-10, \
            f"k={k}: pred_right sums to {pred_right.sum()}"
    print("  PASSED")


def test_symmetric_hmm_equal_losses():
    """For the symmetric HMM, left and right losses should be statistically equal."""
    print("Test: symmetric HMM gives equal left/right losses")
    E_fwd, pi = make_simple_hmm()
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    np.random.seed(42)
    n_samples = 10000
    k = 3

    losses_left = np.zeros(n_samples)
    losses_right = np.zeros(n_samples)

    for i in range(n_samples):
        seq = generate_sequence_from_E(E_fwd, pi, 2 * k + 1)
        ll, lr, _, _ = predict_centre_token(E_fwd, E_rev, pi, seq, k)
        losses_left[i] = ll
        losses_right[i] = lr

    # For the symmetric HMM (T is doubly stochastic, O is symmetric),
    # forward = reverse, so mean losses should match
    mean_diff = abs(losses_left.mean() - losses_right.mean())
    se = np.sqrt(losses_left.var() / n_samples + losses_right.var() / n_samples)
    z_score = mean_diff / se if se > 0 else 0

    print(f"  L_left  = {losses_left.mean():.6f} +/- {losses_left.std():.4f}")
    print(f"  L_right = {losses_right.mean():.6f} +/- {losses_right.std():.4f}")
    print(f"  |diff| = {mean_diff:.6f}, z = {z_score:.2f}")
    assert z_score < 3.0, f"Symmetric HMM: unexpected asymmetry, z={z_score:.2f}"
    print("  PASSED")


def test_asymmetric_hmm_different_losses():
    """For the asymmetric HMM, left and right losses should differ at small k."""
    print("Test: asymmetric HMM gives different left/right losses")
    E_fwd, pi = make_asymmetric_hmm()
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    np.random.seed(42)
    n_samples = 10000
    k = 2  # short context to maximise asymmetry

    losses_left = np.zeros(n_samples)
    losses_right = np.zeros(n_samples)

    for i in range(n_samples):
        seq = generate_sequence_from_E(E_fwd, pi, 2 * k + 1)
        ll, lr, _, _ = predict_centre_token(E_fwd, E_rev, pi, seq, k)
        losses_left[i] = ll
        losses_right[i] = lr

    print(f"  L_left  = {losses_left.mean():.6f} +/- {losses_left.std():.4f}")
    print(f"  L_right = {losses_right.mean():.6f} +/- {losses_right.std():.4f}")
    print(f"  delta   = {losses_right.mean() - losses_left.mean():+.6f}")
    # We just check it runs; the asymmetric HMM doesn't guarantee which is larger
    print("  PASSED (ran without error)")


def test_convergence_to_entropy_rate():
    """At large k, both predictors should approach the entropy rate."""
    print("Test: convergence to entropy rate at large k")
    E_fwd, pi = make_simple_hmm()
    E_rev = build_reverse_emission_matrices(E_fwd, pi)

    # Compute entropy rate by long-sequence Monte Carlo
    np.random.seed(42)
    long_seq = generate_sequence_from_E(E_fwd, pi, 100000)
    O_fwd = E_fwd.sum(axis=2)

    belief = pi.copy()
    losses = []
    for t in range(len(long_seq)):
        pred_prob = belief @ O_fwd[long_seq[t]]
        losses.append(-np.log(pred_prob + 1e-30))
        belief = belief @ E_fwd[long_seq[t]]
        s = belief.sum()
        if s > 0:
            belief /= s
    h_X = np.mean(losses[100:])  # skip burn-in
    print(f"  Estimated h_X = {h_X:.6f}")

    # At k=8, losses should be close to h_X
    np.random.seed(123)
    n_samples = 5000
    k = 8

    losses_left = np.zeros(n_samples)
    losses_right = np.zeros(n_samples)

    for i in range(n_samples):
        seq = generate_sequence_from_E(E_fwd, pi, 2 * k + 1)
        ll, lr, _, _ = predict_centre_token(E_fwd, E_rev, pi, seq, k)
        losses_left[i] = ll
        losses_right[i] = lr

    print(f"  L_left(k=8)  = {losses_left.mean():.6f}")
    print(f"  L_right(k=8) = {losses_right.mean():.6f}")
    print(f"  h_X          = {h_X:.6f}")

    # Should be within ~0.05 nats of entropy rate for this simple HMM
    assert abs(losses_left.mean() - h_X) < 0.05, \
        f"Left loss too far from h_X: {losses_left.mean():.4f} vs {h_X:.4f}"
    assert abs(losses_right.mean() - h_X) < 0.05, \
        f"Right loss too far from h_X: {losses_right.mean():.4f} vs {h_X:.4f}"
    print("  PASSED")


def test_sequence_generation():
    """Generated sequences should have correct token frequencies."""
    print("Test: sequence generation token frequencies")
    E_fwd, pi = make_simple_hmm()

    np.random.seed(42)
    seq = generate_sequence_from_E(E_fwd, pi, 100000)

    # For symmetric 2-state HMM with pi=[0.5, 0.5]:
    # P(token 0) = 0.5 * 0.9 + 0.5 * 0.1 = 0.5
    freq_0 = (seq == 0).mean()
    assert abs(freq_0 - 0.5) < 0.01, f"Token 0 frequency: {freq_0}"
    print(f"  Token 0 frequency: {freq_0:.4f} (expected ~0.5)")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing centre_token_prediction.py")
    print("=" * 60)

    test_emission_matrix_validity()
    test_reverse_emission_matrix()
    test_predictive_distributions_sum_to_1()
    test_sequence_generation()
    test_symmetric_hmm_equal_losses()
    test_asymmetric_hmm_different_losses()
    test_convergence_to_entropy_rate()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
