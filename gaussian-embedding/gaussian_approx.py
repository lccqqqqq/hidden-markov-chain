"""
Gaussian approximation experiment for HMM token frequencies.

Tests the CLT for HMMs: as sequence length L → ∞, the distribution of
unigram token frequencies becomes Gaussian. We verify this by:
1. Computing exact log P(x) via the forward algorithm
2. Fitting a quadratic form in δf = f(x) - μ to log P(x)
3. Comparing the fitted inverse covariance to the empirical Σ⁻¹
"""

import sys
import os
import numpy as np
from scipy.special import logsumexp, gammaln
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hmm import Mess3Proc


def log_forward(sequence, hmm):
    """
    Compute log P(sequence) via the forward algorithm with log-sum-exp stability.

    Args:
        sequence: 1D array of integer tokens, length L
        hmm: HMM instance with emission_matrices and get_stationary_distribution()

    Returns:
        log P(sequence) as a float
    """
    E = hmm.emission_matrices  # (d_vocab, n_states, n_states)
    pi = hmm.get_stationary_distribution()  # (n_states,)
    n_states = hmm.num_hidden_states

    # Initialize: log alpha_0(k) = log[ pi(i) * E[x_0, i, k] ] summed over i
    # But actually alpha_0(k) = sum_i pi(i) * E[x_0, i, k]
    # More standard: alpha_t(s) = P(x_0,...,x_t, S_t=s)
    # With the emission matrix convention E[j, i, k] = P(obs j, transition to k | state i):
    #   alpha_0(k) = sum_i pi(i) * E[x_0, i, k]
    #   alpha_t(k) = sum_i alpha_{t-1}(i) * E[x_t, i, k]

    x0 = sequence[0]
    # log_alpha(k) = log sum_i pi(i) * E[x0, i, k]
    # = log sum_i exp(log pi(i) + log E[x0, i, k])
    log_pi = np.log(pi)
    log_E_x0 = np.log(E[x0] + 1e-300)  # (n_states, n_states), [i, k]
    log_alpha = logsumexp(log_pi[:, None] + log_E_x0, axis=0)  # (n_states,)

    for t in range(1, len(sequence)):
        xt = sequence[t]
        log_E_xt = np.log(E[xt] + 1e-300)  # (n_states, n_states), [i, k]
        # log_alpha_new(k) = log sum_i exp(log_alpha(i) + log E[xt, i, k])
        log_alpha = logsumexp(log_alpha[:, None] + log_E_xt, axis=0)  # (n_states,)

    return logsumexp(log_alpha)


def run_experiment(L=15, N=10000, seed=42):
    """
    Run the Gaussian approximation experiment for sequence length L.

    Args:
        L: sequence length
        N: number of sequences to generate
        seed: random seed
    """
    np.random.seed(seed)
    hmm = Mess3Proc()
    d_vocab = hmm.d_vocab  # 3
    mu = np.ones(d_vocab) / d_vocab  # uniform (1/3, 1/3, 1/3)

    print(f"=== Gaussian Approximation Experiment ===")
    print(f"Process: Mess3 ({d_vocab} tokens, {hmm.num_hidden_states} hidden states)")
    print(f"Sequence length L = {L}, N = {N} sequences")
    print(f"Expected mean frequency: {mu}")
    print(f"Theoretical entropy rate: {hmm.entropy_rate_theory_estimate():.6f} nats/token")
    print()

    # Generate sequences and compute log P and frequencies
    log_probs = np.zeros(N)
    freqs = np.zeros((N, d_vocab))

    for i in range(N):
        _, obs = hmm.generate_sequence(L)
        log_probs[i] = log_forward(obs, hmm)
        for j in range(d_vocab):
            freqs[i, j] = np.sum(obs == j) / L

    delta_f = freqs - mu  # (N, d_vocab)

    # Compute log multinomial coefficient log M(f) = log(L! / prod_j (L*f_j)!)
    counts = (freqs * L).astype(int)  # (N, d_vocab)
    log_multinomial = gammaln(L + 1) - np.sum(gammaln(counts + 1), axis=1)  # (N,)

    # log P(type f) ≈ log P(x) + log M(f)  [if sequences within a type have ~equal prob]
    # More precisely: log P(type f) = log(sum_{x: freq(x)=f} P(x))
    # For rough approximation: P(type f) ≈ M(f) * P(x|type f) ≈ M(f) * exp(E[log P|f])
    log_type_probs = log_probs + log_multinomial

    # --- Basic statistics ---
    print("--- Basic Statistics ---")
    print(f"Mean log P / L = {np.mean(log_probs) / L:.6f} (should ≈ -{hmm.entropy_rate_theory_estimate():.6f})")
    print(f"Mean frequency: {np.mean(freqs, axis=0)}")
    print(f"Std of frequency: {np.std(freqs, axis=0)}")
    print()

    # --- Project to 2D (drop last component, since they sum to 0) ---
    delta_f_2d = delta_f[:, :2]  # (N, 2)

    # --- Method 1: OLS quadratic fit ---
    # Features: 1, δf₀, δf₁, δf₀², δf₀·δf₁, δf₁²
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(delta_f_2d)  # (N, 6)

    reg = LinearRegression(fit_intercept=False)  # bias already in poly features
    reg.fit(X_poly, log_probs)
    log_probs_pred = reg.predict(X_poly)

    ss_res = np.sum((log_probs - log_probs_pred) ** 2)
    ss_tot = np.sum((log_probs - np.mean(log_probs)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Extract Hessian (J_fit) from quadratic coefficients
    # poly features order: [1, δf₀, δf₁, δf₀², δf₀·δf₁, δf₁²]
    feature_names = poly.get_feature_names_out()
    coeffs = reg.coef_

    print("--- OLS Quadratic Fit ---")
    for name, c in zip(feature_names, coeffs):
        print(f"  {name}: {c:.6f}")
    print(f"R² = {r_squared:.6f}")
    print(f"Residual std = {np.std(log_probs - log_probs_pred):.6f}")
    print()

    # Build J_fit (2x2 Hessian matrix, factor of 2 for quadratic terms)
    # log P ≈ c0 + c1*δf0 + c2*δf1 + c3*δf0² + c4*δf0*δf1 + c5*δf1²
    # Hessian H_ij = d²(log P)/d(δfi)d(δfj)
    # H[0,0] = 2*c3, H[0,1] = H[1,0] = c4, H[1,1] = 2*c5
    idx_map = {name: i for i, name in enumerate(feature_names)}
    c_f0f0 = coeffs[idx_map.get('x0^2', idx_map.get('x0 x0', 3))]
    c_f0f1 = coeffs[idx_map.get('x0 x1', 4)]
    c_f1f1 = coeffs[idx_map.get('x1^2', idx_map.get('x1 x1', 5))]

    J_fit = np.array([
        [2 * c_f0f0, c_f0f1],
        [c_f0f1, 2 * c_f1f1]
    ])

    print("--- Inverse Covariance Estimates ---")
    print(f"J_fit (Hessian from quadratic fit):")
    print(f"  {J_fit}")
    print()

    # --- Method 2: Empirical Σ⁻¹ ---
    Sigma_emp = np.cov(delta_f_2d, rowvar=False)  # (2, 2)
    J_emp = np.linalg.inv(Sigma_emp)

    print(f"Σ_emp (empirical covariance):")
    print(f"  {Sigma_emp}")
    print(f"J_emp = Σ⁻¹ (empirical inverse covariance):")
    print(f"  {J_emp}")
    print()

    # --- Comparison ---
    # Scale J_fit to match J_emp: J_fit from log P gives extensive quantity,
    # while Σ_emp is already per-sequence. We need J_fit * (-1) to compare
    # since log P is concave in f around the mode.
    # Actually, let's compare -J_fit/L with J_emp (since log P ~ L * h + L/2 * δf^T J δf
    # would give J_fit ~ -L * J, so J = -J_fit / L)
    # But this depends on the exact scaling. Let's just report both and the ratio.

    # The quadratic fit gives: log P ≈ a + b^T δf + 1/2 δf^T J_fit δf
    # For large L: log P ≈ L*h + L/2 * δf^T * C * δf (where C is some matrix)
    # And Var(δf) = Σ/L, so Σ_emp ≈ Σ/L
    # The fitted quadratic should have J_fit ≈ L * C
    # And J_emp = L * Σ⁻¹
    # If CLT holds: C should relate to -Σ⁻¹, giving J_fit ≈ -L * Σ⁻¹ = -J_emp * L...
    # Hmm, let me think about the scaling more carefully.
    #
    # Actually: if P(f) ∝ exp(-L/2 (f-μ)^T Σ⁻¹_true (f-μ)), then
    # log P(f) = const - L/2 (f-μ)^T Σ⁻¹_true (f-μ)
    # So J_fit (Hessian of log P w.r.t. f) = -L * Σ⁻¹_true
    # And empirical: Σ_emp ≈ Σ_true / L, so Σ⁻¹_emp ≈ L * Σ⁻¹_true
    # Therefore: J_fit ≈ -Σ⁻¹_emp

    print("--- Comparison (sequence-level) ---")
    print(f"-J_fit (should ≈ J_emp if CLT holds):")
    print(f"  {-J_fit}")
    ratio = -J_fit / J_emp
    print(f"Ratio -J_fit / J_emp (should → 1 elementwise):")
    print(f"  {ratio}")
    rel_error = np.linalg.norm(-J_fit - J_emp) / np.linalg.norm(J_emp)
    print(f"Relative Frobenius error ||-J_fit - J_emp|| / ||J_emp|| = {rel_error:.6f}")
    print()

    # --- Type-level analysis (aggregated within types) ---
    # For HMMs, sequences with the same frequency have different P(x) due to ordering.
    # We must aggregate: log P(type f) = log sum_{x: freq(x)=f} P(x)
    # = logsumexp of log P(x) within each type class.
    #
    # Equivalently, we can use the empirical histogram: P(type f) ≈ count(f)/N.
    print("--- Type-Level Analysis (aggregated within types) ---")

    # Group sequences by their count vector (which uniquely determines the type)
    from collections import defaultdict
    type_groups = defaultdict(list)  # counts_tuple -> list of log P(x)
    for i in range(N):
        key = tuple(counts[i])
        type_groups[key].append(log_probs[i])

    n_types = len(type_groups)
    print(f"Number of distinct types: {n_types}")
    group_sizes = [len(v) for v in type_groups.values()]
    print(f"Samples per type: min={min(group_sizes)}, median={np.median(group_sizes):.0f}, max={max(group_sizes)}")

    # Method A: compute log P(type f) = logsumexp(log P(x_i)) for each type
    # This is exact (up to sampling noise) for the sum over sequences in the type.
    # But we want the probability of the TYPE, not the sum of sequence probs.
    # P(type f) = sum_{x in type} P(x), so log P(type f) = logsumexp(log P(x_i) for x_i sampled in type).
    # Correction: our samples are drawn from P, so within a type group,
    # P(type f) ≈ (count_f / N) and also P(type f) = M(f) * E[P(x)|type f].
    # We can estimate E[P(x)|type f] from logsumexp(log P(x_i)) - log(count_f).

    # Method B: just use the empirical histogram log P(type f) = log(count_f / N)
    type_freqs_2d = []
    log_type_prob_hist = []  # from histogram
    log_type_prob_fwd = []   # from forward algorithm aggregation
    type_weights = []

    for key, group_log_probs in type_groups.items():
        count_f = len(group_log_probs)
        f_vec = np.array(key) / L
        delta = f_vec[:2] - mu[:2]
        type_freqs_2d.append(delta)
        type_weights.append(count_f)

        # Histogram estimate
        log_type_prob_hist.append(np.log(count_f / N))

        # Forward-algorithm estimate: log P(type f) = logsumexp(log P(x_i)) + log(M(f)/count_f)
        # Since each x_i is drawn from P, the count_f samples are a random subset of the M(f) sequences.
        # E[P(x)|type] ≈ exp(logsumexp(log P(x_i)) - log(count_f))
        # P(type f) = M(f) * E[P(x)|type] ≈ M(f) * exp(logsumexp(log P) - log(count_f))
        log_M = gammaln(L + 1) - np.sum(gammaln(np.array(key) + 1))
        log_mean_P = logsumexp(group_log_probs) - np.log(count_f)
        log_type_prob_fwd.append(log_M + log_mean_P)

    type_freqs_2d = np.array(type_freqs_2d)
    log_type_prob_hist = np.array(log_type_prob_hist)
    log_type_prob_fwd = np.array(log_type_prob_fwd)
    type_weights = np.array(type_weights)

    # Fit quadratic to type-level log probs (weighted by count for better stats)
    poly_type = PolynomialFeatures(degree=2, include_bias=True)
    X_type = poly_type.fit_transform(type_freqs_2d)

    for label, y_type in [("histogram", log_type_prob_hist), ("forward-agg", log_type_prob_fwd)]:
        reg_t = LinearRegression(fit_intercept=False)
        reg_t.fit(X_type, y_type, sample_weight=type_weights)
        y_pred = reg_t.predict(X_type)

        ss_res = np.sum(type_weights * (y_type - y_pred) ** 2)
        ss_tot = np.sum(type_weights * (y_type - np.average(y_type, weights=type_weights)) ** 2)
        r2 = 1 - ss_res / ss_tot

        ct = reg_t.coef_
        fn = poly_type.get_feature_names_out()
        im = {name: i for i, name in enumerate(fn)}

        J_t = np.array([
            [2 * ct[im.get('x0^2', 3)], ct[im.get('x0 x1', 4)]],
            [ct[im.get('x0 x1', 4)], 2 * ct[im.get('x1^2', 5)]]
        ])

        print(f"\n  [{label}]")
        print(f"  R² = {r2:.6f}")
        print(f"  J_fit_type = {J_t}")
        print(f"  -J_fit_type = {-J_t}")
        ratio_t = -J_t / J_emp
        print(f"  Ratio -J_fit_type / J_emp = {ratio_t}")
        rel_err = np.linalg.norm(-J_t - J_emp) / np.linalg.norm(J_emp)
        print(f"  Relative Frobenius error = {rel_err:.6f}")

        if label == "forward-agg":
            J_fit_type = J_t
            r2_type = r2
            coeffs_type = ct

    print()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        os.path.join(results_dir, f'gaussian_approx_L{L}.npz'),
        L=L, N=N,
        log_probs=log_probs, freqs=freqs,
        log_multinomial=log_multinomial, log_type_probs=log_type_probs,
        J_fit=J_fit, J_emp=J_emp, Sigma_emp=Sigma_emp,
        J_fit_type=J_fit_type,
        r_squared=r_squared, r2_type=r2_type,
        coeffs=coeffs, coeffs_type=coeffs_type,
        feature_names=feature_names,
    )
    print(f"Results saved to {results_dir}/gaussian_approx_L{L}.npz")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=15, help='Sequence length')
    parser.add_argument('--N', type=int, default=10000, help='Number of sequences')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    run_experiment(L=args.L, N=args.N, seed=args.seed)
