"""
Training script for bilinear Deep Sets.

Supports two data modes:
  gaussian — sample from a known Gaussian energy model p(w) ∝ exp(-E(N(w)))
             with exact log-probability labels and known ground truth parameters.
  hmm      — sample from a two-state two-token HMM, label via forward algorithm.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import logsumexp, gammaln

from model import TwoStateTwoTokenHMM, BilinearDeepSets


# ---------------------------------------------------------------------------
# Forward algorithm (from gaussian-embedding/gaussian_approx.py)
# ---------------------------------------------------------------------------

def log_forward(sequence, hmm):
    """Compute log P(sequence) via the forward algorithm."""
    E = hmm.emission_matrices
    pi = hmm.get_stationary_distribution()

    x0 = sequence[0]
    log_pi = np.log(pi)
    log_E_x0 = np.log(E[x0] + 1e-300)
    log_alpha = logsumexp(log_pi[:, None] + log_E_x0, axis=0)

    for t in range(1, len(sequence)):
        xt = sequence[t]
        log_E_xt = np.log(E[xt] + 1e-300)
        log_alpha = logsumexp(log_alpha[:, None] + log_E_xt, axis=0)

    return logsumexp(log_alpha)


# ---------------------------------------------------------------------------
# Gaussian energy model data generation (V=2)
# ---------------------------------------------------------------------------

def generate_gaussian_data(L, N, A, p0=0.5, seed=42):
    """
    Generate sequences from the Gaussian energy model on counts:
        p(w_{1:L}) ∝ exp( -A/(2L) * (N_0 - L*p0)^2 )

    where A > 0 is the inverse-variance (precision) parameter.

    For V=2, the energy depends only on N_0. We:
      1. Compute unnormalized log P for each N_0 ∈ {0,...,L}
      2. Normalize to get P(N_0)
      3. Compute P(sequence | N_0) = 1 / C(L, N_0)
      4. Sample sequences by drawing N_0, then random permutations

    Returns:
        sequences: (N, L) torch.LongTensor
        log_probs: (N,) torch.FloatTensor — exact log p(sequence)
        ground_truth: dict with ground truth network parameters
    """
    np.random.seed(seed)

    # Energy for each count N_0
    N0_vals = np.arange(L + 1)
    energy = A / (2 * L) * (N0_vals - L * p0) ** 2

    # log C(L, N_0) = log(L!) - log(N_0!) - log((L-N_0)!)
    log_multinomial = gammaln(L + 1) - gammaln(N0_vals + 1) - gammaln(L - N0_vals + 1)

    # log p_seq(w) = -energy(N_0) - log C(L, N_0) - log Z_seq
    # where Z_seq = sum_{N_0} C(L,N_0) * exp(-energy(N_0)) = sum_{N_0} exp(log_multinomial - energy)
    log_unnorm_seq = -energy - log_multinomial
    log_Z_seq = logsumexp(-energy + log_multinomial)
    log_p_seq = log_unnorm_seq - log_Z_seq  # log P(specific sequence with count N_0)

    # Distribution over N_0 for sampling:
    # P(N_0 = k) = C(L,k) * exp(-energy(k)) / Z_seq
    log_p_N0 = log_multinomial - energy - log_Z_seq
    p_N0 = np.exp(log_p_N0)
    p_N0 /= p_N0.sum()  # renormalize for numerical safety

    # Sample sequences
    sequences = []
    log_probs_list = []
    for _ in range(N):
        n0 = np.random.choice(L + 1, p=p_N0)
        seq = np.zeros(L, dtype=np.int64)
        # Place n0 zeros at random positions
        positions = np.random.permutation(L)
        seq[positions[:n0]] = 0
        seq[positions[n0:]] = 1
        sequences.append(seq)
        log_probs_list.append(log_p_seq[n0])

    sequences = torch.from_numpy(np.stack(sequences)).long()
    log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

    # Derive ground truth network parameters (section 10.9)
    # For V=2, Σ has rank r=1.
    # The long-run covariance for this Gaussian model:
    #   Σ = L * Cov(N)/L  ... but here the model IS the Gaussian, so:
    #   p(w) ∝ exp(-A/(2L) * (N_0 - L*p0)^2)
    # Rewrite in terms of s = e_0*N_0 + e_1*N_1 = (e_0-e_1)*N_0 + e_1*L
    #
    # The collapsed one-table form (PDF p.10):
    #   φ(i) = (1/√L) Λ_r^{-1} (e_i - ē), then h = Σ φ(w_t), log p = c - 0.5*||h||^2
    #
    # For V=2 with uniform p0=0.5:
    #   Σ = σ^2 * vv^T where v = (1,-1)/√2, σ^2 = 1/A (inverse precision)
    #   E = U Λ^{1/2} where U = v/||v|| = (1,-1)/√2, Λ = σ^2
    #   e_0 = σ/√2, e_1 = -σ/√2
    #   ē = E^T p = 0 (for p0=0.5)
    #
    # Ground truth bilinear parameters:
    #   Embedding: e_i (from covariance factorization)
    #   Using the generic form log p = c - 0.5*(s-d)^T M (s-d):
    #   From 10.4: M = Λ_r^{-2}/L = A^2/L (scalar for r=1)
    #   d = L * ē = 0 (for p0=0.5)
    #
    # But let's verify directly. For r=1:
    #   s = e_0*N_0 + e_1*(L-N_0) = Δ*N_0 + e_1*L where Δ = e_0-e_1
    #   log p = c - 0.5*M*(Δ*N_0 + e_1*L - d)^2
    #   We want this to equal (up to constant): -A/(2L)*(N_0 - L*p0)^2 - log C(L,N_0) - log Z
    #   The -log C(L,N_0) term is NOT quadratic in N_0, so the model can only
    #   fit the quadratic part exactly. For large L, the multinomial coefficient
    #   is approximately quadratic near its peak (Stirling), so the fit improves.
    #
    # For simplicity, compute the "best quadratic fit" to the full target
    # log_p_seq[N_0] as a function of N_0, and derive ground truth from that.

    # Fit quadratic: log_p_seq ≈ α + β*(N_0 - L*p0) + γ*(N_0 - L*p0)^2
    x = N0_vals - L * p0
    # Weighted least squares with weight = P(N_0)
    w = p_N0
    X = np.column_stack([np.ones_like(x), x, x**2])
    W = np.diag(w)
    coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ log_p_seq)
    alpha, beta, gamma = coeffs

    # Map quadratic fit to network parameters.
    # Model output: c - 0.5*M*(Δ*N_0 + e_1*L - d)^2
    # = c - 0.5*M*Δ^2*(N_0 - (d - e_1*L)/Δ)^2
    # Compare with: alpha + beta*(N_0 - L*p0) + gamma*(N_0 - L*p0)^2
    # => -0.5*M*Δ^2 = gamma, center = L*p0 - beta/(2*gamma)
    # Free choice: set Δ = e_0 - e_1 = 1 (gauge choice)
    # Then M = -2*gamma, and the center = (d - e_1*L)/Δ
    # Set e_0 = 0.5, e_1 = -0.5 => Δ = 1, e_1*L = -L/2
    # Center = d + L/2 = L*p0 - beta/(2*gamma) => d = L*p0 - L/2 - beta/(2*gamma)
    # c = alpha + beta^2/(4*gamma) ... hmm, let me redo this cleanly.

    # With e_0=0.5, e_1=-0.5: s = 0.5*N_0 - 0.5*N_1 = N_0 - L/2
    # log p = c - 0.5*M*(N_0 - L/2 - d)^2
    # = c - 0.5*M*(δN - d)^2  where δN = N_0 - L/2
    # = (c - 0.5*M*d^2) + M*d*δN - 0.5*M*δN^2
    # Compare: alpha + beta*δN + gamma*δN^2  (δN = N_0 - L*p0, and here p0=0.5 so δN = N_0-L/2)
    # => -0.5*M = gamma => M = -2*gamma
    # => M*d = beta => d = beta/M = -beta/(2*gamma)
    # => c - 0.5*M*d^2 = alpha => c = alpha + 0.5*M*d^2 = alpha - gamma*d^2 = alpha - beta^2/(4*gamma)

    # Handle general p0: s = 0.5*N_0 - 0.5*(L-N_0) = N_0 - L/2 (regardless of p0)
    # But the quadratic fit is centered at L*p0, not L/2.
    # δN_fit = N_0 - L*p0, δN_model = N_0 - L/2 - d
    # We need δN_model = δN_fit => d = L*p0 - L/2

    M_gt = -2 * gamma
    d_gt = -beta / (2 * gamma) + (L * p0 - L / 2)  # shift for p0 ≠ 0.5
    # Actually let me redo. With embeddings e_0=0.5, e_1=-0.5:
    # s = N_0 - L/2
    # log p = c - 0.5*M*(s - d)^2 = c - 0.5*M*((N_0 - L/2) - d)^2
    # Let u = N_0 - L*p0 (centered at mean). Then N_0 = u + L*p0, s = u + L*p0 - L/2
    # log p = c - 0.5*M*(u + L*p0 - L/2 - d)^2
    # Define D = L*p0 - L/2 - d. Then:
    # log p = c - 0.5*M*(u + D)^2 = c - 0.5*M*D^2 - M*D*u - 0.5*M*u^2
    # Match: alpha + beta*u + gamma*u^2
    # => -0.5*M = gamma => M = -2*gamma
    # => -M*D = beta => D = -beta/M = beta/(2*gamma)
    # => d = L*p0 - L/2 - D = L*(p0 - 0.5) - beta/(2*gamma)
    # => c - 0.5*M*D^2 = alpha => c = alpha + gamma*D^2 = alpha + beta^2/(4*gamma)

    D = beta / (2 * gamma)
    d_gt = L * (p0 - 0.5) - D
    c_gt = alpha + beta**2 / (4 * gamma)
    e_gt = np.array([[0.5], [-0.5]])  # (V, r=1)
    M_gt_arr = np.array([[-2 * gamma]])  # (1, 1)

    ground_truth = {
        'embeddings': e_gt,
        'M': M_gt_arr,
        'd': np.array([d_gt]),
        'c': c_gt,
        'quadratic_fit': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'log_p_seq': log_p_seq,
        'p_N0': p_N0,
    }

    return sequences, log_probs, ground_truth


# ---------------------------------------------------------------------------
# HMM data generation
# ---------------------------------------------------------------------------

def generate_hmm_data(hmm, N, L, seed=42):
    """Generate N sequences of length L from HMM with exact log-probabilities."""
    np.random.seed(seed)
    sequences = []
    log_probs = []
    for _ in range(N):
        _, obs = hmm.generate_sequence(L)
        sequences.append(obs)
        log_probs.append(log_forward(obs, hmm))
    sequences = torch.from_numpy(np.stack(sequences)).long()
    log_probs = torch.tensor(log_probs, dtype=torch.float32)
    return sequences, log_probs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.mode == 'gaussian':
        print(f"=== Gaussian energy model ===")
        print(f"  A={args.A}, p0={args.p0}, L={args.L}, N={args.N}")

        tag = f"gaussian_A{args.A}_p{args.p0}_L{args.L}_N{args.N}"
        data_path = os.path.join(data_dir, f"{tag}.npz")

        if os.path.exists(data_path) and not args.regenerate:
            print(f"Loading cached data from {data_path}")
            data = np.load(data_path, allow_pickle=True)
            sequences = torch.from_numpy(data['sequences']).long()
            log_probs = torch.from_numpy(data['log_probs']).float()
            ground_truth = data['ground_truth'].item()
        else:
            sequences, log_probs, ground_truth = generate_gaussian_data(
                args.L, args.N, args.A, args.p0, seed=args.seed
            )
            np.savez(data_path,
                     sequences=sequences.numpy(),
                     log_probs=log_probs.numpy(),
                     ground_truth=ground_truth)
            print(f"Saved to {data_path}")

        V = 2
        qf = ground_truth['quadratic_fit']
        print(f"  Quadratic fit: α={qf['alpha']:.6f}, β={qf['beta']:.6f}, γ={qf['gamma']:.6f}")
        print(f"  Ground truth: M={ground_truth['M']}, d={ground_truth['d']}, c={ground_truth['c']:.6f}")
        print()

    else:  # hmm mode
        hmm = TwoStateTwoTokenHMM(eps=args.eps, a=args.a)
        print(f"=== HMM mode ===")
        print(f"  eps={args.eps}, a={args.a}")
        print(f"  Entropy rate: {hmm.entropy_rate_theory_estimate():.6f} nats/token")

        tag = f"hmm_eps{args.eps}_a{args.a}_L{args.L}_N{args.N}"
        data_path = os.path.join(data_dir, f"{tag}.npz")

        if os.path.exists(data_path) and not args.regenerate:
            print(f"Loading cached data from {data_path}")
            data = np.load(data_path)
            sequences = torch.from_numpy(data['sequences']).long()
            log_probs = torch.from_numpy(data['log_probs']).float()
        else:
            print(f"Generating {args.N} sequences of length {args.L}...")
            sequences, log_probs = generate_hmm_data(hmm, args.N, args.L, seed=args.seed)
            np.savez(data_path,
                     sequences=sequences.numpy(),
                     log_probs=log_probs.numpy())
            print(f"Saved to {data_path}")

        V = hmm.d_vocab
        ground_truth = None

    print(f"  log P range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
    print(f"  log P mean: {log_probs.mean():.3f}, std: {log_probs.std():.3f}")
    print()

    # Train/val split
    n_val = int(args.N * args.val_fraction)
    n_train = args.N - n_val
    perm = torch.randperm(args.N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(sequences[train_idx], log_probs[train_idx])
    val_ds = TensorDataset(sequences[val_idx], log_probs[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model (with reproducible initialization)
    if args.model_seed is not None:
        torch.manual_seed(args.model_seed)
    model = BilinearDeepSets(V=V, r=args.r)

    # Initialize near ground truth if available and requested
    if ground_truth is not None and args.init_gt:
        print("Initializing near ground truth")
        model.init_from_ground_truth(
            ground_truth['embeddings'],
            ground_truth['M'],
            ground_truth['d'],
            ground_truth['c'],
            noise=args.init_noise,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model: V={V}, r={args.r}, params={sum(p.numel() for p in model.parameters())}")
    print(f"Training: {n_train} train, {n_val} val, {args.epochs} epochs, lr={args.lr}")
    print()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val_batches = 0
                for x_batch, y_batch in val_loader:
                    pred = model(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                    n_val_batches += 1
                val_loss /= n_val_batches
            print(f"Epoch {epoch:4d}  train MSE={train_loss:.6f}  val MSE={val_loss:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_pred = model(sequences)
        residual = all_pred - log_probs
        ss_res = (residual ** 2).sum().item()
        ss_tot = ((log_probs - log_probs.mean()) ** 2).sum().item()
        r_squared = 1 - ss_res / ss_tot

    print()
    print(f"=== Results ===")
    print(f"Final R² = {r_squared:.6f}")
    print(f"Final MSE = {ss_res / len(log_probs):.6f}")
    print()

    # Print learned parameters
    e = model.embedding.weight.data
    M_raw = model.M.data
    M_sym = (0.5 * (M_raw + M_raw.T)).numpy()
    d = model.d.data.numpy()
    c = model.c.data.item()
    print(f"Learned parameters:")
    for i in range(V):
        print(f"  e_{i} = {e[i].numpy()}")
    print(f"  M = {M_sym}")
    print(f"  d = {d}")
    print(f"  c = {c:.6f}")

    if ground_truth is not None:
        print(f"\nGround truth:")
        print(f"  e = {ground_truth['embeddings'].squeeze()}")
        print(f"  M = {ground_truth['M'].squeeze()}")
        print(f"  d = {ground_truth['d'].squeeze()}")
        print(f"  c = {ground_truth['c']:.6f}")

    # Save results to JSON
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        result = {
            'args': vars(args),
            'r_squared': r_squared,
            'mse': ss_res / len(log_probs),
            'learned': {
                'embeddings': e.numpy().tolist(),
                'M': M_sym.tolist(),
                'd': d.tolist(),
                'c': c,
            },
        }
        if ground_truth is not None:
            result['ground_truth'] = {
                'embeddings': ground_truth['embeddings'].tolist(),
                'M': ground_truth['M'].tolist(),
                'd': ground_truth['d'].tolist(),
                'c': float(ground_truth['c']),
            }
        tag = f"seed{args.model_seed}" if args.model_seed is not None else "default"
        result_path = os.path.join(args.save_dir, f"result_{tag}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Train bilinear Deep Sets")
    parser.add_argument("--mode", choices=['gaussian', 'hmm'], default='gaussian')

    # Gaussian mode
    parser.add_argument("--A", type=float, default=2.0, help="Precision parameter for Gaussian energy")
    parser.add_argument("--p0", type=float, default=0.5, help="Mean token-0 frequency")

    # HMM mode
    parser.add_argument("--eps", type=float, default=0.1, help="Emission noise")
    parser.add_argument("--a", type=float, default=0.3, help="Transition probability")

    # Shared
    parser.add_argument("--L", type=int, default=50, help="Sequence length")
    parser.add_argument("--N", type=int, default=50000, help="Number of sequences")
    parser.add_argument("--r", type=int, default=1, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate data")
    parser.add_argument("--init_gt", action="store_true", help="Initialize near ground truth (gaussian mode)")
    parser.add_argument("--init_noise", type=float, default=0.1, help="Noise scale for GT initialization")
    parser.add_argument("--model_seed", type=int, default=None, help="Random seed for model initialization")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save JSON results")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
