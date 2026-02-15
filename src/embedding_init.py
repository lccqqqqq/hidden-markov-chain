"""
Embedding initialization strategies for the fixed-embedding experiment.

Each initializer returns a torch.Tensor of shape [d_vocab, d_model].
"""

import torch
import numpy as np
import warnings
from math import comb
from pathlib import Path


def init_trained_embedding(checkpoint_path, d_vocab, d_model, **kwargs):
    """Load W_E from a trained model checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    W_E = state_dict["embed.W_E"]
    assert W_E.shape == (d_vocab, d_model), (
        f"Checkpoint W_E shape {W_E.shape} != expected ({d_vocab}, {d_model})"
    )
    return W_E.clone()


def init_random_embedding(d_vocab, d_model, seed=0, **kwargs):
    """Random unit-norm embeddings on S^{d_model-1}."""
    rng = torch.Generator().manual_seed(seed)
    W_E = torch.randn(d_vocab, d_model, generator=rng)
    W_E = W_E / W_E.norm(dim=1, keepdim=True)
    return W_E


def init_coulomb_embedding(d_vocab, d_model, seed=0, n_steps=5000, lr=0.01, **kwargs):
    """
    Thomson problem: place d_vocab points on S^{d_model-1} minimizing
    Coulomb energy sum_{i<j} 1/||x_i - x_j||.
    """
    rng = torch.Generator().manual_seed(seed)
    W_E = torch.randn(d_vocab, d_model, generator=rng)
    W_E = W_E / W_E.norm(dim=1, keepdim=True)
    W_E = W_E.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([W_E], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        dists = torch.pdist(W_E)
        energy = (1.0 / dists).sum()
        energy.backward()
        optimizer.step()

        # Project back onto sphere
        with torch.no_grad():
            W_E.data = W_E.data / W_E.data.norm(dim=1, keepdim=True)

        if (step + 1) % 1000 == 0 or step == n_steps - 1:
            print(f"  Coulomb step {step+1}/{n_steps}: energy = {energy.item():.2f}")

    return W_E.detach().clone()


def init_sparse_binary_embedding(d_vocab, d_model, nnz_per_row=3, seed=0, **kwargs):
    """
    Deprecated: use 'onehot' instead for d_model >= d_vocab.

    Each row has nnz_per_row random 1s, then row-normalized to unit norm.
    """
    warnings.warn(
        "sparse_binary is deprecated. Use 'onehot' for identity embeddings "
        "or 'random' for random embeddings.",
        DeprecationWarning,
        stacklevel=2,
    )
    max_unique = comb(d_model, nnz_per_row)
    if max_unique < d_vocab:
        warnings.warn(
            f"C({d_model},{nnz_per_row})={max_unique} < d_vocab={d_vocab}: "
            f"duplicate rows are inevitable. Consider nnz_per_row={nnz_per_row+1} "
            f"which gives C({d_model},{nnz_per_row+1})={comb(d_model, nnz_per_row+1)}."
        )

    rng = np.random.default_rng(seed)
    W_E = np.zeros((d_vocab, d_model), dtype=np.float32)
    for i in range(d_vocab):
        cols = rng.choice(d_model, size=nnz_per_row, replace=False)
        W_E[i, cols] = 1.0

    W_E = torch.from_numpy(W_E)
    W_E = W_E / W_E.norm(dim=1, keepdim=True)
    return W_E


def init_onehot_embedding(d_vocab, d_model, **kwargs):
    """
    One-hot (identity) embedding. Requires d_model >= d_vocab.
    If d_model > d_vocab, the extra dimensions are zero-padded.
    """
    if d_model < d_vocab:
        raise ValueError(
            f"One-hot embedding requires d_model >= d_vocab, "
            f"got d_model={d_model}, d_vocab={d_vocab}"
        )
    W_E = torch.zeros(d_vocab, d_model)
    W_E[:, :d_vocab] = torch.eye(d_vocab)
    return W_E


def init_pmi_embedding(d_vocab, d_model, shard_dir=None, window_size=16, **kwargs):
    """
    SVD of the PPMI matrix computed from co-occurrence statistics.
    Uses cached co-occurrence counts if available.
    """
    cache_path = Path("out/cooccurrence_counts.npy")

    if cache_path.exists():
        print(f"  Loading cached co-occurrence counts from {cache_path}")
        cooccurrence_matrix = np.load(cache_path)
    else:
        if shard_dir is None:
            shard_dir = "data/datasets/cylinder_graph_hmm/shards"
        shard_dir = Path(shard_dir)

        print(f"  Computing co-occurrence from shards in {shard_dir}")
        all_data = []
        for shard_path in sorted(shard_dir.glob("obs_*.npy")):
            shard_data = np.load(shard_path)
            all_data.append(shard_data)
            print(f"    Loaded {shard_path.name}: {len(shard_data)} tokens")

        if not all_data:
            raise FileNotFoundError(
                f"No observation shards found in {shard_dir}. "
                "Run data generation first or provide cached counts."
            )

        observations = np.concatenate(all_data)
        print(f"  Total tokens: {len(observations):,}")

        cooccurrence_matrix = np.zeros((d_vocab, d_vocab), dtype=np.float64)
        num_windows = len(observations) - window_size + 1

        for window_start in range(num_windows):
            window = observations[window_start:window_start + window_size]
            unique_tokens = np.unique(window)
            for ti in unique_tokens:
                for tj in unique_tokens:
                    cooccurrence_matrix[ti, tj] += 1

            if (window_start + 1) % 500000 == 0:
                print(f"    Processed {window_start+1:,} / {num_windows:,} windows")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, cooccurrence_matrix)
        print(f"  Saved co-occurrence counts to {cache_path}")

    # Compute PPMI
    total = cooccurrence_matrix.sum()
    cooc_prob = cooccurrence_matrix / total
    p_i = cooc_prob.sum(axis=1)
    p_j = cooc_prob.sum(axis=0)
    expected = np.outer(p_i, p_j)

    epsilon = 1e-10
    pmi = np.log((cooc_prob + epsilon) / (expected + epsilon))
    ppmi = np.maximum(pmi, 0)

    # SVD → W_E = U[:, :d_model] * sqrt(S[:d_model])
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    W_E = U[:, :d_model] * np.sqrt(S[:d_model])

    return torch.from_numpy(W_E.astype(np.float32))


# ── Registry & Dispatcher ──────────────────────────────────────────────

EMBEDDING_REGISTRY = {
    "trained": init_trained_embedding,
    "random": init_random_embedding,
    "coulomb": init_coulomb_embedding,
    "onehot": init_onehot_embedding,
    "pmi": init_pmi_embedding,
    "sparse_binary": init_sparse_binary_embedding,  # deprecated
}


def create_embedding(mode, d_vocab, d_model, **kwargs):
    """Dispatch to the appropriate embedding initializer."""
    if mode not in EMBEDDING_REGISTRY:
        available = ", ".join(EMBEDDING_REGISTRY.keys())
        raise ValueError(f"Unknown embedding mode '{mode}'. Available: {available}")
    return EMBEDDING_REGISTRY[mode](d_vocab=d_vocab, d_model=d_model, **kwargs)
