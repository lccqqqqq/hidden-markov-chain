"""
Comprehensive BPE tokenization analysis for the Cylinder Graph HMM.

Produces figures for the BPE experiment report:
  1. Merge table heatmap (which HMM tokens get merged)
  2. Compression ratio vs vocab size
  3. Token frequency distributions (raw HMM vs BPE)
  4. Merge n-gram length distribution
  5. Entropy analysis: raw vs BPE token entropy rates

Usage:
    python scripts/analyze_bpe.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from bpe_tokenizer import (
    train_bpe, encode_sequence, decode_to_int_seq,
    int_seq_to_string, CHAR_OFFSET
)

FIG_DIR = Path("figures/bpe_experiment")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SHARDS_DIR = Path("data/datasets/cylinder_graph_hmm/shards")


def load_data():
    """Load and concatenate all shards."""
    shard_files = sorted(SHARDS_DIR.glob("obs_*.npy"))
    all_obs = np.concatenate([np.load(f) for f in shard_files])
    print(f"Loaded {len(all_obs)} tokens, range [{all_obs.min()}, {all_obs.max()}]")
    return all_obs


def train_tokenizer(all_obs, vocab_size=128):
    """Train BPE tokenizer and return it."""
    tokenizer = train_bpe(str(SHARDS_DIR), vocab_size=vocab_size)
    return tokenizer


def get_merge_info(tokenizer, n_base=48):
    """Extract merge table: list of (bpe_id, hmm_token_list)."""
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    merges = []
    for idx in range(n_base, tokenizer.get_vocab_size()):
        if idx in id_to_token:
            bpe_tok = id_to_token[idx]
            hmm_tokens = tuple(ord(c) - CHAR_OFFSET for c in bpe_tok)
            merges.append((idx, hmm_tokens))
    return merges


def fig1_merge_cooccurrence(merges, n_base=48):
    """Heatmap showing which pairs of HMM tokens appear together in BPE merges."""
    cooccur = np.zeros((n_base, n_base), dtype=int)
    for _, hmm_toks in merges:
        for i in range(len(hmm_toks) - 1):
            cooccur[hmm_toks[i], hmm_toks[i + 1]] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cooccur, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Next HMM token')
    ax.set_ylabel('Current HMM token')
    ax.set_title('BPE Merge Bigram Co-occurrence\n(which HMM token pairs are merged)')
    plt.colorbar(im, ax=ax, label='Count in merge table')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "merge_cooccurrence.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "merge_cooccurrence.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved merge_cooccurrence")


def fig2_merge_lengths(merges):
    """Distribution of n-gram lengths in the BPE merge table."""
    lengths = [len(hmm_toks) for _, hmm_toks in merges]
    length_counts = Counter(lengths)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    lens = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lens]
    ax.bar(lens, counts, color='C0', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Merge n-gram length (HMM tokens)')
    ax.set_ylabel('Number of merges')
    ax.set_title('BPE Merge Table: N-gram Length Distribution')
    ax.set_xticks(lens)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "merge_lengths.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "merge_lengths.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved merge_lengths")


def fig3_compression_vs_vocab(all_obs):
    """Compression ratio as a function of BPE vocabulary size."""
    vocab_sizes = [56, 64, 80, 96, 112, 128, 160, 192, 256]
    ratios = []

    # Use a subsample for speed
    subsample = all_obs[:1_000_000]

    for vs in vocab_sizes:
        print(f"  Training vocab_size={vs}...")
        tok = train_bpe(str(SHARDS_DIR), vocab_size=vs)
        bpe_ids = encode_sequence(tok, subsample)
        ratio = len(subsample) / len(bpe_ids)
        ratios.append(ratio)
        print(f"    compression ratio = {ratio:.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(vocab_sizes, ratios, 'o-', color='C0')
    ax.set_xlabel('BPE vocabulary size')
    ax.set_ylabel('Compression ratio (raw / BPE tokens)')
    ax.set_title('Compression vs Vocabulary Size')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No compression')
    ax.axvline(x=128, color='C1', linestyle='--', alpha=0.7, label='Used: V=128')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compression_vs_vocab.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "compression_vs_vocab.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved compression_vs_vocab")
    return dict(zip(vocab_sizes, ratios))


def fig4_token_frequency(all_obs, tokenizer):
    """Compare raw HMM token frequencies vs BPE token frequencies."""
    # Raw HMM token distribution
    raw_counts = Counter(all_obs.tolist())
    n_base = int(all_obs.max()) + 1

    # BPE token distribution (use subsample)
    subsample = all_obs[:2_000_000]
    bpe_ids = encode_sequence(tokenizer, subsample)
    bpe_counts = Counter(bpe_ids.tolist())
    vocab_size = tokenizer.get_vocab_size()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw HMM tokens
    ax = axes[0]
    raw_freqs = [raw_counts.get(i, 0) for i in range(n_base)]
    raw_freqs = np.array(raw_freqs) / sum(raw_freqs)
    ax.bar(range(n_base), raw_freqs, color='C0', alpha=0.7)
    ax.set_xlabel('HMM token ID')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Raw HMM Token Distribution (V={n_base})')
    ax.set_xlim(-0.5, n_base - 0.5)

    # BPE tokens
    ax = axes[1]
    bpe_freqs = [bpe_counts.get(i, 0) for i in range(vocab_size)]
    bpe_freqs = np.array(bpe_freqs) / sum(bpe_freqs)
    ax.bar(range(vocab_size), bpe_freqs, color='C1', alpha=0.7, width=1.0)
    ax.set_xlabel('BPE token ID')
    ax.set_ylabel('Frequency')
    ax.set_title(f'BPE Token Distribution (V={vocab_size})')
    ax.axvline(x=n_base - 0.5, color='red', linestyle='--', alpha=0.5,
               label=f'Base tokens (0-{n_base-1})')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_frequencies.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "token_frequencies.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved token_frequencies")

    # Compute entropy
    raw_H = -np.sum(raw_freqs[raw_freqs > 0] * np.log(raw_freqs[raw_freqs > 0]))
    bpe_H = -np.sum(bpe_freqs[bpe_freqs > 0] * np.log(bpe_freqs[bpe_freqs > 0]))
    return raw_H, bpe_H


def fig5_cluster_merge_structure(merges, n_base=48, tokens_per_cluster=16, depth=3):
    """Show which cluster (hidden state) each merged token maps to."""
    n_clusters = n_base // tokens_per_cluster  # = 3 layers * (n/depth?)
    # Actually: cylinder graph has depth=3 layers, n=6 nodes per layer
    # tokens_per_cluster=16, total = 6*3 = 18 clusters?? No: d_vocab = depth * tokens_per_cluster = 48
    # Each cluster (hidden state) maps to tokens_per_cluster emission tokens
    # hidden states = n * depth = 18, but d_vocab = 48 = depth * tpc
    # Actually from config: n=6, depth=3, tokens_per_cluster=16
    # num_hidden_states = n * depth = 18
    # d_vocab = depth * tokens_per_cluster = 48
    # So the mapping is: tokens 0..15 -> layer 0, 16..31 -> layer 1, 32..47 -> layer 2

    # Count merges within-cluster vs cross-cluster
    within = 0
    cross_layer = 0
    within_layer_cross_cluster = 0  # not applicable here since layers = emission groups

    def token_to_layer(t):
        return t // tokens_per_cluster

    for _, hmm_toks in merges:
        layers = [token_to_layer(t) for t in hmm_toks]
        if len(set(layers)) == 1:
            within += 1
        else:
            cross_layer += 1

    print(f"\nMerge cluster structure:")
    print(f"  Within same emission layer: {within}")
    print(f"  Cross emission layers: {cross_layer}")

    # Heatmap: merge frequency by (layer_from, layer_to)
    layer_trans = np.zeros((depth, depth), dtype=int)
    for _, hmm_toks in merges:
        for i in range(len(hmm_toks) - 1):
            l_from = token_to_layer(hmm_toks[i])
            l_to = token_to_layer(hmm_toks[i + 1])
            layer_trans[l_from, l_to] += 1

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(layer_trans, cmap='Oranges', aspect='auto')
    ax.set_xlabel('To emission layer')
    ax.set_ylabel('From emission layer')
    ax.set_title('BPE Merges by Emission Layer')
    ax.set_xticks(range(depth))
    ax.set_yticks(range(depth))
    for i in range(depth):
        for j in range(depth):
            ax.text(j, i, str(layer_trans[i, j]),
                    ha='center', va='center', fontsize=12,
                    color='white' if layer_trans[i, j] > layer_trans.max() / 2 else 'black')
    plt.colorbar(im, ax=ax, label='Number of bigrams in merges')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "merge_layer_structure.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "merge_layer_structure.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved merge_layer_structure")

    return within, cross_layer, layer_trans


def fig6_entropy_rate_analysis(all_obs, tokenizer):
    """
    Compare empirical bigram entropy rates for raw vs BPE sequences.
    This measures how much predictable structure BPE captures.
    """
    subsample = all_obs[:2_000_000]

    # Raw bigram entropy rate
    n_base = int(all_obs.max()) + 1
    raw_bigram = np.zeros((n_base, n_base), dtype=np.float64)
    for i in range(len(subsample) - 1):
        raw_bigram[subsample[i], subsample[i + 1]] += 1

    # Normalize to conditional probabilities
    raw_row_sums = raw_bigram.sum(axis=1, keepdims=True)
    raw_row_sums[raw_row_sums == 0] = 1
    raw_cond = raw_bigram / raw_row_sums

    # Stationary distribution from bigram counts
    raw_marginal = raw_bigram.sum(axis=1)
    raw_marginal = raw_marginal / raw_marginal.sum()

    # Bigram entropy rate: H = -sum_i pi(i) sum_j P(j|i) log P(j|i)
    raw_H_rate = 0
    for i in range(n_base):
        for j in range(n_base):
            if raw_cond[i, j] > 0:
                raw_H_rate -= raw_marginal[i] * raw_cond[i, j] * np.log(raw_cond[i, j])

    # BPE bigram entropy rate
    bpe_ids = encode_sequence(tokenizer, subsample)
    vocab_size = tokenizer.get_vocab_size()
    bpe_bigram = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for i in range(len(bpe_ids) - 1):
        bpe_bigram[bpe_ids[i], bpe_ids[i + 1]] += 1

    bpe_row_sums = bpe_bigram.sum(axis=1, keepdims=True)
    bpe_row_sums[bpe_row_sums == 0] = 1
    bpe_cond = bpe_bigram / bpe_row_sums

    bpe_marginal = bpe_bigram.sum(axis=1)
    bpe_marginal = bpe_marginal / bpe_marginal.sum()

    bpe_H_rate = 0
    for i in range(vocab_size):
        for j in range(vocab_size):
            if bpe_cond[i, j] > 0:
                bpe_H_rate -= bpe_marginal[i] * bpe_cond[i, j] * np.log(bpe_cond[i, j])

    # Get compression ratio to convert BPE nats/token to nats/raw-token
    compression = len(subsample) / len(bpe_ids)
    bpe_H_rate_per_raw = bpe_H_rate / compression

    print(f"\nEntropy rate analysis (bigram approximation):")
    print(f"  Raw:  {raw_H_rate:.4f} nats/token")
    print(f"  BPE:  {bpe_H_rate:.4f} nats/BPE-token")
    print(f"  BPE (per raw token): {bpe_H_rate_per_raw:.4f} nats/raw-token")
    print(f"  Compression ratio: {compression:.3f}x")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ['C0', 'C1', 'C1']
    alphas = [1.0, 1.0, 0.5]
    labels = ['Raw\n(nats/token)', 'BPE\n(nats/BPE-token)', 'BPE\n(nats/raw-token)']
    values = [raw_H_rate, bpe_H_rate, bpe_H_rate_per_raw]
    bars = []
    for i, (lbl, val) in enumerate(zip(labels, values)):
        b = ax.bar(i, val, color=colors[i], alpha=alphas[i],
                   edgecolor='black', linewidth=0.5)
        bars.append(b[0])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Bigram entropy rate (nats)')
    ax.set_title('Bigram Entropy Rate: Raw vs BPE')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [raw_H_rate, bpe_H_rate, bpe_H_rate_per_raw]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_rate_comparison.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / "entropy_rate_comparison.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved entropy_rate_comparison")

    return raw_H_rate, bpe_H_rate, bpe_H_rate_per_raw, compression


def print_top_merges(merges, n=20):
    """Print first N merges with their HMM token sequences."""
    print(f"\nTop {n} BPE merges (in merge order):")
    for i, (bpe_id, hmm_toks) in enumerate(merges[:n]):
        print(f"  {bpe_id:4d}: [{', '.join(str(t) for t in hmm_toks)}]  (len={len(hmm_toks)})")


def main():
    print("=" * 60)
    print("BPE Tokenization Analysis for Cylinder Graph HMM")
    print("=" * 60)

    all_obs = load_data()

    # Train tokenizer at default vocab size
    print("\n--- Training BPE tokenizer (V=128) ---")
    tokenizer = train_tokenizer(all_obs, vocab_size=128)
    merges = get_merge_info(tokenizer)
    print_top_merges(merges)

    # Generate figures
    print("\n--- Figure 1: Merge co-occurrence ---")
    fig1_merge_cooccurrence(merges)

    print("\n--- Figure 2: Merge n-gram lengths ---")
    fig2_merge_lengths(merges)

    print("\n--- Figure 3: Compression vs vocab size ---")
    comp_data = fig3_compression_vs_vocab(all_obs)

    print("\n--- Figure 4: Token frequency distributions ---")
    raw_H, bpe_H = fig4_token_frequency(all_obs, tokenizer)
    print(f"  Raw unigram entropy: {raw_H:.4f} nats")
    print(f"  BPE unigram entropy: {bpe_H:.4f} nats")

    print("\n--- Figure 5: Merge layer structure ---")
    within, cross, layer_trans = fig5_cluster_merge_structure(merges)

    print("\n--- Figure 6: Entropy rate analysis ---")
    raw_H_rate, bpe_H_rate, bpe_H_per_raw, compression = fig6_entropy_rate_analysis(
        all_obs, tokenizer
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Raw vocabulary: 48 tokens")
    print(f"BPE vocabulary: {tokenizer.get_vocab_size()} tokens")
    print(f"Number of merges: {len(merges)}")
    print(f"Compression ratio (V=128): {comp_data.get(128, 'N/A'):.3f}x")
    print(f"Merge lengths: {Counter(len(t) for _, t in merges)}")
    print(f"Within-layer merges: {within}, cross-layer: {cross}")
    print(f"Raw bigram H rate: {raw_H_rate:.4f} nats/token")
    print(f"BPE bigram H rate: {bpe_H_rate:.4f} nats/BPE-token")
    print(f"BPE bigram H rate (per raw token): {bpe_H_per_raw:.4f} nats/raw-token")
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
