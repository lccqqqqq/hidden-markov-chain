"""
Train a BPE tokenizer on raw HMM observation sequences.

Maps each HMM token (0–47) to a unique ASCII character, then uses
HuggingFace `tokenizers` library for standard BPE training.
"""

import argparse
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# Mapping: HMM token i <-> chr(0x21 + i), i.e. '!' through 'P' for 0–47
CHAR_OFFSET = 0x21


def int_seq_to_string(seq):
    """Convert an integer sequence (array-like) to a character string."""
    return ''.join(chr(CHAR_OFFSET + int(t)) for t in seq)


def string_to_int_seq(s):
    """Convert a character string back to an integer numpy array."""
    return np.array([ord(c) - CHAR_OFFSET for c in s], dtype=np.int64)


def train_bpe(shards_dir, vocab_size=128, save_path=None):
    """
    Train a BPE tokenizer on raw .npy shard files.

    Args:
        shards_dir: Directory containing obs_*.npy shard files
        vocab_size: Target vocabulary size (base chars + merges)
        save_path: Where to save the trained tokenizer JSON

    Returns:
        Trained Tokenizer object
    """
    # Load and concatenate all shards
    shard_files = sorted(Path(shards_dir).glob("obs_*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"No obs_*.npy files found in {shards_dir}")

    print(f"Loading {len(shard_files)} shards...")
    all_obs = np.concatenate([np.load(f) for f in shard_files])
    print(f"Total tokens: {len(all_obs)}, vocab range: [{all_obs.min()}, {all_obs.max()}]")

    # Convert to string
    full_string = int_seq_to_string(all_obs)

    # Split into ~10k-char chunks for BPE training
    chunk_size = 10_000
    chunks = [full_string[i:i+chunk_size] for i in range(0, len(full_string), chunk_size)]
    print(f"Training BPE on {len(chunks)} chunks ({chunk_size} chars each)")

    # Build base alphabet from observed characters
    n_base = int(all_obs.max()) + 1
    base_alphabet = [chr(CHAR_OFFSET + i) for i in range(n_base)]
    n_merges = vocab_size - n_base

    # Initialize BPE tokenizer with no pre-tokenization (treat each chunk as one word)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
    tokenizer.decoder = decoders.Fuse()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=base_alphabet,
        min_frequency=2,
    )

    # Train from iterator
    tokenizer.train_from_iterator(chunks, trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Trained BPE: {n_base} base chars + {actual_vocab - n_base} merges = {actual_vocab} total vocab")

    # Print merge table
    print_merge_table(tokenizer, n_base)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")

    return tokenizer


def encode_sequence(tokenizer, int_seq):
    """Encode an integer HMM sequence to BPE token IDs."""
    s = int_seq_to_string(int_seq)
    encoding = tokenizer.encode(s)
    return np.array(encoding.ids, dtype=np.int64)


def decode_to_int_seq(tokenizer, bpe_ids):
    """Decode BPE token IDs back to HMM integer sequence."""
    s = tokenizer.decode(list(bpe_ids))
    return string_to_int_seq(s)


def print_merge_table(tokenizer, n_base=48):
    """Pretty-print the BPE merge table showing HMM token n-grams."""
    vocab = tokenizer.get_vocab()
    # Sort by ID to get merge order
    id_to_token = {v: k for k, v in vocab.items()}

    print("\n=== BPE Merge Table ===")
    print(f"{'ID':>5} | {'BPE Token':>15} | HMM tokens")
    print("-" * 50)

    for idx in range(n_base, tokenizer.get_vocab_size()):
        if idx not in id_to_token:
            continue
        bpe_token = id_to_token[idx]
        hmm_tokens = [ord(c) - CHAR_OFFSET for c in bpe_token]
        hmm_str = " ".join(str(t) for t in hmm_tokens)
        print(f"{idx:5d} | {repr(bpe_token):>15} | [{hmm_str}]")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on HMM sequences")
    parser.add_argument("--shards_dir", type=str, required=True,
                        help="Directory containing obs_*.npy shards")
    parser.add_argument("--vocab_size", type=int, default=128,
                        help="Target BPE vocabulary size")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save tokenizer JSON")
    args = parser.parse_args()

    train_bpe(args.shards_dir, args.vocab_size, args.save_path)
