"""
Create BPE-tokenized datasets for forward and reversed HMM sequences.

Loads a trained BPE tokenizer and raw shards, encodes sequences,
chunks into fixed-length tensors, and saves train/test splits.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tokenizers import Tokenizer
from bpe_tokenizer import encode_sequence, decode_to_int_seq, int_seq_to_string


def round_trip_test(tokenizer, all_obs, n_tests=100, subseq_len=200):
    """Verify encode-then-decode round-trips exactly."""
    rng = np.random.RandomState(0)
    max_start = len(all_obs) - subseq_len
    for i in range(n_tests):
        start = rng.randint(0, max_start)
        subseq = all_obs[start:start + subseq_len]
        bpe_ids = encode_sequence(tokenizer, subseq)
        recovered = decode_to_int_seq(tokenizer, bpe_ids)
        assert np.array_equal(subseq, recovered), (
            f"Round-trip failed at test {i}: start={start}, "
            f"original[:10]={subseq[:10]}, recovered[:10]={recovered[:10]}"
        )
    print(f"Round-trip test PASSED ({n_tests} subsequences of length {subseq_len})")


def encode_and_chunk(tokenizer, raw_seq, seq_length, direction="forward"):
    """
    Encode a raw integer sequence with BPE and chunk into fixed-length sequences.

    Each output sequence has length seq_length+1 (for input/target split in training).

    Returns:
        sequences: int64 tensor of shape (N, seq_length+1)
        compression_ratio: raw_tokens / bpe_tokens
    """
    bpe_ids = encode_sequence(tokenizer, raw_seq)

    # Sanity checks
    vocab_size = tokenizer.get_vocab_size()
    assert bpe_ids.max() < vocab_size, f"Token ID {bpe_ids.max()} >= vocab_size {vocab_size}"
    assert bpe_ids.min() >= 0, f"Negative token ID: {bpe_ids.min()}"

    compression_ratio = len(raw_seq) / len(bpe_ids)
    print(f"  {direction}: {len(raw_seq)} raw tokens -> {len(bpe_ids)} BPE tokens "
          f"(compression ratio: {compression_ratio:.3f}x)")

    # Chunk into non-overlapping sequences of length seq_length+1
    chunk_len = seq_length + 1
    n_seqs = len(bpe_ids) // chunk_len
    bpe_ids = bpe_ids[:n_seqs * chunk_len]
    sequences = bpe_ids.reshape(n_seqs, chunk_len)

    return torch.from_numpy(sequences), compression_ratio


def save_dataset(sequences, output_dir, train_ratio=0.99, seed=42):
    """Shuffle and split sequences into train/test, then save."""
    output_dir = Path(output_dir)

    # Shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sequences))
    sequences = sequences[perm]

    # Split
    n_train = int(train_ratio * len(sequences))
    train_data = sequences[:n_train]
    test_data = sequences[n_train:]

    # Save
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    torch.save(train_data, output_dir / "train" / "observations.pt")
    torch.save(test_data, output_dir / "test" / "observations.pt")

    print(f"  Saved to {output_dir}: {len(train_data)} train, {len(test_data)} test sequences")
    return len(train_data), len(test_data)


def main():
    parser = argparse.ArgumentParser(description="Create BPE-tokenized datasets")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to trained BPE tokenizer JSON")
    parser.add_argument("--shards_dir", type=str, required=True,
                        help="Directory containing raw obs_*.npy shards")
    parser.add_argument("--seq_length", type=int, default=16,
                        help="Context length for training sequences")
    parser.add_argument("--output_base", type=str,
                        default="data/datasets",
                        help="Base directory for output datasets")
    parser.add_argument("--train_ratio", type=float, default=0.99)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer: vocab_size={vocab_size}")

    # Load raw shards
    shard_files = sorted(Path(args.shards_dir).glob("obs_*.npy"))
    print(f"Loading {len(shard_files)} shards...")
    all_obs = np.concatenate([np.load(f) for f in shard_files])
    print(f"Total raw tokens: {len(all_obs)}")

    # Round-trip sanity check
    round_trip_test(tokenizer, all_obs)

    # Print first 10 merges
    vocab = tokenizer.get_vocab()
    n_base = int(all_obs.max()) + 1
    id_to_token = {v: k for k, v in vocab.items()}
    print(f"\nFirst 10 BPE merges:")
    for idx in range(n_base, min(n_base + 10, vocab_size)):
        if idx in id_to_token:
            bpe_tok = id_to_token[idx]
            hmm_toks = [ord(c) - 0x21 for c in bpe_tok]
            print(f"  {idx}: [{' '.join(str(t) for t in hmm_toks)}]")

    # Forward BPE dataset
    print(f"\n--- Forward BPE ---")
    fwd_dir = Path(args.output_base) / "cylinder_graph_hmm_bpe"
    fwd_seqs, fwd_ratio = encode_and_chunk(tokenizer, all_obs, args.seq_length, "forward")
    n_train_fwd, n_test_fwd = save_dataset(fwd_seqs, fwd_dir, args.train_ratio)

    # Reversed BPE dataset: tokenize forward, then reverse BPE token order
    # (matches "Arrows of Time for LLMs" methodology — reverse at token level)
    print(f"\n--- Reversed BPE (token-level reversal) ---")
    rev_dir = Path(args.output_base) / "cylinder_graph_hmm_bpe_reversed"
    rev_seqs = fwd_seqs.flip(dims=[1])  # reverse each sequence's token order
    n_train_rev, n_test_rev = save_dataset(rev_seqs, rev_dir, args.train_ratio)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"BPE vocab size: {vocab_size}")
    print(f"Seq length: {args.seq_length}")
    print(f"Compression ratio: {fwd_ratio:.3f}x")
    print(f"Forward:  sequences={n_train_fwd + n_test_fwd} ({n_train_fwd} train / {n_test_fwd} test)")
    print(f"Reversed: sequences={n_train_rev + n_test_rev} ({n_train_rev} train / {n_test_rev} test)")
    print(f"(Same compression — reversal is at BPE token level)")
    print(f"{'='*60}")

    # Save metadata for each dataset
    for direction, out_dir, n_tr, n_te in [
        ("forward", fwd_dir, n_train_fwd, n_test_fwd),
        ("reversed_token_level", rev_dir, n_train_rev, n_test_rev),
    ]:
        meta = {
            "direction": direction,
            "reversal_method": "token_level" if "reversed" in direction else "none",
            "tokenizer_path": args.tokenizer_path,
            "vocab_size": vocab_size,
            "seq_length": args.seq_length,
            "compression_ratio": fwd_ratio,
            "train_size": n_tr,
            "test_size": n_te,
            "raw_tokens": len(all_obs),
        }
        with open(Path(out_dir) / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
