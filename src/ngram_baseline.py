"""
N-gram baseline cross-entropy evaluator for CylinderGraph HMM datasets.

Computes count-based n-gram cross-entropy (in nats) for n=1..max_n,
providing a "no learned representation" reference point comparable
to transformer next-token prediction loss.

Usage:
    python src/ngram_baseline.py
    python src/ngram_baseline.py --max_n 8 --smoothing_k 0.1
"""

import argparse
import json
import os
import sys
import math
from collections import defaultdict, Counter

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils import create_process_from_dict


def parse_args():
    parser = argparse.ArgumentParser(description="N-gram baseline for HMM datasets")
    parser.add_argument("--config", type=str, default="base_config.yaml",
                        help="Path to config YAML (default: base_config.yaml)")
    parser.add_argument("--max_n", type=int, default=6,
                        help="Maximum n-gram order to evaluate (default: 6)")
    parser.add_argument("--smoothing_k", type=float, default=1.0,
                        help="Laplace smoothing parameter (default: 1.0)")
    return parser.parse_args()


def build_ngram_counts(sequences, n, vocab_size):
    """
    Build n-gram count table from training sequences.

    Args:
        sequences: tensor [num_seq, seq_len], dtype int64
        n: n-gram order (1=unigram, 2=bigram, etc.)
        vocab_size: number of distinct tokens

    Returns:
        dict mapping (n-1)-tuple context -> Counter of next tokens
    """
    counts = defaultdict(Counter)
    for seq in sequences:
        for t in range(n - 1, len(seq)):
            context = tuple(seq[t - n + 1:t].tolist())  # length (n-1)
            token = seq[t].item()
            counts[context][token] += 1
    return counts


def evaluate_ngram(sequences, ngram_counts_by_order, n, vocab_size, k=1.0):
    """
    Evaluate n-gram model cross-entropy on sequences with backoff and Laplace smoothing.

    For each test position t in 1..seq_len-1 (matching transformer targets):
      - Try context of length min(n-1, t)
      - If context not found, back off to shorter context
      - Compute smoothed probability: (count + k) / (total + k * vocab_size)

    Args:
        sequences: tensor [num_seq, seq_len]
        ngram_counts_by_order: dict {order: counts_dict} for orders 1..n
        n: maximum n-gram order to use
        vocab_size: number of distinct tokens
        k: Laplace smoothing parameter

    Returns:
        average cross-entropy in nats
    """
    total_log_prob = 0.0
    total_tokens = 0

    for seq in sequences:
        seq_len = len(seq)
        # Evaluate positions 1..seq_len-1 (next-token prediction targets)
        for t in range(1, seq_len):
            token = seq[t].item()
            # Try from highest order down to unigram
            max_order = min(n, t + 1)  # can't use more context than available
            found = False
            for order in range(max_order, 0, -1):
                context = tuple(seq[t - order + 1:t].tolist())  # length (order-1)
                counts = ngram_counts_by_order[order]
                if context in counts:
                    token_count = counts[context][token]
                    total_count = sum(counts[context].values())
                    prob = (token_count + k) / (total_count + k * vocab_size)
                    total_log_prob += math.log(prob)
                    found = True
                    break
            if not found:
                # Fallback: uniform distribution
                total_log_prob += math.log(1.0 / vocab_size)

            total_tokens += 1

    return -total_log_prob / total_tokens


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    vocab_size = cfg["model"]["vocab_size"]
    data_dir = cfg.get("data", {}).get("data_dir", cfg["data_generator"]["save_dir"])

    print(f"N-gram Baseline Results (vocab_size={vocab_size}, smoothing_k={args.smoothing_k})")
    print("=" * 65)

    # Load data
    train_path = os.path.join(data_dir, "train", "observations.pt")
    test_path = os.path.join(data_dir, "test", "observations.pt")
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    print()

    # Build counts for all orders
    print("Building n-gram counts...")
    ngram_counts = {}
    for n in range(1, args.max_n + 1):
        ngram_counts[n] = build_ngram_counts(train_data, n, vocab_size)
        print(f"  {n}-gram: {len(ngram_counts[n])} contexts")
    print()

    # Evaluate each order
    results = []
    print(f"  {'n':>2} | {'train CE (nats)':>15} | {'test CE (nats)':>14} | {'# contexts':>12}")
    print(f"  {'-'*2}-+-{'-'*15}-+-{'-'*14}-+-{'-'*12}")
    for n in range(1, args.max_n + 1):
        train_ce = evaluate_ngram(train_data, ngram_counts, n, vocab_size, args.smoothing_k)
        test_ce = evaluate_ngram(test_data, ngram_counts, n, vocab_size, args.smoothing_k)
        num_contexts = len(ngram_counts[n])
        print(f"  {n:>2} | {train_ce:>15.4f} | {test_ce:>14.4f} | {num_contexts:>12}")
        results.append({
            "n": n,
            "train_ce": train_ce,
            "test_ce": test_ce,
            "num_contexts": num_contexts,
        })

    print(f"  {'-'*2}-+-{'-'*15}-+-{'-'*14}-+-{'-'*12}")

    # HMM theoretical entropy rate
    proc = create_process_from_dict(cfg["data_generator"]["process"])
    h_rate = proc.entropy_rate_theory_estimate()
    print(f"  HMM entropy rate (lower bound): {h_rate:.4f} nats")

    # Save results
    os.makedirs("out", exist_ok=True)
    output = {
        "vocab_size": vocab_size,
        "smoothing_k": args.smoothing_k,
        "max_n": args.max_n,
        "hmm_entropy_rate": h_rate,
        "results": results,
    }
    out_path = "out/ngram_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
