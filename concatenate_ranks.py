"""
Script to concatenate MPI-generated rank files into final dataset file(s).

Usage:
    python concatenate_ranks.py data/datasets/psl7/train
    python concatenate_ranks.py data/datasets/psl7/val --max-size 100
    python concatenate_ranks.py data/datasets/psl7/test --keep-ranks
"""

import torch
import os
import json
import argparse
import glob
from datetime import datetime


def get_rank_files(split_dir):
    """Find all observations_rank*.pt files in the directory."""
    pattern = os.path.join(split_dir, "observations_rank*.pt")
    rank_files = glob.glob(pattern)

    # Sort by rank number
    def extract_rank(filepath):
        filename = os.path.basename(filepath)
        rank_str = filename.replace("observations_rank", "").replace(".pt", "")
        return int(rank_str)

    rank_files.sort(key=extract_rank)
    return rank_files


def estimate_tensor_size_mb(tensor):
    """Estimate size of tensor in MB when saved to disk."""
    # Rough estimate: num_elements * bytes_per_element
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    size_bytes = num_elements * bytes_per_element
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def split_tensor_by_size(tensor, max_size_mb=200):
    """Split tensor into chunks where each chunk is <= max_size_mb."""
    total_size_mb = estimate_tensor_size_mb(tensor)

    if total_size_mb <= max_size_mb:
        # Small enough - return as single chunk
        return [tensor]

    # Calculate how many rows per chunk
    num_rows = tensor.shape[0]
    bytes_per_row = tensor[0:1].numel() * tensor.element_size()
    rows_per_chunk = int((max_size_mb * 1024 * 1024) / bytes_per_row)

    if rows_per_chunk < 1:
        raise ValueError(f"Single row is too large ({bytes_per_row / (1024*1024):.1f} MB) to fit in {max_size_mb} MB chunks")

    # Split into chunks
    chunks = []
    for start_idx in range(0, num_rows, rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk, num_rows)
        chunks.append(tensor[start_idx:end_idx])

    return chunks


def concatenate_rank_files(split_dir, max_size_mb=200, keep_ranks=False):
    """
    Concatenate all rank files in a directory.

    Args:
        split_dir: Directory containing observations_rank*.pt files
        max_size_mb: Maximum size per output file in MB
        keep_ranks: If True, keep intermediate rank files after concatenation
    """
    print("=" * 70)
    print("RANK FILE CONCATENATION")
    print("=" * 70)
    print(f"Directory: {split_dir}")
    print(f"Max file size: {max_size_mb} MB")

    # Find all rank files
    rank_files = get_rank_files(split_dir)

    if not rank_files:
        print(f"ERROR: No observations_rank*.pt files found in {split_dir}")
        return 1

    print(f"Found {len(rank_files)} rank files")

    # Load metadata if it exists
    meta_path = os.path.join(split_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded existing metadata: {metadata.get('num_samples', 'unknown')} samples expected")
    else:
        metadata = {}

    # Load and concatenate all rank files
    print("\nLoading rank files...")
    all_observations = []
    total_samples = 0

    for i, rank_file in enumerate(rank_files):
        print(f"  [{i+1}/{len(rank_files)}] Loading {os.path.basename(rank_file)}...", end=" ")
        rank_data = torch.load(rank_file)
        num_samples = rank_data.shape[0]
        print(f"{num_samples} samples ({estimate_tensor_size_mb(rank_data):.1f} MB)")
        all_observations.append(rank_data)
        total_samples += num_samples

    # Concatenate
    print(f"\nConcatenating {total_samples} total samples...")
    observations = torch.cat(all_observations, dim=0)
    print(f"Final shape: {observations.shape}")
    print(f"Data type: {observations.dtype}")

    total_size_mb = estimate_tensor_size_mb(observations)
    print(f"Total size: {total_size_mb:.1f} MB")

    # Split into chunks if needed
    chunks = split_tensor_by_size(observations, max_size_mb=max_size_mb)
    num_parts = len(chunks)

    print(f"\nSaving {num_parts} file(s)...")

    if num_parts == 1:
        # Single file
        output_path = os.path.join(split_dir, "observations.pt")
        print(f"  Saving {output_path} ({total_size_mb:.1f} MB)...")
        torch.save(observations, output_path)
        metadata["output_files"] = ["observations.pt"]
    else:
        # Multiple files
        output_files = []
        for i, chunk in enumerate(chunks):
            output_file = f"observations_part{i}.pt"
            output_path = os.path.join(split_dir, output_file)
            chunk_size_mb = estimate_tensor_size_mb(chunk)
            print(f"  Saving {output_file} ({chunk.shape[0]} samples, {chunk_size_mb:.1f} MB)...")
            torch.save(chunk, output_path)
            output_files.append(output_file)
        metadata["output_files"] = output_files

    # Update metadata
    metadata["concatenated"] = True
    metadata["concatenation_date"] = datetime.now().isoformat()
    metadata["num_samples"] = total_samples
    metadata["num_rank_files"] = len(rank_files)
    metadata["num_output_files"] = num_parts
    metadata["max_file_size_mb"] = max_size_mb

    print(f"\nUpdating metadata: {meta_path}")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Cleanup rank files if requested
    if not keep_ranks:
        print(f"\nCleaning up {len(rank_files)} intermediate rank files...")
        for rank_file in rank_files:
            os.remove(rank_file)
        print("Cleanup complete")
    else:
        print(f"\nKeeping {len(rank_files)} intermediate rank files (--keep-ranks specified)")

    print("\n" + "=" * 70)
    print("CONCATENATION COMPLETE")
    print("=" * 70)
    print(f"Output: {split_dir}/")
    if num_parts == 1:
        print(f"  observations.pt: {total_size_mb:.1f} MB")
    else:
        for i, chunk in enumerate(chunks):
            print(f"  observations_part{i}.pt: {estimate_tensor_size_mb(chunk):.1f} MB")
    print(f"Total samples: {total_samples}")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate MPI-generated rank files into final dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate rank files in train split
  python concatenate_ranks.py data/datasets/psl7/train

  # Use smaller max file size (100 MB per file)
  python concatenate_ranks.py data/datasets/psl7/val --max-size 100

  # Keep intermediate rank files after concatenation
  python concatenate_ranks.py data/datasets/psl7/test --keep-ranks
        """
    )

    parser.add_argument(
        'split_dir',
        type=str,
        help='Directory containing observations_rank*.pt files'
    )

    parser.add_argument(
        '--max-size',
        type=float,
        default=200,
        help='Maximum file size in MB (default: 200)'
    )

    parser.add_argument(
        '--keep-ranks',
        action='store_true',
        help='Keep intermediate rank files after concatenation'
    )

    args = parser.parse_args()

    # Validate directory exists
    if not os.path.exists(args.split_dir):
        print(f"ERROR: Directory not found: {args.split_dir}")
        return 1

    if not os.path.isdir(args.split_dir):
        print(f"ERROR: Not a directory: {args.split_dir}")
        return 1

    try:
        return concatenate_rank_files(
            args.split_dir,
            max_size_mb=args.max_size,
            keep_ranks=args.keep_ranks
        )
    except Exception as e:
        print(f"\nERROR during concatenation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
