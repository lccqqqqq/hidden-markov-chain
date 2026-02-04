"""
Script to concatenate MPI-generated rank files into final dataset file.

Usage:
    python concatenate_ranks.py data/datasets/psl7/train
    python concatenate_ranks.py data/datasets/psl7/test --keep-ranks
"""

import torch
import os
import json
import argparse
import glob
from datetime import datetime
from tqdm import tqdm


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


def concatenate_rank_files(split_dir, keep_ranks=False):
    """
    Concatenate all rank files in a directory.

    Args:
        split_dir: Directory containing observations_rank*.pt files
        keep_ranks: If True, keep intermediate rank files after concatenation
    """
    print("=" * 70)
    print("RANK FILE CONCATENATION")
    print("=" * 70)
    print(f"Directory: {split_dir}")

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

    for rank_file in tqdm(rank_files, desc="Loading rank files", unit="file"):
        rank_data = torch.load(rank_file)
        num_samples = rank_data.shape[0]
        all_observations.append(rank_data)
        total_samples += num_samples

    # Concatenate
    print(f"\nConcatenating {total_samples} total samples...")
    observations = torch.cat(all_observations, dim=0)
    print(f"Final shape: {observations.shape}")
    print(f"Data type: {observations.dtype}")

    total_size_mb = estimate_tensor_size_mb(observations)
    print(f"Total size: {total_size_mb:.1f} MB")

    # Save as single file
    output_path = os.path.join(split_dir, "observations.pt")
    print(f"\nSaving {os.path.basename(output_path)} ({total_size_mb:.1f} MB)...")
    with tqdm(total=1, desc="Saving file", unit="file") as pbar:
        torch.save(observations, output_path)
        pbar.update(1)

    # Update metadata
    metadata["concatenated"] = True
    metadata["concatenation_date"] = datetime.now().isoformat()
    metadata["num_samples"] = total_samples
    metadata["num_rank_files"] = len(rank_files)

    # Remove concatenation_required flag if it exists
    metadata.pop("concatenation_required", None)
    metadata.pop("concatenation_command", None)

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
    print(f"  observations.pt: {total_size_mb:.1f} MB")
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
            keep_ranks=args.keep_ranks
        )
    except Exception as e:
        print(f"\nERROR during concatenation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
