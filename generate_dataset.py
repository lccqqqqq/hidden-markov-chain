"""
Script to pre-generate datasets for HMM training.

Usage:
    python generate_dataset.py config/psl7.yaml
    python generate_dataset.py config/test_config.yaml --chunk_size 5000
"""

import torch
import yaml
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM


def generate_dataset(config_path, chunk_size=10000, force=False, split='all'):
    """
    Generate and save a pre-computed dataset based on config file.

    Args:
        config_path: Path to YAML config file
        chunk_size: Number of samples to generate at a time (to avoid memory issues)
        force: If True, overwrite existing dataset
        split: Which split to generate - 'train', 'val', 'test', or 'all'
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    train_config = config['train']
    model_config = config['model']
    dataset_config = config.get('dataset', {})

    batch_size = train_config['batch_size']
    num_epochs = train_config['num_epochs']
    process_name = train_config['process']
    seq_length = model_config['n_ctx'] + 1  # +1 for next token prediction
    vocab_size = model_config['vocab_size']

    # Calculate samples for each split
    train_samples = batch_size * num_epochs
    val_samples = dataset_config.get('val_samples', 10000)
    test_samples = dataset_config.get('test_samples', 10000)

    # Determine which splits to generate
    if split == 'all':
        splits_to_generate = ['train', 'val', 'test']
    else:
        splits_to_generate = [split]

    # Map split names to sample counts
    split_sizes = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    print("=" * 70)
    print("DATASET GENERATION")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Process: {process_name}")
    print(f"Sequence length: {seq_length} (n_ctx={model_config['n_ctx']} + 1)")
    print(f"Vocab size: {vocab_size}")
    print(f"Splits to generate: {', '.join(splits_to_generate)}")
    print(f"\nSplit sizes:")
    for split_name in splits_to_generate:
        samples = split_sizes[split_name]
        bytes_size = samples * seq_length * 8 / (1024 * 1024)
        print(f"  {split_name}: {samples:,} samples (~{bytes_size:.1f} MB)")
    print(f"Chunk size: {chunk_size:,}")
    print("=" * 70)

    # Setup base output directory
    base_output_dir = os.path.join("data", "datasets", process_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Initialize HMM process
    print("\nInitializing HMM process...")
    if process_name == "rrxor":
        process = RRXOR()
    elif process_name == "z1r":
        process = Z1R()
    elif process_name == "mess3":
        process = Mess3Proc()
    elif process_name == "psl7":
        process = PSL7HMM()
    else:
        raise ValueError(f"Unknown process: {process_name}")

    print(f"Process initialized: {process.__class__.__name__}")
    print(f"  Hidden states: {process.num_hidden_states}")
    print(f"  Vocab size: {process.d_vocab}")

    # Verify vocab size matches config
    if process.d_vocab != vocab_size:
        print(f"\nWARNING: Process vocab size ({process.d_vocab}) doesn't match "
              f"config vocab size ({vocab_size})")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborting.")
            return

    # Generate each split
    for split_name in splits_to_generate:
        total_samples = split_sizes[split_name]

        print(f"\n{'='*70}")
        print(f"Generating {split_name.upper()} split")
        print(f"{'='*70}")

        # Setup split-specific output directory
        split_dir = os.path.join(base_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        obs_path = os.path.join(split_dir, "observations.pt")
        meta_path = os.path.join(split_dir, "meta.json")

        # Check if split already exists
        if os.path.exists(obs_path) and not force:
            print(f"Split already exists at: {obs_path}")
            print("Skipping... (use --force to overwrite)")
            continue

        # Generate data in chunks
        print(f"Generating {total_samples:,} samples in chunks of {chunk_size:,}...")

        all_observations = []
        num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

        for chunk_idx in tqdm(range(num_chunks), desc=f"{split_name} chunks"):
            # Calculate how many samples for this chunk
            samples_remaining = total_samples - (chunk_idx * chunk_size)
            current_chunk_size = min(chunk_size, samples_remaining)

            # Generate chunk
            chunk_data = process.generate_data(
                batch_size=current_chunk_size,
                length=seq_length,
                use_tqdm=False  # We already have an outer progress bar
            )

            all_observations.append(chunk_data)

        # Concatenate all chunks
        print("Concatenating chunks...")
        observations = torch.cat(all_observations, dim=0)

        print(f"Final shape: {observations.shape}")
        print(f"Data type: {observations.dtype}")

        # Save observations
        bytes_per_sample = seq_length * 8  # int64 = 8 bytes
        size_mb = (total_samples * bytes_per_sample) / (1024 * 1024)

        print(f"Saving observations to: {obs_path}")
        torch.save(observations, obs_path)

        # Save metadata
        metadata = {
            "split": split_name,
            "process": process_name,
            "num_samples": total_samples,
            "seq_length": seq_length,
            "vocab_size": vocab_size,
            "n_ctx": model_config['n_ctx'],
            "num_hidden_states": process.num_hidden_states,
            "created": datetime.now().isoformat(),
            "config_file": config_path,
            "storage_format": "torch",
            "data_type": str(observations.dtype),
            "size_mb": size_mb
        }

        if split_name == 'train':
            metadata["batch_size"] = batch_size
            metadata["num_epochs"] = num_epochs

        print(f"Saving metadata to: {meta_path}")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"{split_name.upper()} split complete: {size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Dataset saved to: {base_output_dir}/")
    for split_name in splits_to_generate:
        size_mb = split_sizes[split_name] * seq_length * 8 / (1024 * 1024)
        print(f"  {split_name}/: {size_mb:.1f} MB")
    print("\nTo use this dataset, update your config file:")
    print("```yaml")
    print("dataset:")
    print("  mode: \"precomputed\"")
    print(f"  path: \"{base_output_dir}\"")
    print("```")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate datasets for HMM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all splits (train, val, test)
  python generate_dataset.py config/psl7.yaml

  # Generate only training split
  python generate_dataset.py config/test_config.yaml --split train

  # Generate with custom chunk size
  python generate_dataset.py config/test_config.yaml --chunk_size 5000

  # Force overwrite existing datasets
  python generate_dataset.py config/z1r.yaml --force
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=10000,
        help='Number of samples to generate per chunk (default: 10000)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing dataset if it exists'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to generate: train, val, test, or all (default: all)'
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        generate_dataset(args.config, chunk_size=args.chunk_size, force=args.force, split=args.split)
        return 0
    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
