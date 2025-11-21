"""
Script to pre-generate datasets for HMM training.

Usage:
    python generate_dataset.py config/psl7.yaml
    python generate_dataset.py config/test_config.yaml --chunk_size 5000

    # With MPI (using 4 processes)
    mpirun -n 4 python generate_dataset.py config/psl7.yaml --use-mpi
"""

import torch
import yaml
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM
import random

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None


def generate_dataset(config_path, chunk_size=10000, force=False, split='all', use_mpi=False):
    """
    Generate and save a pre-computed dataset based on config file.

    Args:
        config_path: Path to YAML config file
        chunk_size: Number of samples to generate at a time (to avoid memory issues)
        force: If True, overwrite existing dataset
        split: Which split to generate - 'train', 'val', 'test', or 'all'
        use_mpi: If True, use MPI to parallelize chunk generation across ranks
    """
    # Initialize MPI if requested
    if use_mpi:
        if not MPI_AVAILABLE:
            raise RuntimeError("MPI requested but mpi4py is not installed. Install with: pip install mpi4py")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1
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

    if rank == 0:
        print("=" * 70)
        print("DATASET GENERATION")
        print("=" * 70)
        print(f"Config: {config_path}")
        print(f"Process: {process_name}")
        print(f"Sequence length: {seq_length} (n_ctx={model_config['n_ctx']} + 1)")
        print(f"Vocab size: {vocab_size}")
        print(f"Splits to generate: {', '.join(splits_to_generate)}")
        if use_mpi:
            print(f"MPI enabled: {size} ranks")
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
    if rank == 0:
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

    if rank == 0:
        print(f"Process initialized: {process.__class__.__name__}")
        print(f"  Hidden states: {process.num_hidden_states}")
        print(f"  Vocab size: {process.d_vocab}")

    # Verify vocab size matches config
    if process.d_vocab != vocab_size:
        if rank == 0:
            print(f"\nWARNING: Process vocab size ({process.d_vocab}) doesn't match "
                  f"config vocab size ({vocab_size})")
            response = input("Continue anyway? [y/N]: ")
            should_continue = response.lower() == 'y'
        else:
            should_continue = None

        if use_mpi:
            should_continue = comm.bcast(should_continue, root=0)

        if not should_continue:
            if rank == 0:
                print("Aborting.")
            return

    # Generate each split
    for split_name in splits_to_generate:
        total_samples = split_sizes[split_name]

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Generating {split_name.upper()} split")
            print(f"{'='*70}")

        # Setup split-specific output directory
        split_dir = os.path.join(base_output_dir, split_name)
        if rank == 0:
            os.makedirs(split_dir, exist_ok=True)

        obs_path = os.path.join(split_dir, "observations.pt")
        meta_path = os.path.join(split_dir, "meta.json")

        # Check if split already exists (only rank 0 checks)
        skip_split = False
        if rank == 0:
            if os.path.exists(obs_path) and not force:
                print(f"Split already exists at: {obs_path}")
                print("Skipping... (use --force to overwrite)")
                skip_split = True

        if use_mpi:
            skip_split = comm.bcast(skip_split, root=0)

        if skip_split:
            continue

        # Generate data in chunks
        if rank == 0:
            print(f"Generating {total_samples:,} samples in chunks of {chunk_size:,}...")

        num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

        # Distribute chunks across MPI ranks
        if use_mpi:
            # Each rank processes a subset of chunks
            chunks_per_rank = num_chunks // size
            remainder = num_chunks % size

            # Rank i processes chunks from start_chunk to end_chunk
            if rank < remainder:
                start_chunk = rank * (chunks_per_rank + 1)
                end_chunk = start_chunk + chunks_per_rank + 1
            else:
                start_chunk = rank * chunks_per_rank + remainder
                end_chunk = start_chunk + chunks_per_rank

            my_chunks = list(range(start_chunk, end_chunk))
        else:
            my_chunks = list(range(num_chunks))

        # Generate chunks assigned to this rank
        my_observations = []

        # Print work assignment for this rank
        if use_mpi and len(my_chunks) > 0:
            print(f"Rank {rank}: Processing {len(my_chunks)} chunks (indices {my_chunks[0]}-{my_chunks[-1]}) for {split_name} split...")
        elif not use_mpi:
            # Only show progress bar when not using MPI (single process, likely interactive)
            my_chunks = tqdm(my_chunks, desc=f"Generating {split_name} chunks")

        import time
        start_time = time.time()

        for i, chunk_idx in enumerate(my_chunks):
            # Calculate how many samples for this chunk
            samples_remaining = total_samples - (chunk_idx * chunk_size)
            current_chunk_size = min(chunk_size, samples_remaining)

            # Generate chunk
            chunk_data = process.generate_data(
                batch_size=current_chunk_size,
                length=seq_length,
                init_state=-1,  # Use -1 to signal random init per sequence (avoids stationary distribution computation)
                use_tqdm=False  # We already have an outer progress bar
            )

            my_observations.append(chunk_data)

            # Print periodic progress updates when using MPI (every 10 chunks)
            if use_mpi and len(my_chunks) > 0 and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                progress_pct = 100 * (i + 1) / len(my_chunks)
                print(f"Rank {rank}: {i + 1}/{len(my_chunks)} chunks ({progress_pct:.0f}%) - {elapsed:.1f}s elapsed")

        # Print completion status for this rank
        if use_mpi and len(my_chunks) > 0:
            elapsed_time = time.time() - start_time
            print(f"Rank {rank}: Completed {len(my_chunks)} chunks in {elapsed_time:.1f} seconds")

        # Concatenate local chunks
        if len(my_observations) > 0:
            local_observations = torch.cat(my_observations, dim=0)
        else:
            # Create empty tensor with correct shape
            local_observations = torch.empty((0, seq_length), dtype=torch.long)

        # Save per-rank files when using MPI, then concatenate on rank 0
        if use_mpi:
            # Each rank saves its own file
            rank_file = os.path.join(split_dir, f"observations_rank{rank}.pt")
            if rank == 0:
                print("Each rank saving its data...")

            print(f"Rank {rank}: Saving {local_observations.shape[0]} samples to {rank_file}")
            torch.save(local_observations, rank_file)

            # Synchronize - wait for all ranks to finish saving
            comm.Barrier()

            # Rank 0 concatenates all files
            if rank == 0:
                print("Rank 0: Concatenating all rank files...")
                all_observations = []

                for r in range(size):
                    rank_file = os.path.join(split_dir, f"observations_rank{r}.pt")
                    print(f"Rank 0: Loading {rank_file}...")
                    rank_data = torch.load(rank_file)
                    all_observations.append(rank_data)

                observations = torch.cat(all_observations, dim=0)

                # Clean up intermediate files
                print("Rank 0: Cleaning up intermediate rank files...")
                for r in range(size):
                    rank_file = os.path.join(split_dir, f"observations_rank{r}.pt")
                    os.remove(rank_file)
            else:
                observations = None
        else:
            observations = local_observations

        # Only rank 0 saves the data
        if rank == 0:
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
                "size_mb": size_mb,
                "generated_with_mpi": use_mpi,
                "mpi_ranks": size if use_mpi else 1
            }

            if split_name == 'train':
                metadata["batch_size"] = batch_size
                metadata["num_epochs"] = num_epochs

            print(f"Saving metadata to: {meta_path}")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"{split_name.upper()} split complete: {size_mb:.1f} MB")

    if rank == 0:
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

  # Use MPI with 4 processes to parallelize chunk generation
  mpirun -n 4 python generate_dataset.py config/psl7.yaml --use-mpi

  # MPI with custom chunk size
  mpirun -n 8 python generate_dataset.py config/psl7.yaml --use-mpi --chunk_size 5000
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

    parser.add_argument(
        '--use-mpi',
        action='store_true',
        help='Use MPI to parallelize chunk generation across multiple ranks'
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        generate_dataset(args.config, chunk_size=args.chunk_size, force=args.force, split=args.split, use_mpi=args.use_mpi)
        return 0
    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
