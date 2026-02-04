"""
Multi-checkpoint processing with MPI support for HMM models.

This script processes multiple model checkpoints from a training run, optionally using
MPI for parallelization across multiple CPU cores sharing a single GPU.

Usage:
    # Single-rank processing
    python pca_multi_checkpoint.py --run-dir records/20251207_232030

    # Multi-rank with MPI (4 processes)
    mpirun -n 4 python pca_multi_checkpoint.py --run-dir records/20251207_232030 --use-mpi
"""

from pca import (
    load_model_from_weights,
    extract_residual_stream,
    pca_layer_wise_actvs,
    get_device,
)
from hmm import Mess3Proc, RRXOR, Z1R, PSL7HMM
import yaml
import torch
import os
import glob
import re
import pandas as pd
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import sys

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None


def discover_checkpoints(run_dir: str, pattern: str = "checkpoint_step_*.pt", include_final: bool = True) -> list[dict]:
    """
    Discover all checkpoint files in run_dir/checkpoints/ and optionally the final model.

    If include_final=True, also adds final_model_*.pt from run_dir/ (if exists).
    Final model is added at the end with step=999999999 for sorting purposes.

    Args:
        run_dir: Path to training run directory
        pattern: Glob pattern for checkpoint files
        include_final: Whether to include final model file

    Returns:
        List of dicts with keys: 'path', 'step', 'filename', 'name', 'is_final'
        Sorted by step number ascending
    """
    checkpoints = []

    # Discover checkpoints in checkpoints/ subdirectory
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        files = glob.glob(os.path.join(checkpoint_dir, pattern))

        for f in files:
            # Extract step number from filename
            # checkpoint_step_80000.pt -> 80000
            filename = os.path.basename(f)
            match = re.search(r'checkpoint_step_(\d+)\.pt', filename)
            if match:
                step = int(match.group(1))
                checkpoints.append({
                    'path': f,
                    'step': step,
                    'filename': filename,
                    'name': f'checkpoint_step_{step}',
                    'is_final': False
                })

    # Discover final model in run_dir/
    if include_final:
        final_files = glob.glob(os.path.join(run_dir, "final_model_*.pt"))
        if final_files:
            if len(final_files) > 1:
                print(f"Warning: Multiple final models found, using {final_files[0]}")

            final_path = final_files[0]
            filename = os.path.basename(final_path)
            checkpoints.append({
                'path': final_path,
                'step': 999999999,  # Use high number for sorting
                'filename': filename,
                'name': filename.replace('.pt', ''),
                'is_final': True
            })
        else:
            print(f"Warning: No final model found in {run_dir}")

    # Sort by step number
    checkpoints.sort(key=lambda x: x['step'])

    return checkpoints


def validate_cache_metadata(cache_dir: str, batch_size: int, seq_length: int) -> bool:
    """
    Validate that cache metadata matches current parameters.

    Args:
        cache_dir: Path to cache directory
        batch_size: Expected batch size
        seq_length: Expected sequence length

    Returns:
        True if cache is valid, False otherwise
    """
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")

    if not os.path.exists(metadata_path):
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if parameters match
        if (metadata.get('batch_size') != batch_size or
            metadata.get('seq_length') != seq_length):
            return False

        # Check if cache files exist
        obs_path = os.path.join(cache_dir, "observations.pt")
        belief_path = os.path.join(cache_dir, "belief_states.pt")

        if not (os.path.exists(obs_path) and os.path.exists(belief_path)):
            return False

        return True
    except Exception as e:
        print(f"Error validating cache metadata: {e}")
        return False


def get_or_create_cache(
    cache_dir: str,
    process,
    batch_size: int,
    seq_length: int,
    force_regenerate: bool,
    rank: int,
    comm,
) -> tuple:
    """
    Load or generate cached observations and belief states.

    Flow:
    1. Rank 0 checks if cache exists and is valid
    2. If not exists or force_regenerate:
       - Rank 0 generates observations using process.generate_data()
       - Rank 0 computes belief_states using process.mixed_state_presentation()
       - Rank 0 saves to cache_dir/observations.pt and belief_states.pt
       - Rank 0 saves metadata to cache_dir/cache_metadata.json
    3. comm.Barrier() - wait for rank 0 to complete
    4. All ranks load from cache files

    Args:
        cache_dir: Path to cache directory
        process: HMM process instance
        batch_size: Number of observation sequences
        seq_length: Length of each sequence
        force_regenerate: Force regeneration even if cache exists
        rank: MPI rank
        comm: MPI communicator

    Returns:
        (observations, belief_states) tensors
    """
    obs_path = os.path.join(cache_dir, "observations.pt")
    belief_path = os.path.join(cache_dir, "belief_states.pt")
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")

    # Rank 0 generates cache if needed
    if rank == 0:
        cache_valid = not force_regenerate and validate_cache_metadata(cache_dir, batch_size, seq_length)

        if not cache_valid:
            print(f"Rank 0: Generating observations cache...")
            os.makedirs(cache_dir, exist_ok=True)

            # Generate observations
            batch_obs = process.generate_data(
                batch_size=batch_size,
                length=seq_length,
                use_tqdm=True
            )

            # Compute belief states
            print(f"Rank 0: Computing belief states...")
            belief_states = process.mixed_state_presentation(batch_obs)

            # Save to cache
            print(f"Rank 0: Saving cache...")
            torch.save(batch_obs, obs_path)
            torch.save(torch.tensor(belief_states), belief_path)

            # Save metadata
            metadata = {
                'batch_size': batch_size,
                'seq_length': seq_length,
                'process': process.__class__.__name__,
                'd_vocab': belief_states.shape[-1],
                'created': datetime.now().isoformat(),
                'observations_shape': list(batch_obs.shape),
                'belief_states_shape': list(belief_states.shape)
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Rank 0: Cache created successfully")
        else:
            print(f"Rank 0: Using existing cache")

    # Synchronize - wait for rank 0 to finish
    if comm is not None:
        comm.Barrier()

    # All ranks load from cache
    if rank == 0:
        print(f"Rank 0: Loading from cache...")

    observations = torch.load(obs_path)
    belief_states = torch.load(belief_path)

    return observations, belief_states


def distribute_checkpoints(checkpoints: list[dict], rank: int, size: int) -> list[dict]:
    """
    Distribute checkpoints across MPI ranks using round-robin.

    Example with 4 ranks, 10 checkpoints:
    - Rank 0: [0, 4, 8]
    - Rank 1: [1, 5, 9]
    - Rank 2: [2, 6]
    - Rank 3: [3, 7]

    Args:
        checkpoints: List of checkpoint metadata dicts
        rank: MPI rank
        size: Total number of MPI ranks

    Returns:
        Subset of checkpoints assigned to this rank
    """
    my_checkpoints = []
    for i, ckpt in enumerate(checkpoints):
        if i % size == rank:
            my_checkpoints.append(ckpt)
    return my_checkpoints


def get_cached_activations(cache_dir: str, checkpoint_name: str):
    """
    Load cached activations for a checkpoint if they exist.

    Args:
        cache_dir: Path to cache directory
        checkpoint_name: Checkpoint identifier

    Returns:
        Activations tensor or None if not cached
    """
    activations_dir = os.path.join(cache_dir, "activations")
    activ_path = os.path.join(activations_dir, f"{checkpoint_name}.pt")

    if os.path.exists(activ_path):
        try:
            return torch.load(activ_path)
        except Exception as e:
            print(f"Warning: Failed to load cached activations from {activ_path}: {e}")
            return None
    return None


def cache_activations(cache_dir: str, checkpoint_name: str, activations):
    """
    Save activations to cache.

    Args:
        cache_dir: Path to cache directory
        checkpoint_name: Checkpoint identifier
        activations: Activations tensor to cache
    """
    activations_dir = os.path.join(cache_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    activ_path = os.path.join(activations_dir, f"{checkpoint_name}.pt")
    torch.save(activations, activ_path)


def process_single_checkpoint(
    checkpoint_info: dict,
    observations,
    belief_states,
    run_dir: str,
    results_dir: str,
    cache_dir: str,
    save_activations: bool,
    save_regression_data: bool,
    rank: int,
) -> dict:
    """
    Process a single checkpoint or final model.

    Flow:
    1. Load model from checkpoint using load_model_from_weights()
       - For checkpoints: load from checkpoint_info['path']
       - For final model: pass checkpoint_info['filename'] as file_name parameter
    2. Check activation cache:
       - If exists: Load from cache_dir/activations/{checkpoint_name}.pt
       - If not: Run extract_residual_stream() and cache if save_activations=True
    3. Run pca_layer_wise_actvs() for regression analysis
    4. Save results to results_dir/{checkpoint_name}_metrics.csv
    5. Return metadata dict with checkpoint info and metrics summary

    Args:
        checkpoint_info: Checkpoint metadata dict
        observations: Observation sequences tensor
        belief_states: Belief states tensor
        run_dir: Path to training run directory
        results_dir: Path to results directory
        cache_dir: Path to cache directory
        save_activations: Whether to cache activations
        save_regression_data: Whether to save full regression data
        rank: MPI rank

    Returns:
        Metadata dict with checkpoint info and metrics
    """
    checkpoint_name = checkpoint_info['name']
    is_final = checkpoint_info.get('is_final', False)

    label = "final model" if is_final else f"step {checkpoint_info['step']}"
    print(f"Rank {rank}: Processing {label}...")

    try:
        # Load model
        if is_final:
            # For final model, pass filename directly
            model = load_model_from_weights(
                model_dir=run_dir,
                file_name=checkpoint_info['filename']
            )
        else:
            # For checkpoint, use full path
            # Extract just the relative path from run_dir
            rel_path = os.path.relpath(checkpoint_info['path'], run_dir)
            model = load_model_from_weights(
                model_dir=run_dir,
                file_name=rel_path
            )

        # Check activation cache
        actvs = get_cached_activations(cache_dir, checkpoint_name)

        if actvs is None:
            # Run model inference
            actvs = extract_residual_stream(
                model=model,
                sequences=observations,
            )

            # Cache if requested
            if save_activations:
                cache_activations(cache_dir, checkpoint_name, actvs)
                print(f"Rank {rank}: Cached activations for {checkpoint_name}")
        else:
            print(f"Rank {rank}: Using cached activations for {checkpoint_name}")

        # Run PCA analysis
        device = get_device()
        regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
            raw_actvs=actvs,
            belief_states=torch.tensor(belief_states, device=device),
            save_dir=results_dir,
            run_id=checkpoint_name,
            save_data=save_regression_data,
        )

        print(f"Rank {rank}: Completed {label}")

        return {
            'checkpoint_name': checkpoint_name,
            'step': checkpoint_info['step'],
            'is_final': is_final,
            'n_layers': len(mses),
            'status': 'success'
        }

    except Exception as e:
        print(f"Rank {rank}: ERROR processing {label}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        return {
            'checkpoint_name': checkpoint_name,
            'step': checkpoint_info['step'],
            'is_final': is_final,
            'status': 'failed',
            'error': str(e)
        }


def aggregate_results(results_dir: str, comm, rank: int):
    """
    Aggregate all checkpoint results (rank 0 only, after barrier).

    Flow:
    1. comm.Barrier() - wait for all ranks to finish
    2. Rank 0: Scan results_dir for all *_metrics.csv files
    3. Rank 0: Load and concatenate into master DataFrame
    4. Rank 0: Add checkpoint metadata (step, name)
    5. Rank 0: Save to results_dir/all_checkpoints_summary.csv
    6. Return aggregated DataFrame (None for non-0 ranks)

    Args:
        results_dir: Path to results directory
        comm: MPI communicator
        rank: MPI rank

    Returns:
        Aggregated DataFrame (rank 0 only), None otherwise
    """
    # Wait for all ranks to finish
    if comm is not None:
        comm.Barrier()

    if rank != 0:
        return None

    print(f"Rank 0: Aggregating results...")

    # Find all metrics files
    metrics_files = glob.glob(os.path.join(results_dir, "*_metrics.csv"))

    if not metrics_files:
        print(f"Rank 0: No metrics files found in {results_dir}")
        return None

    # Load and concatenate all metrics
    dfs = []
    for metrics_file in metrics_files:
        try:
            df = pd.read_csv(metrics_file)

            # Extract checkpoint name from filename
            filename = os.path.basename(metrics_file)
            checkpoint_name = filename.replace('_metrics.csv', '')

            # Add checkpoint metadata
            df['checkpoint_name'] = checkpoint_name

            # Extract step number if it's a checkpoint
            if checkpoint_name.startswith('checkpoint_step_'):
                match = re.search(r'checkpoint_step_(\d+)', checkpoint_name)
                if match:
                    df['step'] = int(match.group(1))
                else:
                    df['step'] = -1
            else:
                # Final model
                df['step'] = 999999999

            dfs.append(df)
        except Exception as e:
            print(f"Rank 0: Warning - failed to load {metrics_file}: {e}")

    if not dfs:
        print(f"Rank 0: No valid metrics files found")
        return None

    # Concatenate all dataframes
    results_df = pd.concat(dfs, ignore_index=True)

    # Sort by step and layer
    results_df = results_df.sort_values(['step', 'layer'])

    # Save aggregated results
    summary_path = os.path.join(results_dir, "all_checkpoints_summary.csv")
    results_df.to_csv(summary_path, index=False)

    print(f"Rank 0: Aggregated results saved to {summary_path}")
    print(f"Rank 0: Total checkpoints processed: {results_df['checkpoint_name'].nunique()}")
    print(f"Rank 0: Total rows: {len(results_df)}")

    return results_df


def process_checkpoints_mpi(
    run_dir: str,
    checkpoint_pattern: str = "checkpoint_step_*.pt",
    include_final: bool = True,
    batch_size: int = 100_000,
    seq_length: int = 10,
    use_mpi: bool = False,
    cache_dir: str = None,
    results_dir: str = None,
    force_regenerate_obs: bool = False,
    save_activations: bool = True,
    save_regression_data: bool = False,
) -> dict:
    """
    Main entry point for multi-checkpoint processing.

    Args:
        run_dir: Path to training run directory
        checkpoint_pattern: Glob pattern for checkpoint files
        include_final: Include final model file
        batch_size: Number of observation sequences
        seq_length: Sequence length
        use_mpi: Enable MPI parallelization
        cache_dir: Cache directory (default: run_dir/.cache)
        results_dir: Results directory (default: run_dir/pca_results)
        force_regenerate_obs: Force regeneration of observations
        save_activations: Cache activations per checkpoint
        save_regression_data: Save full regression data

    Returns:
        dict with keys:
        - 'checkpoints': List of checkpoint metadata
        - 'results_df': Aggregated DataFrame (rank 0) or None
        - 'cache_dir': Path to cache directory
        - 'results_dir': Path to results directory
        - 'n_processed': Number of checkpoints processed by this rank
    """
    # Initialize MPI
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

    # Set default directories
    if cache_dir is None:
        cache_dir = os.path.join(run_dir, ".cache")
    if results_dir is None:
        results_dir = os.path.join(run_dir, "pca_results")

    # Create results directory
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        print(f"=" * 70)
        print(f"MULTI-CHECKPOINT PROCESSING")
        print(f"=" * 70)
        print(f"Run directory: {run_dir}")
        print(f"Cache directory: {cache_dir}")
        print(f"Results directory: {results_dir}")
        print(f"MPI enabled: {use_mpi}")
        if use_mpi:
            print(f"MPI ranks: {size}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_length}")
        print(f"Include final model: {include_final}")
        print(f"=" * 70)

    # Load config
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    process_name = config['train']['process']

    # Instantiate HMM process
    process_map = {
        'mess3': Mess3Proc,
        'rrxor': RRXOR,
        'z1r': Z1R,
        'psl7': PSL7HMM,
    }

    if process_name not in process_map:
        raise ValueError(f"Unknown process: {process_name}")

    process = process_map[process_name]()

    if rank == 0:
        print(f"Process: {process_name}")

    # Discover checkpoints (rank 0 only, then broadcast)
    if rank == 0:
        checkpoints = discover_checkpoints(run_dir, checkpoint_pattern, include_final)
        print(f"Discovered {len(checkpoints)} checkpoints/models to process")
    else:
        checkpoints = None

    if comm is not None:
        checkpoints = comm.bcast(checkpoints, root=0)

    if len(checkpoints) == 0:
        print(f"Rank {rank}: No checkpoints found!")
        return {
            'checkpoints': [],
            'results_df': None,
            'cache_dir': cache_dir,
            'results_dir': results_dir,
            'n_processed': 0
        }

    # Get or create cache
    observations, belief_states = get_or_create_cache(
        cache_dir=cache_dir,
        process=process,
        batch_size=batch_size,
        seq_length=seq_length,
        force_regenerate=force_regenerate_obs,
        rank=rank,
        comm=comm,
    )

    if rank == 0:
        print(f"Observations shape: {observations.shape}")
        print(f"Belief states shape: {belief_states.shape}")

    # Distribute checkpoints across ranks
    my_checkpoints = distribute_checkpoints(checkpoints, rank, size)

    print(f"Rank {rank}: Assigned {len(my_checkpoints)} checkpoints")

    # Process each checkpoint
    results = []
    for ckpt_info in my_checkpoints:
        result = process_single_checkpoint(
            checkpoint_info=ckpt_info,
            observations=observations,
            belief_states=belief_states,
            run_dir=run_dir,
            results_dir=results_dir,
            cache_dir=cache_dir,
            save_activations=save_activations,
            save_regression_data=save_regression_data,
            rank=rank,
        )
        results.append(result)

    # Aggregate results
    results_df = aggregate_results(results_dir, comm, rank)

    if rank == 0:
        print(f"=" * 70)
        print(f"PROCESSING COMPLETE")
        print(f"=" * 70)

    return {
        'checkpoints': checkpoints,
        'results_df': results_df,
        'cache_dir': cache_dir,
        'results_dir': results_dir,
        'n_processed': len(my_checkpoints)
    }


def main():
    """Command-line interface for multi-checkpoint processing."""
    parser = argparse.ArgumentParser(
        description="Process multiple model checkpoints with MPI support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to training run directory (e.g., records/20251207_232030)"
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="checkpoint_step_*.pt",
        help="Glob pattern for checkpoint files"
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        default=True,
        help="Include final model file"
    )
    parser.add_argument(
        "--no-include-final",
        dest="include_final",
        action="store_false",
        help="Don't include final model file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of observation sequences to generate"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=10,
        help="Sequence length"
    )
    parser.add_argument(
        "--use-mpi",
        action="store_true",
        help="Enable MPI parallelization"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: {run_dir}/.cache)"
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory (default: {run_dir}/pca_results)"
    )
    parser.add_argument(
        "--force-regenerate-obs",
        action="store_true",
        help="Force regeneration of observations cache"
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        default=True,
        help="Cache activations per checkpoint"
    )
    parser.add_argument(
        "--no-save-activations",
        dest="save_activations",
        action="store_false",
        help="Don't cache activations"
    )
    parser.add_argument(
        "--save-regression-data",
        action="store_true",
        help="Save full regression data (large files)"
    )

    args = parser.parse_args()

    # Run processing
    result = process_checkpoints_mpi(
        run_dir=args.run_dir,
        checkpoint_pattern=args.checkpoint_pattern,
        include_final=args.include_final,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        use_mpi=args.use_mpi,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        force_regenerate_obs=args.force_regenerate_obs,
        save_activations=args.save_activations,
        save_regression_data=args.save_regression_data,
    )

    # Print summary (rank 0 only)
    if not args.use_mpi or (MPI_AVAILABLE and MPI.COMM_WORLD.Get_rank() == 0):
        print(f"\nSummary:")
        print(f"  Total checkpoints: {len(result['checkpoints'])}")
        print(f"  Cache directory: {result['cache_dir']}")
        print(f"  Results directory: {result['results_dir']}")
        if result['results_df'] is not None:
            print(f"  Aggregated results: {result['results_dir']}/all_checkpoints_summary.csv")


if __name__ == "__main__":
    main()
