"""
CLI entry point for running linear probes on trained transformer models.

Usage:
    # Single model
    python src/run_linear_probe.py --model-dir models/cylinderhmm/20250301_run1/

    # Batch of models
    python src/run_linear_probe.py --model-dirs models/cylinderhmm/*/

    # Batch with MPI
    mpirun -n 4 python src/run_linear_probe.py --model-dirs models/cylinderhmm/*/ --use-mpi
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch as t

from activation_extraction import (
    get_device,
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from linear_probe import probe_layer_wise, probe_concatenated
from probe_io import save_probe_results

# Optional MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def run_single_model(
    model_dir: str,
    batch_size: int = 100_000,
    seq_length: int = 16,
    mode: str = "both",
    test_fraction: float = 0.2,
    device: str | None = None,
    inference_batch_size: int | None = None,
    save_data: bool = False,
    output_dir: str | None = None,
    force: bool = False,
    # Pre-generated data (for batch mode to avoid regeneration)
    precomputed_sequences: t.Tensor | None = None,
    precomputed_beliefs: np.ndarray | None = None,
) -> str:
    """
    Run the full linear probing pipeline for a single model directory.

    Returns:
        Path to the output directory where results were saved.
    """
    if device is None:
        device = get_device()

    if output_dir is None:
        output_dir = os.path.join(model_dir, "probe_results")

    # Check skip-if-done
    metrics_path = os.path.join(output_dir, "metrics.csv")
    if os.path.exists(metrics_path) and not force:
        print(f"Skipping {model_dir} — results already exist. Use --force to re-run.")
        return output_dir

    # Load config
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_n_ctx = config["model"]["n_ctx"]

    # Generate or use pre-computed data
    if precomputed_sequences is not None and precomputed_beliefs is not None:
        sequences = precomputed_sequences
        belief_states = precomputed_beliefs
    else:
        process = get_process_from_config(config)
        print(f"Generating {batch_size} sequences of length {seq_length} from {process.__class__.__name__}")
        sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
        belief_states = process.mixed_state_presentation(sequences.numpy())

    # Truncate to model's context window
    if model_n_ctx < seq_length:
        print(f"Truncating sequences from {seq_length} to {model_n_ctx} (model n_ctx)")
        sequences = sequences[:, :model_n_ctx]
        belief_states = belief_states[:, :model_n_ctx, :]

    # Load model and extract activations
    print(f"Loading model from {model_dir}")
    model = load_model_from_dir(model_dir, device=device)

    print("Extracting residual stream activations...")
    raw_actvs = extract_residual_stream(
        model, sequences, device=device, batch_size=inference_batch_size
    )

    # Free model memory
    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()

    # Prepare probe data
    layer_actvs, targets = prepare_probe_data(raw_actvs, belief_states, use_last_pos=True)
    del raw_actvs

    # Run probes
    if mode in ("layerwise", "both"):
        print("Running layer-wise probes...")
        lw_results = probe_layer_wise(layer_actvs, targets, test_fraction=test_fraction)
        save_probe_results(
            lw_results, output_dir, run_id="layerwise",
            save_regressors=True, save_data=save_data,
        )
        for i, r in enumerate(lw_results):
            print(f"  Layer {i}: train R²={r.train_r2:.4f}, test R²={r.test_r2:.4f}, test MSE={r.test_mse:.6f}")

    if mode in ("concat", "both"):
        print("Running concatenated probe...")
        concat_result = probe_concatenated(layer_actvs, targets, test_fraction=test_fraction)
        save_probe_results(
            [concat_result], output_dir, run_id="concat",
            save_regressors=True, save_data=save_data,
        )
        print(f"  Concat: train R²={concat_result.train_r2:.4f}, test R²={concat_result.test_r2:.4f}")

    print(f"Results saved to {output_dir}")
    return output_dir


def run_batch(
    model_dirs: list[str],
    use_mpi: bool = False,
    batch_size: int = 100_000,
    seq_length: int = 16,
    device: str | None = None,
    **kwargs,
) -> None:
    """
    Run probes on multiple model directories with optional MPI parallelism.

    Rank 0 generates shared data and broadcasts to all ranks.
    """
    if use_mpi:
        if not MPI_AVAILABLE:
            raise ImportError("MPI requested but mpi4py not available.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    # Device resolution with MPI-aware GPU assignment
    if device is None:
        device = get_device()
    if use_mpi and device.startswith("cuda") and device == "cuda":
        if t.cuda.is_available():
            gpu_id = rank % t.cuda.device_count()
            device = f"cuda:{gpu_id}"
            print(f"[Rank {rank}] Assigned to GPU {gpu_id}")

    print(f"[Rank {rank}/{size}] Using device: {device}")

    # Rank 0 generates data, broadcasts to all
    if rank == 0:
        # Read config from first valid directory to determine the HMM process
        config_path = None
        for d in model_dirs:
            cp = os.path.join(d, "config.yaml")
            if os.path.exists(cp):
                config_path = cp
                break
        if config_path is None:
            raise FileNotFoundError("No config.yaml found in any model directory.")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        process = get_process_from_config(config)
        print(f"[Rank 0] Generating {batch_size} sequences of length {seq_length}")
        sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
        belief_states = process.mixed_state_presentation(sequences.numpy())
    else:
        sequences = None
        belief_states = None

    if use_mpi:
        sequences = comm.bcast(sequences, root=0)
        belief_states = comm.bcast(belief_states, root=0)

    # Distribute directories
    dirs_per_rank = np.array_split(model_dirs, size)
    my_dirs = list(dirs_per_rank[rank])

    print(f"[Rank {rank}] Processing {len(my_dirs)} model(s)")

    successful = []
    failed = []

    for model_dir in my_dirs:
        try:
            run_single_model(
                model_dir,
                batch_size=batch_size,
                seq_length=seq_length,
                device=device,
                precomputed_sequences=sequences,
                precomputed_beliefs=belief_states,
                **kwargs,
            )
            successful.append(model_dir)
        except Exception as e:
            print(f"[Rank {rank}] FAILED {model_dir}: {e}")
            failed.append((model_dir, str(e)))

    # Summary
    print(f"\n[Rank {rank}] Done: {len(successful)} succeeded, {len(failed)} failed")
    for d, err in failed:
        print(f"  FAILED: {d} — {err}")

    if use_mpi:
        comm.Barrier()


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probes on trained transformer models"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=str, help="Single model directory")
    group.add_argument("--model-dirs", nargs="+", help="Multiple model directories")

    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--mode", choices=["layerwise", "concat", "both"], default="both")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--use-mpi", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")

    args = parser.parse_args()

    if args.model_dir:
        run_single_model(
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            mode=args.mode,
            test_fraction=args.test_fraction,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            save_data=args.save_data,
            output_dir=args.output_dir,
            force=args.force,
        )
    else:
        run_batch(
            model_dirs=args.model_dirs,
            use_mpi=args.use_mpi,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            mode=args.mode,
            test_fraction=args.test_fraction,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            save_data=args.save_data,
            force=args.force,
        )


if __name__ == "__main__":
    main()
