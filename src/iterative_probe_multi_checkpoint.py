"""
Multi-checkpoint iterative probing with MPI support.

Runs iterative probing (no counterfactual) on each epoch checkpoint to track
how the number of decodable belief-state copies and subspace structure evolve
during training.

Usage:
    # Single-rank (no MPI)
    python src/iterative_probe_multi_checkpoint.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --seq-length 10 --batch-size 10000 --max-iterations 15 --r2-threshold 0.01 \
        --device cpu

    # Multi-rank with MPI
    mpirun -n 6 python src/iterative_probe_multi_checkpoint.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --seq-length 10 --batch-size 10000 --max-iterations 15 --r2-threshold 0.01 \
        --device cpu --use-mpi
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
import torch as t
import yaml
from datetime import datetime

from activation_extraction import (
    get_device,
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from iterative_probe import iterative_probe_all_layers

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_epoch_checkpoints(model_dir: str, include_best: bool = True) -> list[dict]:
    """
    Discover epoch or step checkpoint files and optionally the best model.

    Supports both checkpoint_epoch_*.pt and checkpoint_step_*.pt files.
    Returns list of dicts with keys: filename, epoch (=global_step for sorting), name.
    Sorted by global step ascending, with best_model last.

    When both epoch and step checkpoints exist, epoch checkpoints are resolved
    to their actual global_step by reading the saved checkpoint metadata.
    """
    checkpoints = []
    has_step_checkpoints = bool(glob.glob(os.path.join(model_dir, "checkpoint_step_*.pt")))

    # Find checkpoint_epoch_*.pt
    epoch_pattern = os.path.join(model_dir, "checkpoint_epoch_*.pt")
    for path in glob.glob(epoch_pattern):
        filename = os.path.basename(path)
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
        if match:
            epoch = int(match.group(1))
            # When step checkpoints also exist, resolve epoch to global_step
            # so sorting is consistent
            if has_step_checkpoints:
                try:
                    ckpt = t.load(path, map_location="cpu", weights_only=False)
                    sort_key = ckpt.get("global_step", epoch)
                except Exception:
                    sort_key = epoch
            else:
                sort_key = epoch
            checkpoints.append({
                "filename": filename,
                "epoch": sort_key,
                "name": f"epoch_{epoch:02d}",
            })

    # Find checkpoint_step_*.pt (dense step-level checkpoints)
    step_pattern = os.path.join(model_dir, "checkpoint_step_*.pt")
    for path in glob.glob(step_pattern):
        filename = os.path.basename(path)
        match = re.search(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            step = int(match.group(1))
            checkpoints.append({
                "filename": filename,
                "epoch": step,  # use step number for ordering
                "name": f"step_{step:06d}",
            })

    checkpoints.sort(key=lambda x: x["epoch"])

    if include_best:
        best_path = os.path.join(model_dir, "best_model.pt")
        if os.path.exists(best_path):
            # Epoch for best_model: use max epoch + 1 for sorting
            max_epoch = checkpoints[-1]["epoch"] if checkpoints else 0
            checkpoints.append({
                "filename": "best_model.pt",
                "epoch": max_epoch + 1,
                "name": "best_model",
            })

    return checkpoints


# ---------------------------------------------------------------------------
# Shared data cache
# ---------------------------------------------------------------------------

def get_or_create_shared_data(
    model_dir: str,
    batch_size: int,
    seq_length: int,
    cache_dir: str,
    rank: int,
    comm,
    force: bool = False,
) -> tuple[t.Tensor, np.ndarray]:
    """
    Generate (rank 0) or load (all ranks) shared observation/belief-state data.

    Rank 0 generates sequences and belief states, saves to cache_dir.
    All ranks load from cache after barrier.
    """
    obs_path = os.path.join(cache_dir, "observations.pt")
    belief_path = os.path.join(cache_dir, "belief_states.npy")
    meta_path = os.path.join(cache_dir, "cache_metadata.json")

    if rank == 0:
        cache_valid = (
            not force
            and os.path.exists(obs_path)
            and os.path.exists(belief_path)
            and os.path.exists(meta_path)
        )

        if cache_valid:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("batch_size") != batch_size or meta.get("seq_length") != seq_length:
                    cache_valid = False
            except Exception:
                cache_valid = False

        if not cache_valid:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Rank 0: Generating {batch_size} sequences of length {seq_length}...")

            config_path = os.path.join(model_dir, "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            process = get_process_from_config(config)
            sequences = process.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True)
            belief_states = process.mixed_state_presentation(sequences.numpy())

            # Truncate to model context window if needed
            model_n_ctx = config["model"]["n_ctx"]
            if model_n_ctx < seq_length:
                print(f"Rank 0: Truncating {seq_length} -> {model_n_ctx} (model n_ctx)")
                sequences = sequences[:, :model_n_ctx]
                belief_states = belief_states[:, :model_n_ctx, :]

            t.save(sequences, obs_path)
            np.save(belief_path, belief_states)

            metadata = {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "process": process.__class__.__name__,
                "observations_shape": list(sequences.shape),
                "belief_states_shape": list(belief_states.shape),
                "created": datetime.now().isoformat(),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Rank 0: Cache saved to {cache_dir}")
        else:
            print(f"Rank 0: Using existing cache in {cache_dir}")

    # Synchronize
    if comm is not None:
        comm.Barrier()

    # All ranks load from cache
    sequences = t.load(obs_path, weights_only=True)
    belief_states = np.load(belief_path)

    return sequences, belief_states


# ---------------------------------------------------------------------------
# Per-checkpoint processing
# ---------------------------------------------------------------------------

def process_single_checkpoint(
    ckpt_info: dict,
    sequences: t.Tensor,
    belief_states: np.ndarray,
    model_dir: str,
    output_dir: str,
    args,
    rank: int,
) -> pd.DataFrame | None:
    """
    Run iterative probing on a single checkpoint and save per-checkpoint CSV.

    Returns DataFrame with columns: epoch, layer, iteration, train_r2, test_r2, ...
    """
    name = ckpt_info["name"]
    epoch = ckpt_info["epoch"]
    filename = ckpt_info["filename"]

    print(f"Rank {rank}: Processing {name} ({filename})...")

    try:
        # Load model
        model = load_model_from_dir(model_dir, checkpoint_filename=filename, device=args.device)

        # Extract activations
        raw_actvs = extract_residual_stream(
            model, sequences, device=args.device, batch_size=args.inference_batch_size,
        )

        # Prepare probe data (last position)
        layer_actvs, targets = prepare_probe_data(raw_actvs, belief_states, use_last_pos=True)
        del raw_actvs

        # Run iterative probing
        results = iterative_probe_all_layers(
            layer_actvs, targets,
            max_iterations=args.max_iterations,
            r2_threshold=args.r2_threshold,
            test_fraction=args.test_fraction,
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Build DataFrame
        rows = []
        for lr in results:
            cum_dims = 0
            for i, it in enumerate(lr.iterations):
                r = it.probe_result
                rank_val = it.subspace.shape[0]
                cum_dims += rank_val
                rows.append({
                    "epoch": epoch,
                    "name": name,
                    "layer": lr.layer,
                    "iteration": i,
                    "train_r2": r.train_r2,
                    "test_r2": r.test_r2,
                    "train_mse": r.train_mse,
                    "test_mse": r.test_mse,
                    "subspace_rank": rank_val,
                    "cumulative_dims": cum_dims,
                    "singular_values": ",".join(f"{s:.6f}" for s in it.singular_values),
                })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{name}_iterative_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Rank {rank}: {name} done — {len(rows)} rows saved to {csv_path}")

        # Free model memory
        del model
        if t.cuda.is_available():
            t.cuda.empty_cache()

        return df

    except Exception as e:
        print(f"Rank {rank}: ERROR processing {name}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(output_dir: str, comm, rank: int) -> pd.DataFrame | None:
    """
    After barrier, rank 0 aggregates all per-checkpoint CSVs.
    """
    if comm is not None:
        comm.Barrier()

    if rank != 0:
        return None

    print("Rank 0: Aggregating results...")

    csv_files = glob.glob(os.path.join(output_dir, "*_iterative_metrics.csv"))
    if not csv_files:
        print("Rank 0: No per-checkpoint CSV files found!")
        return None

    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Rank 0: Warning — failed to load {path}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["epoch", "layer", "iteration"])

    out_path = os.path.join(output_dir, "all_checkpoints_iterative_metrics.csv")
    merged.to_csv(out_path, index=False)

    n_ckpts = merged["name"].nunique()
    print(f"Rank 0: Aggregated {n_ckpts} checkpoints, {len(merged)} rows -> {out_path}")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run iterative probing across training checkpoints"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Model directory with config.yaml and checkpoints")
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--r2-threshold", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--use-mpi", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate data cache even if it exists")

    args = parser.parse_args()

    if args.device is None:
        args.device = get_device()

    # MPI setup
    if args.use_mpi:
        if not MPI_AVAILABLE:
            raise RuntimeError("--use-mpi requested but mpi4py not installed")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    # Directories
    output_dir = os.path.join(args.model_dir, "iterative_probe_checkpoints")
    cache_dir = os.path.join(output_dir, "cache")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print("=" * 60)
        print("ITERATIVE PROBE — MULTI-CHECKPOINT")
        print("=" * 60)
        print(f"Model dir:       {args.model_dir}")
        print(f"Output dir:      {output_dir}")
        print(f"Device:          {args.device}")
        print(f"MPI ranks:       {size}")
        print(f"Max iterations:  {args.max_iterations}")
        print(f"R² threshold:    {args.r2_threshold}")
        print(f"Batch size:      {args.batch_size}")
        print(f"Seq length:      {args.seq_length}")
        print("=" * 60)

    # Discover checkpoints
    if rank == 0:
        checkpoints = discover_epoch_checkpoints(args.model_dir, include_best=True)
        print(f"Found {len(checkpoints)} checkpoints: "
              + ", ".join(c["name"] for c in checkpoints))
    else:
        checkpoints = None

    if comm is not None:
        checkpoints = comm.bcast(checkpoints, root=0)

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Shared data
    sequences, belief_states = get_or_create_shared_data(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        cache_dir=cache_dir,
        rank=rank,
        comm=comm,
        force=args.force,
    )

    if rank == 0:
        print(f"Observations: {sequences.shape}, Belief states: {belief_states.shape}")

    # Round-robin distribution
    my_checkpoints = [c for i, c in enumerate(checkpoints) if i % size == rank]
    print(f"Rank {rank}: assigned {len(my_checkpoints)} checkpoints: "
          + ", ".join(c["name"] for c in my_checkpoints))

    # Process each checkpoint
    for ckpt_info in my_checkpoints:
        process_single_checkpoint(
            ckpt_info=ckpt_info,
            sequences=sequences,
            belief_states=belief_states,
            model_dir=args.model_dir,
            output_dir=output_dir,
            args=args,
            rank=rank,
        )

    # Aggregate
    merged = aggregate_results(output_dir, comm, rank)

    if rank == 0:
        print("=" * 60)
        print("DONE")
        print("=" * 60)
        if merged is not None:
            # Summary per epoch
            for name in merged["name"].unique():
                sub = merged[merged["name"] == name]
                n_layers = sub["layer"].nunique()
                total_iters = len(sub)
                print(f"  {name}: {n_layers} layers, {total_iters} total iteration rows")


if __name__ == "__main__":
    main()
