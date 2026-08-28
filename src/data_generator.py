"""
Generate synthetic training data and save to disk
"""

import yaml
import inspect
from hmm import HMM
from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM, CylinderGraphHMM
from utils import create_process_from_dict
import os
import numpy as np
from pathlib import Path
import torch

PROCESS_REGISTRY = {
    "rrxor": RRXOR,
    "z1r": Z1R,
    "mess3": Mess3Proc,
    "psl7": PSL7HMM,
    "cylinder_graph": CylinderGraphHMM,
}


def generate_data(data_generator_config_path="config/base_config.yaml"):
    """Generate dataset from config file. Run under mpirun; each rank writes one shard."""
    from mpi4py import MPI
    with open(data_generator_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Parse data generator config
    data_generator_cfg = cfg["data_generator"]

    # Create process using factory (handles parameter filtering automatically)
    process_config = data_generator_cfg["process"]
    proc = create_process_from_dict(process_config)

    # Extract data generation settings
    num_tokens = data_generator_cfg["num_tokens"]
    save_dir = data_generator_cfg["save_dir"]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "shards"), exist_ok=True)
    
    comm.Barrier()
    tokens_per_worker = num_tokens // size
    np.random.seed(42 + rank)
    states, obs = proc.generate_sequence(tokens_per_worker, use_tqdm=True)
    np.save(os.path.join(save_dir, f"shards/obs_{rank:05d}.npy"), obs)

def consolidate_and_split(data_dir, seq_length=16, train_ratio=0.95, seed=42):
    """Consolidate MPI shards into train/test splits."""
    import json

    # Load all shards
    shards = [np.load(f) for f in sorted(Path(data_dir).glob("shards/obs_*.npy"))]
    all_obs = np.concatenate(shards)

    # Create non-overlapping sequences
    num_sequences = (len(all_obs) - seq_length) // seq_length
    sequences = np.array([
        all_obs[i*seq_length:(i+1)*seq_length+1]
        for i in range(num_sequences)
    ])

    # Shuffle before split
    np.random.seed(seed)
    np.random.shuffle(sequences)

    # Split into train/test (95/5)
    n = len(sequences)
    train_end = int(train_ratio * n)

    train_data = torch.from_numpy(sequences[:train_end])
    test_data = torch.from_numpy(sequences[train_end:])

    # Create directories and save
    (Path(data_dir) / "train").mkdir(exist_ok=True)
    (Path(data_dir) / "test").mkdir(exist_ok=True)

    torch.save(train_data, f"{data_dir}/train/observations.pt")
    torch.save(test_data, f"{data_dir}/test/observations.pt")

    # Save metadata
    metadata = {
        "seq_length": seq_length,
        "vocab_size": int(all_obs.max() + 1),
        "total_tokens": len(all_obs),
        "num_sequences": len(sequences),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_ratio": train_ratio,
        "seed": seed,
    }

    with open(f"{data_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base_config.yaml",
                        help="Path to the YAML config (default: config/base_config.yaml)")
    parser.add_argument("--stage", choices=["generate", "consolidate", "all"], default="all",
                        help="generate: MPI shard generation (run under mpirun). "
                             "consolidate: merge shards into train/test .pt. "
                             "all: both, consolidating on rank 0 (default)")
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Sequence length for consolidation (default: train.seq_length from config)")
    parser.add_argument("--train-ratio", type=float, default=0.99,
                        help="Fraction of sequences used for training (default: 0.99)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["data_generator"]["save_dir"]
    seq_length = args.seq_length or cfg["train"]["seq_length"]

    if args.stage in ("generate", "all"):
        generate_data(args.config)

    if args.stage in ("consolidate", "all"):
        # Under mpirun only rank 0 consolidates; without MPI this is a no-op guard.
        rank = 0
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
            rank = MPI.COMM_WORLD.Get_rank()
        except ImportError:
            pass
        if rank == 0:
            consolidate_and_split(data_dir=data_dir, seq_length=seq_length,
                                  train_ratio=args.train_ratio)
            print(f"Consolidated {data_dir} (seq_length={seq_length}, "
                  f"train_ratio={args.train_ratio})")


if __name__ == "__main__":
    main()
