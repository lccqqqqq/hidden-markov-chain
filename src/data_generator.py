"""
Generate synthetic training data and save to disk
"""

import yaml
import inspect
from src.hmm import HMM
from src.hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM, CylinderGraphHMM
from src.utils import create_process_from_dict
from mpi4py import MPI
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


def generate_data(data_generator_config_path="base_config.yaml"):
    """Generate dataset from config file."""
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


if __name__ == "__main__":
    # generate_data()
    consolidate_and_split(data_dir="data/datasets/cylinder_graph_hmm")
