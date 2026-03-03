"""
Create a time-reversed version of the CylinderGraph HMM dataset.

Flips each observation sequence along the time axis, so that
next-token prediction on the reversed data corresponds to
previous-token prediction on the original data.

Usage:
    python src/reverse_dataset.py
"""

import json
import os
import torch

SRC_DIR = "data/datasets/cylinder_graph_hmm"
DST_DIR = "data/datasets/cylinder_graph_hmm_reversed"


def main():
    for split in ["train", "test"]:
        src_path = os.path.join(SRC_DIR, split, "observations.pt")
        dst_split_dir = os.path.join(DST_DIR, split)
        os.makedirs(dst_split_dir, exist_ok=True)

        data = torch.load(src_path)
        reversed_data = data.flip(dims=[1])
        torch.save(reversed_data, os.path.join(dst_split_dir, "observations.pt"))

        print(f"{split}: {data.shape} -> reversed {reversed_data.shape}")
        assert reversed_data.shape == data.shape
        assert torch.equal(reversed_data[0], data[0].flip(0))

    # Copy and extend metadata
    with open(os.path.join(SRC_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    metadata["reversed"] = True
    metadata["source_dir"] = SRC_DIR
    with open(os.path.join(DST_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nReversed dataset saved to {DST_DIR}")


if __name__ == "__main__":
    main()
