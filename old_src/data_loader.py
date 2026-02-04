import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


class HMMDataset(Dataset):
    """
    PyTorch Dataset for pre-computed HMM observation sequences.

    Loads pre-generated observation sequences from disk and returns
    (input, target) pairs for next-token prediction training.
    """

    def __init__(self, dataset_path):
        """
        Initialize dataset from pre-computed observations.

        Args:
            dataset_path: Path to directory containing observations.pt and meta.json
        """
        self.dataset_path = dataset_path

        # Load metadata
        meta_path = os.path.join(dataset_path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Load observations with memory mapping for efficiency
        obs_path = os.path.join(dataset_path, "observations.pt")
        if not os.path.exists(obs_path):
            raise FileNotFoundError(f"Observations file not found: {obs_path}")

        self.observations = torch.load(obs_path, map_location='cpu', mmap=True)

        # Validate shape
        expected_samples = self.metadata['num_samples']
        expected_seq_len = self.metadata['seq_length']
        actual_shape = self.observations.shape

        if actual_shape != (expected_samples, expected_seq_len):
            raise ValueError(
                f"Shape mismatch: expected {(expected_samples, expected_seq_len)}, "
                f"got {actual_shape}"
            )

        print(f"Loaded dataset: {self.metadata['process']}")
        print(f"  Samples: {expected_samples:,}")
        print(f"  Sequence length: {expected_seq_len}")
        print(f"  Vocab size: {self.metadata['vocab_size']}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.observations)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input, target) tensors where:
            - input: all tokens except the last (for model input)
            - target: all tokens except the first (for next-token prediction)
        """
        sequence = self.observations[idx]

        # Split into input and target
        input_seq = sequence[:-1]   # All tokens except last
        target_seq = sequence[1:]   # All tokens except first

        return input_seq, target_seq


def create_dataloader(dataset_path, batch_size, shuffle=True, device=None):
    """
    Create a PyTorch DataLoader for pre-computed HMM datasets.

    Args:
        dataset_path: Path to directory containing observations.pt and meta.json
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data (default: True)
        device: Device to move data to (default: None, stays on CPU)

    Returns:
        DataLoader instance
    """
    dataset = HMMDataset(dataset_path)

    # Create DataLoader
    # Note: We don't use num_workers because the data is already in memory
    # and we may want to move it to GPU
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device is not None and device.type == 'cuda')
    )

    return dataloader


def load_all_splits(base_dataset_path, batch_size, device=None):
    """
    Load train, validation, and test DataLoaders from a dataset directory.

    Args:
        base_dataset_path: Base path to dataset directory containing train/, val/, test/ subdirectories
        batch_size: Number of samples per batch
        device: Device to move data to (default: None, stays on CPU)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        FileNotFoundError: If any of the required splits are missing
    """
    # Check that all splits exist
    train_path = os.path.join(base_dataset_path, "train")
    val_path = os.path.join(base_dataset_path, "val")
    test_path = os.path.join(base_dataset_path, "test")

    for split_name, split_path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Missing {split_name} split at {split_path}. "
                f"Run 'python generate_dataset.py <config>' to generate all splits."
            )

    # Create DataLoaders for each split
    # Train: shuffle=True for better training
    # Val/Test: shuffle=False for deterministic evaluation
    train_loader = create_dataloader(train_path, batch_size, shuffle=True, device=device)
    val_loader = create_dataloader(val_path, batch_size, shuffle=False, device=device)
    test_loader = create_dataloader(test_path, batch_size, shuffle=False, device=device)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <dataset_path>")
        print("Example: python data_loader.py data/datasets/z1r")
        sys.exit(1)

    dataset_path = sys.argv[1]

    print(f"Testing data loader with dataset: {dataset_path}")
    print("-" * 50)

    # Create dataset
    dataset = HMMDataset(dataset_path)

    print(f"\nDataset size: {len(dataset)}")

    # Test getting a few samples
    print("\nSample 0:")
    input_seq, target_seq = dataset[0]
    print(f"  Input shape: {input_seq.shape}")
    print(f"  Target shape: {target_seq.shape}")
    print(f"  Input: {input_seq[:10]}...")  # First 10 tokens
    print(f"  Target: {target_seq[:10]}...")

    # Test DataLoader
    print("\nTesting DataLoader with batch_size=64:")
    dataloader = create_dataloader(dataset_path, batch_size=64, shuffle=True)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"  Batch {batch_idx}: inputs {inputs.shape}, targets {targets.shape}")
        if batch_idx >= 2:  # Just show first 3 batches
            break

    print("\nData loader test complete!")
