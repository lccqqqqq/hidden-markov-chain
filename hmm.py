import torch
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, Int
from typing import Tuple
import os
from itertools import product
from tqdm import tqdm


class HMM(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def emission_matrices(self) -> np.ndarray:
        pass

    @property
    def num_hidden_states(self) -> int:
        return self.emission_matrices.shape[1]

    @property
    def d_vocab(self) -> int:
        return self.emission_matrices.shape[0]

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calculate the stationary distribution (initial belief state) for the HMM.
        Based on the combined transition matrix from all emission matrices.
        """
        # Combined transition matrix - sum over all vocabulary tokens
        T = np.sum(self.emission_matrices, axis=0)  # Shape: (num_hidden_states, num_hidden_states)

        # Find eigenvalues and eigenvectors of transpose
        eigenvalues, eigenvectors = np.linalg.eig(T.T)

        # Find the index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))

        # Extract the corresponding eigenvector and take real part
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to ensure it sums to 1
        stationary = stationary / np.sum(stationary)

        return stationary

    def generate_sequence(self, length: int, init_state: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data generation process

        Returns:
            Tuple of (states, observations) as numpy arrays
        """
        if init_state is None:
            # Sample initial state from stationary distribution instead of uniform random
            stationary = self.get_stationary_distribution()
            init_state = np.random.choice(self.num_hidden_states, p=stationary)

        states = np.zeros(length, dtype=np.int64)
        obs = np.zeros(length, dtype=np.int64)
        current_state = init_state

        for t_idx in range(length):
            states[t_idx] = current_state
            probs = self.emission_matrices[:, current_state, :].flatten()  # the current state's emission matrix
            if t_idx <= 10:
                assert abs(probs.sum() - 1.) < 1e-6, f"Probs sum to {probs.sum()}"

            # Sample from the joint distribution of (observation, next_state)
            generated_sample = np.random.choice(len(probs), p=probs)
            generated_observation = generated_sample // self.num_hidden_states
            generated_next_state = generated_sample % self.num_hidden_states

            obs[t_idx] = generated_observation
            current_state = generated_next_state

        return states, obs

    def mixed_state_presentation(self, obs: np.ndarray) -> np.ndarray:
        """
        Present the generated sequence as traces of mixed states in the probability simplex
        Supports both single sequences (length,) and batched inputs (batch_size, length)

        Args:
            obs: Observation sequences, shape (length,) or (batch_size, length)

        Returns:
            Belief states, shape (length, num_hidden_states) or (batch_size, length, num_hidden_states)
        """

        # Handle single sequence input by adding batch dimension
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, length = obs.shape

        # Initialize belief with stationary distribution
        stationary = self.get_stationary_distribution()
        belief = np.tile(stationary, (batch_size, 1))  # (batch_size, num_hidden_states)

        beliefs = []

        for t_idx in range(length):
            # Apply emission matrix for current observations across all batches
            obs_t = obs[:, t_idx]  # (batch_size,)

            # Gather emission matrices for current observations
            emission_mats = self.emission_matrices[obs_t]  # (batch_size, num_hidden_states, num_hidden_states)

            # Update beliefs: belief @ emission_matrix for each batch
            # belief: (batch_size, num_hidden_states)
            # emission_mats: (batch_size, num_hidden_states, num_hidden_states)
            # Result: (batch_size, num_hidden_states)
            belief = np.einsum('bi,bij->bj', belief, emission_mats)

            # Normalize beliefs
            belief = belief / belief.sum(axis=1, keepdims=True)
            beliefs.append(belief)

        result = np.stack(beliefs, axis=1)  # (batch_size, length, num_hidden_states)

        # Remove batch dimension if input was single sequence
        if squeeze_output:
            result = np.squeeze(result, axis=0)

        return result

    def generate_process_and_save_to_disk(
        self,
        length: int,
        save_dir: str,
    ):
        """Generate sequence and save beliefs to disk (as PyTorch tensor for compatibility)"""
        states, obs = self.generate_sequence(length)
        beliefs = self.mixed_state_presentation(obs)

        os.makedirs(save_dir, exist_ok=True)
        # Convert to torch tensor for saving
        torch.save(torch.from_numpy(beliefs), os.path.join(save_dir, f"beliefs_{self.__class__.__name__}_{length}.pt"))

    def generate_data(self, batch_size: int, length: int, init_state: int | None = None, use_tqdm: bool = False) -> torch.Tensor:
        """
        Generate a batch of observation sequences.

        Args:
            init_state: If -1, use random init state per sequence (avoids stationary dist computation).
                       If None, sample from stationary distribution.
                       If int >= 0, use that specific state for all sequences.

        Returns:
            PyTorch tensor of shape (batch_size, length) for compatibility with training pipeline
        """
        obs_batch = []
        for _ in tqdm(range(batch_size), desc="Generating data") if use_tqdm else range(batch_size):
            # Generate random init_state per sequence if init_state == -1 (sentinel value)
            if init_state == -1:
                import random
                seq_init_state = random.randint(0, self.num_hidden_states - 1)
            else:
                seq_init_state = init_state

            states, obs = self.generate_sequence(length, init_state=seq_init_state)
            obs_batch.append(obs)

        # Stack and convert to torch tensor for compatibility with training code
        obs_batch_np = np.stack(obs_batch, axis=0)
        return torch.from_numpy(obs_batch_np)

    def entropy_rate_theory_estimate(self):
        """Calculate theoretical entropy rate"""
        stationary = self.get_stationary_distribution()
        entropy_rate = 0.
        for i in range(self.num_hidden_states):
            prob = stationary[i]
            for j, k in product(range(self.d_vocab), range(self.num_hidden_states)):
                if self.emission_matrices[j, i, k] != 0:
                    entropy_rate += -prob * self.emission_matrices[j, i, k] * np.log(self.emission_matrices[j, i, k])

        return float(entropy_rate)

    def entropy_rate_empirical_estimate(self, length: int, burn_in: int = 0):
        """Calculate empirical entropy rate estimate"""
        # get stationary distribution
        stationary = self.get_stationary_distribution()
        # generate the sequence of belief states
        _, obs = self.generate_sequence(length + burn_in)

        # generate belief states one by one...
        current_belief = stationary
        entropy = 0.
        count = 0
        for i in range(len(obs)):
            if i >= burn_in:
                # Calculate entropy term
                # current_belief: (num_hidden_states,)
                # emission_matrices: (vocab, num_hidden_states, num_hidden_states)
                # Result: (vocab, num_hidden_states)
                entropy_term = np.einsum('i,jik->jk', current_belief, self.emission_matrices)
                entropy_term = entropy_term.sum(axis=1)  # Sum over next states

                entropy += -np.sum(entropy_term * np.log(entropy_term + 1e-6))
                count += 1

            current_belief = current_belief @ self.emission_matrices[obs[i]]
            current_belief = current_belief / current_belief.sum()

        return float(entropy / count)


class RRXOR(HMM):
    def __init__(self):
        pass

    @property
    def emission_matrices(self) -> np.ndarray:
        E = np.zeros((2, 5, 5))
        E[0] = np.array([
            [0, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ])
        E[1] = np.array([
            [0, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0.5],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return E


class Z1R(HMM):
    def __init__(self):
        pass

    @property
    def emission_matrices(self) -> np.ndarray:
        E = np.zeros((2, 3, 3))
        E[0, 0, 1] = 1.
        E[0, 2, 0] = 1/2
        E[1, 1, 2] = 1.
        E[1, 2, 0] = 1/2
        return E


class Mess3Proc(HMM):
    def __init__(self):
        pass

    @property
    def emission_matrices(self) -> np.ndarray:
        E = np.zeros((3, 3, 3))
        E[0] = np.array([
            [0.765, 0.00375, 0.00375],
            [0.0425, 0.0675, 0.00375],
            [0.0425, 0.00375, 0.0675]
        ])
        E[1] = np.array([
            [0.0675, 0.0425, 0.00375],
            [0.00375, 0.765, 0.00375],
            [0.00375, 0.0425, 0.0675]
        ])
        E[2] = np.array([
            [0.0675, 0.00375, 0.0425],
            [0.00375, 0.0675, 0.0425],
            [0.00375, 0.00375, 0.765]
        ])
        return E


class PSL7HMM(HMM):
    def __init__(self):
        # Load and cache emission matrix once at init time to avoid:
        # 1. Repeated disk I/O during generation (much faster!)
        # 2. File system contention when using MPI with many ranks
        # 3. Potential corruption from concurrent reads
        self._emission_matrices = np.load("data/psl_instance_emission_matrix.npy")

    @property
    def emission_matrices(self) -> np.ndarray:
        return self._emission_matrices


def main():
    # Test with Z1R
    z1r = Z1R()
    print("Testing Z1R HMM")
    print(f"Vocab size: {z1r.d_vocab}")
    print(f"Hidden states: {z1r.num_hidden_states}")
    print(f"Stationary distribution: {z1r.get_stationary_distribution()}")

    # Test sequence generation
    states, obs = z1r.generate_sequence(20)
    print(f"\nGenerated sequence (length 20):")
    print(f"States: {states}")
    print(f"Observations: {obs}")

    # Test batch generation
    batch = z1r.generate_data(batch_size=5, length=10)
    print(f"\nBatch generation (5 sequences of length 10):")
    print(f"Shape: {batch.shape}")
    print(f"Type: {type(batch)}")

    # Test entropy rate
    print(f"\nTheoretical entropy rate: {z1r.entropy_rate_theory_estimate():.6f}")
    print(f"Empirical entropy rate: {z1r.entropy_rate_empirical_estimate(10000, burn_in=1000):.6f}")


if __name__ == "__main__":
    main()
