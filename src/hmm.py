import torch
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from jaxtyping import Float, Int
from typing import Tuple
import os
import itertools
import einops
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

    def _sampling_tables(self):
        """
        Precompute per-state CDFs over the flattened (observation, next_state) joint
        distribution, plus decode tables from a flat index back to (obs, next_state).

        Uses the same flat-index convention as the reference sampler in
        `generate_sequence`: idx = observation * num_hidden_states + next_state.
        Cached on first use; the emission matrices are read exactly once.
        """
        if getattr(self, "_sampling_tables_cache", None) is None:
            E = self.emission_matrices                 # (d_vocab, n_states, n_states)
            S = self.num_hidden_states
            # Row i is E[:, i, :].flatten(), i.e. the joint over (obs, next) given state i
            joint = np.transpose(E, (1, 0, 2)).reshape(S, -1)
            joint = joint / joint.sum(axis=1, keepdims=True)
            cdfs = np.cumsum(joint, axis=1)
            cdfs[:, -1] = 1.0                          # guard against float drift
            n_pairs = joint.shape[1]
            self._sampling_tables_cache = (
                [row.tolist() for row in cdfs],        # plain lists -> C bisect, no numpy call overhead
                [i // S for i in range(n_pairs)],      # flat idx -> observation
                [i % S for i in range(n_pairs)],       # flat idx -> next state
            )
        return self._sampling_tables_cache

    def _generate_sequence_fast(self, length: int, init_state: int, use_tqdm: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse-CDF sampler over precomputed per-state tables.

        Reads the emission matrices once instead of once per token, draws uniforms in
        bulk, and uses `bisect` on Python lists to avoid per-token numpy call overhead.

        Output is bit-identical to the reference loop under the same seed: this consumes
        exactly one uniform per token via inverse-CDF with side='right', which is what
        `np.random.choice(n, p=...)` does internally. Verified over millions of tokens
        for every process in this module, but note it rests on that implementation
        detail of numpy's legacy RandomState rather than on a documented guarantee.
        """
        from bisect import bisect_right

        cdfs, obs_of, next_of = self._sampling_tables()
        states = np.zeros(length, dtype=np.int64)
        obs = np.zeros(length, dtype=np.int64)
        states_out, obs_out = states, obs      # locals for the hot loop
        current_state = init_state

        CHUNK = 1_000_000
        pbar = tqdm(total=length, desc="Generating sequence") if use_tqdm else None
        pos = 0
        while pos < length:
            n = min(CHUNK, length - pos)
            uniforms = np.random.random(n).tolist()
            for u in uniforms:
                states_out[pos] = current_state
                idx = bisect_right(cdfs[current_state], u)
                obs_out[pos] = obs_of[idx]
                current_state = next_of[idx]
                pos += 1
            if pbar is not None:
                pbar.update(n)
        if pbar is not None:
            pbar.close()

        return states, obs

    def generate_sequence(self, length: int, init_state: int | None = None, use_tqdm: bool = False,
                          fast: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data generation process

        Args:
            fast: Use the precomputed-CDF sampler (see `_generate_sequence_fast`).
                  ~50-150x faster and bit-identical to the default under the same seed.

        Returns:
            Tuple of (states, observations) as numpy arrays
        """
        if init_state is None:
            # Sample initial state from stationary distribution instead of uniform random
            stationary = self.get_stationary_distribution()
            init_state = np.random.choice(self.num_hidden_states, p=stationary)

        if fast:
            return self._generate_sequence_fast(length, init_state, use_tqdm)

        states = np.zeros(length, dtype=np.int64)
        obs = np.zeros(length, dtype=np.int64)
        current_state = init_state

        for t_idx in tqdm(range(length), desc="Generating sequence") if use_tqdm else range(length):
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

    def generate_data(self, batch_size: int, length: int, init_state: int | None = None, use_tqdm: bool = False,
                      fast: bool = False) -> torch.Tensor:
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

            states, obs = self.generate_sequence(length, init_state=seq_init_state, fast=fast)
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

class DirectedCycleHMM(HMM):
    """
    HMM on a directed cycle with noisy emissions.

    Hidden states sit on a cycle of length num_states.
    Transition: state i -> (i+1)%N with prob `bias`, (i-1)%N with prob `1-bias`.
    Emission: state i emits token i with prob `1-emission_noise`, uniform noise elsewhere.
    d_vocab = num_states (one token per state).
    """

    def __init__(self, num_states=5, bias=0.9, emission_noise=0.3, seed=None):
        self.num_states = num_states
        self.bias = bias
        self.emission_noise = emission_noise
        # seed unused — HMM is fully deterministic given (num_states, bias, emission_noise)
        self._E = self._build_emission_matrices()

    def _build_emission_matrices(self):
        N = self.num_states
        b = self.bias
        eps = self.emission_noise

        # Transition matrix: T[i, j] = P(next=j | current=i)
        T = np.zeros((N, N))
        for i in range(N):
            T[i, (i + 1) % N] = b
            T[i, (i - 1) % N] = 1 - b

        # Observation matrix: O[j, i] = P(emit token j | state i)
        O = np.full((N, N), eps / (N - 1))
        np.fill_diagonal(O, 1 - eps)

        # Emission matrices: E[j, i, k] = P(observe j AND transition to k | state i) = O[j,i] * T[i,k]
        E = np.einsum('ji,ik->jik', O, T)
        return E

    @property
    def emission_matrices(self):
        return self._E


class CylinderGraphHMM(HMM):
    """HMM based on a cylinder graph structure with depth and nodes per level."""

    def __init__(
        self,
        n: int,
        depth: int,
        tokens_per_cluster: int = 16,
        dirichlet_alpha: float = 0.5,
        p: float = 0.1,
        int_range: int = 2,
        seed: int = None
    ):
        """
        Initialize a Cylinder Graph HMM.

        Args:
            n: Number of nodes per level (width of cylinder)
            depth: Number of levels (height of cylinder)
            tokens_per_cluster: Number of tokens per depth level
            dirichlet_alpha: Concentration parameter for Dirichlet prior on token distributions
            p: Probability of transitioning to next layer (vs staying in current layer)
            int_range: Integer range parameter (currently unused in cylinder graph)
            seed: Random seed for reproducibility
        """
        self.n = n
        self.depth = depth
        self.tokens_per_cluster = tokens_per_cluster
        self.dirichlet_alpha = dirichlet_alpha
        self.p = p
        self.seed = seed

        # Build the complete HMM
        self.graph = self._build_cylinder_graph()

    def _build_cylinder_graph(self) -> nx.DiGraph:
        """Construct the cylinder graph with token distributions."""
        if self.seed is not None:
            np.random.seed(self.seed)

        G = nx.DiGraph()

        # Add nodes
        for i, (l, j) in enumerate(itertools.product(range(self.depth), range(self.n))):
            G.add_node(i, depth=l, position=j)

        # Add edges with weights
        for l in range(self.depth):
            for j in range(self.n):
                # NN and NNN transitions within layer
                current_node = l * self.n + j
                NN_node = l * self.n + (j + 1) % self.n
                NNN_node = l * self.n + (j + 2) % self.n
                next_layer_node = ((l + 1) % self.depth) * self.n + j

                random_weight = np.random.uniform(0, 1 - self.p)
                G.add_edge(current_node, NN_node, weight=random_weight)
                G.add_edge(current_node, NNN_node, weight=1 - self.p - random_weight)
                G.add_edge(current_node, next_layer_node, weight=self.p)

        # Add token distributions to nodes
        self._add_token_distributions(G)

        return G

    def _add_token_distributions(self, G: nx.DiGraph) -> None:
        """Add token emission distributions to each node."""
        d_vocab = self.tokens_per_cluster * self.depth

        for node, attrs in G.nodes(data=True):
            # Sample from symmetric Dirichlet for this cluster
            token_dist = np.random.dirichlet(np.ones(self.tokens_per_cluster) * self.dirichlet_alpha)

            # Embed into full vocabulary (only non-zero for this depth's tokens)
            token_dist_all = np.zeros(d_vocab)
            start_idx = attrs['depth'] * self.tokens_per_cluster
            end_idx = (attrs['depth'] + 1) * self.tokens_per_cluster
            token_dist_all[start_idx:end_idx] = token_dist

            attrs['token_dist'] = token_dist_all

    def _get_emission_matrix(self) -> Float[np.ndarray, "d_vocab num_hidden_states num_hidden_states"]:
        """Compute emission matrix from graph structure."""
        # Get state transition matrix
        state_transition = nx.adjacency_matrix(self.graph, weight='weight').todense()

        # Get emission matrix (d_vocab x num_states)
        d_vocab = self.graph.nodes(data=True)[0]['token_dist'].shape[0]
        emission = np.zeros((d_vocab, self.graph.number_of_nodes()))
        for i in range(self.graph.number_of_nodes()):
            emission[:, i] = self.graph.nodes(data=True)[i]['token_dist']

        # Combine: P(token, next_state | current_state)
        tot_emission_matrix = einops.einsum(
            state_transition, emission,
            "current_state next_state, d_vocab current_state -> d_vocab current_state next_state"
        )
        return tot_emission_matrix
    
    @property
    def emission_matrices(self):
        # Cached: the matrix is fully determined by the (seeded) graph built in
        # __init__, but generate_sequence touches this property once per token.
        if getattr(self, "_emission_cache", None) is None:
            self._emission_cache = self._get_emission_matrix()
        return self._emission_cache


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
