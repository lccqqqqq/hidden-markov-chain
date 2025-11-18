import torch as t
from abc import ABC, abstractmethod
from jaxtyping import Float, Int
from typing import Tuple
import os
from itertools import product
import einops
from tqdm import tqdm

class HMM(ABC):
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def emission_matrices(self) -> t.Tensor:
        pass
    
    @property
    def num_hidden_states(self) -> int:
        return self.emission_matrices.shape[1]
    
    @property
    def d_vocab(self) -> int:
        return self.emission_matrices.shape[0]
    
    def get_stationary_distribution(self) -> t.Tensor:
        """
        Calculate the stationary distribution (initial belief state) for the HMM.
        Based on the combined transition matrix from all emission matrices.
        """
        # Combined transition matrix - sum over all vocabulary tokens
        T = t.sum(self.emission_matrices, dim=0)  # Shape: (num_hidden_states, num_hidden_states)
        
        # Find eigenvalues and eigenvectors of transpose
        eigenvalues, eigenvectors = t.linalg.eig(T.T)
        
        # Find the index of eigenvalue closest to 1
        idx = t.argmin(t.abs(eigenvalues - 1.0))
        
        # Extract the corresponding eigenvector and take real part
        stationary = t.real(eigenvectors[:, idx])
        
        # Normalize to ensure it sums to 1
        stationary = stationary / t.sum(stationary)
        
        return stationary
    
    def generate_sequence(self, length: int, init_state: int | None = None) -> Tuple[Int[t.Tensor, "length"], Int[t.Tensor, "length"]]:
        """
        Data generation process
        """
        if init_state is None:
            # Sample initial state from stationary distribution instead of uniform random
            stationary = self.get_stationary_distribution()
            init_state = t.multinomial(stationary, 1).item()
        
        states = t.zeros(length, dtype=t.int64)
        obs = t.zeros(length, dtype=t.int64)
        current_state = init_state
        
        for t_idx in range(length):
            states[t_idx] = current_state
            probs = self.emission_matrices[:, current_state, :].flatten() # the current state's emission matrix
            if t_idx <= 10:
                assert abs(probs.sum().item() - 1.) < 1e-6, f"Probs sum to {probs.sum()}"
            generated_sample = t.multinomial(probs, num_samples=1).item()
            generated_observation = generated_sample // self.num_hidden_states
            generated_next_state = generated_sample % self.num_hidden_states
            
            obs[t_idx] = generated_observation
            current_state = generated_next_state
        
        return states, obs
    
    def mixed_state_presentation(self, obs: Int[t.Tensor, "batch_size length"] = None) -> Float[t.Tensor, "batch_size length num_hidden_states"]:
        """
        Present the generated sequence as traces of mixed states in the probability simplex
        Supports both single sequences (length,) and batched inputs (batch_size, length)
        """
        
        # Handle single sequence input by adding batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, length = obs.shape
        
        # Initialize belief with stationary distribution instead of uniform
        stationary = self.get_stationary_distribution()
        belief = stationary.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_hidden_states)
        
        beliefs = []
        
        for t_idx in range(length):
            # Apply emission matrix for current observations across all batches
            obs_t = obs[:, t_idx]  # (batch_size,)
            
            # Gather emission matrices for current observations
            emission_mats = self.emission_matrices[obs_t]  # (batch_size, num_hidden_states, num_hidden_states)
            
            # Update beliefs: belief @ emission_matrix for each batch
            belief = t.bmm(belief.unsqueeze(1), emission_mats).squeeze(1)  # (batch_size, num_hidden_states)
            
            # Normalize beliefs
            belief = belief / belief.sum(dim=1, keepdim=True)
            beliefs.append(belief)
        
        result = t.stack(beliefs, dim=1)  # (batch_size, length, num_hidden_states)
        
        # Remove batch dimension if input was single sequence
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def generate_process_and_save_to_disk(
        self,
        length: int,
        save_dir: str,
    ):
        states, obs = self.generate_sequence(length)
        beliefs = self.mixed_state_presentation(obs)

        os.makedirs(save_dir, exist_ok=True)
        t.save(beliefs, os.path.join(save_dir, f"beliefs_{self.__class__.__name__}_{length}.pt"))
    
    def generate_data(self, batch_size: int, length: int, use_tqdm: bool = False) -> Int[t.Tensor, "batch_size length"]:
        obs_batch = []
        for _ in tqdm(range(batch_size), desc="Generating data") if use_tqdm else range(batch_size):
            states, obs = self.generate_sequence(length)
            obs_batch.append(obs)

        return t.stack(obs_batch, dim=0)
    
    @t.no_grad()
    def entropy_rate_theory_estimate(self):
        stationary = self.get_stationary_distribution()
        entropy_rate = 0.
        for i in range(self.num_hidden_states):
            prob = stationary[i]
            for j, k in product(range(self.d_vocab), range(self.num_hidden_states)):
                if self.emission_matrices[j, i, k] != 0:
                    entropy_rate += -prob * self.emission_matrices[j, i, k] * t.log(self.emission_matrices[j, i, k])
                    
        return entropy_rate.item()
    
    @t.no_grad()
    def entropy_rate_empirical_estimate(self, length: int, burn_in: int = 0):
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
                entropy_term = einops.einsum(
                    current_belief,
                    self.emission_matrices,
                    "current_state, vocab current_state next_state -> vocab next_state"
                ).sum(dim=1)
            
                entropy += - (entropy_term * t.log(entropy_term+1e-6)).sum()
                count += 1
            
            current_belief = current_belief @ self.emission_matrices[obs[i]]
            current_belief = current_belief / current_belief.sum()
                
        return entropy.item() / count
            

        


class RRXOR(HMM):
    def __init__(self):
        pass
    
    @property
    def emission_matrices(self) -> t.Tensor:
        E = t.zeros(2, 5, 5)
        # E[0, 2, 0] = 1.
        # E[0, 3, 2] = 1/2
        # E[0, 0, 3] = 1/2
        # E[0, 1, 4] = 1/2
        # E[1, 4, 0] = 1.
        # E[1, 3, 4] = 1/2
        # E[1, 0, 1] = 1/2
        # E[1, 1, 2] = 1/2
        E[0] = t.tensor(
            [
                [0, 0.5, 0, 0, 0],
                [0, 0, 0, 0, 0.5],
                [0, 0, 0, 0.5, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0]
            ]
        )
        E[1] = t.tensor(
            [
                [0, 0, 0.5, 0, 0],
                [0, 0, 0, 0.5, 0],
                [0, 0, 0, 0, 0.5],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        )
        return E

class Z1R(HMM):
    def __init__(self):
        pass
    
    @property
    def emission_matrices(self) -> t.Tensor: 
        E = t.zeros(2, 3, 3)
        E[0, 0, 1] = 1.
        E[0, 2, 0] = 1/2
        E[1, 1, 2] = 1.
        E[1, 2, 0] = 1/2
        return E

class Mess3Proc(HMM):
    def __init__(self):
        pass
    
    @property
    def emission_matrices(self) -> t.Tensor:
        E = t.zeros(3, 3, 3)
        E[0] = t.tensor(
            [
                [0.765, 0.00375, 0.00375],
                [0.0425, 0.0675, 0.00375],
                [0.0425, 0.00375, 0.0675]
            ]
        )
        E[1] = t.tensor(
            [
                [0.0675, 0.0425, 0.00375],
                [0.00375, 0.765, 0.00375],
                [0.00375, 0.0425, 0.0675]
            ]
        )
        E[2] = t.tensor(
            [
                [0.0675, 0.00375, 0.0425],
                [0.00375, 0.0675, 0.0425],
                [0.00375, 0.00375, 0.765]
            ]
        )
        return E


def main():
    # mess3 = Mess3Proc()
    # mess3.generate_process_and_save_to_disk(100000, "data")
    
    # z1r = Z1R()
    # z1r.generate_process_and_save_to_disk(100, "data")
    # print(z1r.entropy_rate_theory_estimate())
    # print(z1r.entropy_rate_empirical_estimate(10000, burn_in=1000))
    # rrxor = RRXOR()
    # rrxor.generate_process_and_save_to_disk(1000, "data")
    # print(rrxor.entropy_rate_theory_estimate())
    # print(rrxor.entropy_rate_empirical_estimate(10000, burn_in=1000))
    mess3 = Mess3Proc()
    mess3.generate_process_and_save_to_disk(1000, "data")
    print(mess3.entropy_rate_theory_estimate())
    print(mess3.entropy_rate_empirical_estimate(10000, burn_in=1000))
    

if __name__ == "__main__":
    main()
    