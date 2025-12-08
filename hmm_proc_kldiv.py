"""
Compute the KL divergence between the true and learned HMMs.
"""

import numpy as np
from scipy.special import logsumexp
from hmm import Mess3Proc, HMM
import torch as t
from tqdm import tqdm
from model import HookedTransformerModel
import os

# Constants
MESS3_WEIGHT_PATHS = {
    4: 'records/20251127_122141',
    6: 'records/20251127_135107',
    8: 'records/20251127_135742',
    10: 'records/20251127_134441',
    15: 'records/20251127_135035',
}
CONFIG_PATHS = {key: os.path.join(value, 'config.yaml') for key, value in MESS3_WEIGHT_PATHS.items()}
MODEL_PATHS = {key: os.path.join(value, 'final_model_mess3.pt') for key, value in MESS3_WEIGHT_PATHS.items()}
DEFAULT_DEVICE = 'cuda' if t.cuda.is_available() else 'mps' if t.backends.mps.is_available() else 'cpu'


def compute_log_likelihood_forward_conditional(hmm: HMM, observations: np.ndarray, offset: int = 0, reduction: str | None = None, use_tqdm: bool = False) -> np.ndarray:
    """
    Compute log P(x_offset+1, ..., x_T | x_0, ..., x_offset) using forward algorithm.
    
    This matches transformer which can't predict first token.
    
    Args:
        hmm: HMM instance
        observations: (batch_size, seq_length) where observations[:,0] is conditioning variable
    
    Returns:
        log_likelihoods: (batch_size,) log P(x_0,...,x_T) and (batch_size,) log P(x_0,...,x_offset)
    
    Some pending optimization on code logic... but passed all tests.
    """
    batch_size, seq_length = observations.shape
    n_states = hmm.num_hidden_states
    emission_matrices = hmm.emission_matrices
    log_emission_matrices = np.log(emission_matrices)
    
    # Get stationary distribution for initialization
    init_state = hmm.get_stationary_distribution()
    log_init_state = np.log(init_state)
    
    log_likelihoods = []
    baseline_log_likelihoods = []
    for batch_idx in tqdm(range(batch_size), desc="Computing conditional log-likelihoods", disable=not use_tqdm):
        obs_seq = observations[batch_idx]
        
        # Initialize with x₀: compute belief state after observing x₀
        # belief(s') = Σ_s π(s) × T[x₀, s, s']
        log_emission_0 = log_emission_matrices[obs_seq[0]]  # (n_states, n_states)
        log_alpha = logsumexp(log_init_state[:, np.newaxis] + log_emission_0, axis=0)
        
        # Now compute P(x₁, ..., x_{T-1} | belief after x₀)
        # This is the conditional likelihood given x₀
        log_conditional_likelihood = 0.0
        if offset == 0:
            baseline_log_likelihood = logsumexp(log_alpha)
        elif offset >= seq_length:
            raise ValueError(f"Offset {offset} is too large for sequence length {seq_length}")
        elif offset == -1:
            baseline_log_likelihood = logsumexp(log_init_state)
        for t in range(1, seq_length):
            log_emission_t = log_emission_matrices[obs_seq[t]]
            
            # Compute P(x_t | previous belief)
            # = Σ_s belief(s) × Σ_s' T[x_t, s, s']
            log_prob_xt = logsumexp(log_alpha[:, np.newaxis] + log_emission_t)
            log_conditional_likelihood += log_prob_xt
            
            # Update belief: α_new(s') = Σ_s α(s) × T[x_t, s, s']
            log_alpha = logsumexp(log_alpha[:, np.newaxis] + log_emission_t, axis=0)
            if t == offset:
                baseline_log_likelihood = logsumexp(log_alpha)
        
        # print(log_conditional_likelihood)
        # print(logsumexp(log_alpha))
        baseline_log_likelihoods.append(logsumexp(baseline_log_likelihood))
        log_likelihoods.append(logsumexp(log_alpha))
    
    cond_log_likelihoods = np.array(log_likelihoods) - np.array(baseline_log_likelihoods)
    if reduction == 'mean':
        return cond_log_likelihoods.mean()
    elif reduction == 'sum':
        return cond_log_likelihoods.sum()
    elif reduction is None:
        return cond_log_likelihoods
    else:
        print(f"Invalid reduction: {reduction}, returning all values")
        return cond_log_likelihoods


@t.no_grad()
def compute_transformer_log_likelihood(model: HookedTransformerModel, sequences: t.Tensor, reduction: str | None = None) -> np.ndarray:
    """
    Compute log P(x_n_ctx, ..., x_T | x_0, ..., x_{n_ctx-1}) from transformer next-token predictions.
    Uses GPU-accelerated batching for efficiency.
    """
    batch_size, seq_length = sequences.shape
    n_ctx = model.model.cfg.n_ctx
    
    # Create all chopped sequences for all batches at once
    # For each sequence, create chunks [i:i+n_ctx] for i in range(seq_length - n_ctx)
    indices = range(seq_length - n_ctx)
    chopped_seqs = t.stack([sequences[:, i:i+n_ctx] for i in indices], dim=1)
    # Shape: (batch_size, seq_length - n_ctx, n_ctx)
    
    # Flatten batch dimensions for model input
    flat_input = chopped_seqs.reshape(-1, n_ctx)
    # Shape: (batch_size * (seq_length - n_ctx), n_ctx)
    
    # Process all at once
    logits = model(flat_input)
    # Shape: (batch_size * (seq_length - n_ctx), n_ctx, vocab_size)
    
    # Reshape back to separate batch dimension
    logits = logits.reshape(batch_size, seq_length - n_ctx, n_ctx, -1)
    # Take last position predictions
    log_probs = t.log_softmax(logits[:, :, -1, :], dim=-1)
    # Shape: (batch_size, seq_length - n_ctx, vocab_size)
    
    # Gather correct tokens
    correct_tokens = sequences[:, n_ctx:]  # (batch_size, seq_length - n_ctx)
    log_probs_selected = t.gather(log_probs, dim=-1, index=correct_tokens.unsqueeze(-1)).squeeze(-1)
    # Shape: (batch_size, seq_length - n_ctx)
    
    # Sum over sequence length to get per-sequence log-likelihoods
    seq_log_likelihoods = log_probs_selected.sum(dim=1)
    # Shape: (batch_size,)
    
    if reduction == 'mean':
        return seq_log_likelihoods.mean().cpu().numpy()
    elif reduction == 'sum':
        return seq_log_likelihoods.sum().cpu().numpy()
    elif reduction is None:
        return seq_log_likelihoods.cpu().numpy()
    else:
        print(f"Invalid reduction: {reduction}, returning all values")
        return seq_log_likelihoods.cpu().numpy()


def generate_test_sequences(hmm: HMM, batch_size: int = 1000, seq_length: int = 1000, device: str = DEFAULT_DEVICE, seed: int = 42) -> t.Tensor:
    """
    Generate test sequences from the HMM.
    """
    if seed is not None:
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed(seed)
    
    return hmm.generate_data(batch_size=batch_size, length=seq_length, use_tqdm=True).to(device)


def main(verbose: bool = True, save_dir: str = 'data/kl'):
    """
    Compute KL divergences for all context lengths and save results.

    Args:
        verbose: If True, print progress and results
        save_dir: Directory to save the results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    true_hmm = Mess3Proc()
    test_sequences = generate_test_sequences(true_hmm, batch_size=100, seq_length=1000, device=DEFAULT_DEVICE, seed=45)
    test_sequences_np = test_sequences.cpu().numpy()

    # Compute full sequence log likelihood once
    full_seq_log_lik = compute_log_likelihood_forward_conditional(true_hmm, test_sequences_np, offset=-1, reduction=None)

    for ctx_len, config_path in CONFIG_PATHS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing context length: {ctx_len}")
            print(f"{'='*60}")
            print(f"Loading model from: {MODEL_PATHS[ctx_len]}")

        model = HookedTransformerModel(config_path)
        model.load_state_dict(t.load(MODEL_PATHS[ctx_len], map_location='cpu'))
        model.eval()
        model.to(DEFAULT_DEVICE)

        if verbose:
            print(f"Model loaded. Device: {DEFAULT_DEVICE}")
            print("Computing KL divergence...")

        # Compute prefix log likelihood for this context length
        prefix_log_lik = compute_log_likelihood_forward_conditional(true_hmm, test_sequences_np[:, :ctx_len], offset=-1, reduction=None)
        true_cond_log_likelihoods = full_seq_log_lik - prefix_log_lik

        learned_cond_log_likelihoods = compute_transformer_log_likelihood(model, test_sequences.to(DEFAULT_DEVICE), reduction=None)
        time_avg_llr = (true_cond_log_likelihoods - learned_cond_log_likelihoods)/(test_sequences.shape[1])

        # Save to file
        output_path = os.path.join(save_dir, f'llr_mess3_ctxlen{ctx_len}.npy')
        np.save(output_path, time_avg_llr)

        if verbose:
            print(f"Context length {ctx_len}:")
            print(f"  Mean LLR: {time_avg_llr.mean():.6f}")
            print(f"  Std LLR: {time_avg_llr.std():.6f}")
            print(f"  Results saved to: {output_path}")
        


if __name__ == "__main__":
    main(verbose=True, save_dir='data/kl')