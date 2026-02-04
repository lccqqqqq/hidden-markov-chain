import torch as t
from hmm import HMM
from model import HookedTransformerModel
from tqdm import tqdm
from scipy.special import logsumexp
import numpy as np


@t.no_grad()
def log_likelihood_tsfm(model: HookedTransformerModel, sequences: t.Tensor, device: str = 'cuda', batch_mean: bool = True, seq_mean: bool = True) -> t.Tensor:
    model.to(device)
    sequences = sequences.to(device)
    model.eval()
    n_ctx = model.model.cfg.n_ctx
    T = sequences.shape[1]
    batch_size = sequences.shape[0]
    vocab_size = model.model.cfg.d_vocab

    targets = sequences[:, n_ctx:] # Shape: (batch_size, T - n_ctx)

    seqs = t.stack(
        [sequences[:, i:i+n_ctx] for i in range(T - n_ctx)], dim=1
    ) # Shape: (batch_size, T - n_ctx, n_ctx)
    flat_input = seqs.reshape(batch_size * (T - n_ctx), n_ctx) # Shape: (batch_size * (T - n_ctx), n_ctx)
    logits = model(flat_input) # Shape: (batch_size * (T - n_ctx), n_ctx, vocab_size)
    logits = logits[:, -1, :] # Shape: (batch_size * (T - n_ctx), vocab_size)
    logits = logits.reshape(batch_size, T - n_ctx, vocab_size)
    log_probs = t.log_softmax(logits, dim=-1) # Shape: (batch_size, (T - n_ctx), vocab_size)

    log_probs_selected = t.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Shape: (batch_size, T - n_ctx)
    
    if batch_mean:
        log_probs_selected = log_probs_selected.mean(dim=0)
    if seq_mean:
        log_probs_selected = log_probs_selected.mean(dim=1)

    return log_probs_selected

def log_likelihood_true(hmm: HMM, sequences_np: np.ndarray, device: str = 'cuda', batch_mean: bool = True, seq_mean: bool = True, use_tqdm: bool = False) -> t.Tensor:
    batch_size, T = sequences_np.shape
    E = hmm.emission_matrices
    log_E = np.log(E)

    init_state = hmm.get_stationary_distribution()
    log_init_state = np.log(init_state)

    log_likelihoods = []
    for batch_idx in tqdm(range(batch_size), desc="Computing true log-likelihood", disable=not use_tqdm):
        seq = sequences_np[batch_idx]
        
        log_emission_0 = log_E[seq[0]]
        log_alpha = logsumexp(log_init_state[:, np.newaxis] + log_emission_0, axis=0)

        for t in range(1, T):
            log_emission_t = log_E[seq[t]]
            log_prob_xt = logsumexp(log_alpha[:, np.newaxis] + log_emission_t)
            log_alpha = logsumexp(log_alpha[:, np.newaxis] + log_emission_t, axis=0)
        
        log_likelihoods.append(logsumexp(log_alpha))

    if batch_mean:
        log_likelihoods = np.array(log_likelihoods).mean(axis=0)
    if seq_mean:
        log_likelihoods = np.array(log_likelihoods).mean(axis=1)

    return log_likelihoods


def llr_test(hmm, model, sequences, device: str = 'cuda'):
    n_ctx = model.model.cfg.n_ctx
    sequences_np = sequences.detach().cpu().numpy()
    llh_full_seq = log_likelihood_true(hmm, sequences_np, batch_mean=False, seq_mean=False, use_tqdm=True)
    llh_prefix_seq = log_likelihood_true(hmm, sequences_np[:, :n_ctx], batch_mean=False, seq_mean=False, use_tqdm=True)
    log_prob_tsfm = log_likelihood_tsfm(model, sequences, device=device, batch_mean=False, seq_mean=False).cpu().numpy()

    # formatting
    cond_log_prob_true = np.array(llh_full_seq) - np.array(llh_prefix_seq)
    cond_log_prob_tsfm = log_prob_tsfm.sum(axis=1)
    mean_llr = np.mean(cond_log_prob_true - cond_log_prob_tsfm)
    return mean_llr
    
def llr_test_models(hmm, model_paths, )
