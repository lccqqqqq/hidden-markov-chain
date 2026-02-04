from src.hmm import HMM, Mess3Proc
from src.model import HookedTransformerModel
from transformer_lens import HookedTransformer
from tqdm import tqdm
from scipy.special import logsumexp
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import itertools
from collections import Counter
import torch as t
import pickle
from jaxtyping import Float
import sys

DEFAULT_DEVICE = 'cuda' if t.cuda.is_available() else 'mps' if t.backends.mps.is_available() else 'cpu'

def load_model(model_dir_path, device=DEFAULT_DEVICE):
    model = HookedTransformerModel(os.path.join(model_dir_path, 'config.yaml'))
    model.load_state_dict(t.load(os.path.join(model_dir_path, 'final_model_mess3.pt'), map_location=device))
    model.eval()
    return model

def load_regressors(model_dir_path):
    regressors = pickle.load(open(os.path.join(model_dir_path, 'regressors.pkl'), 'rb'))
    return regressors

def generate_test_sequences(seq_len, num_sequences=1000, save_to_disk=False, save_dir="data"):
    proc = Mess3Proc()
    test_sequences = proc.generate_data(num_sequences, seq_len, init_state=None)
    if save_to_disk:
        os.makedirs(save_dir, exist_ok=True)
        t.save(test_sequences, os.path.join(save_dir, f"test_sequences_{seq_len}.pt"))
    return test_sequences, proc

def test_mse(model, test_sequences, regressors, layer, proc):
    model.eval()
    with t.no_grad():
        logits, cache = model.model.run_with_cache(test_sequences)
        resid = cache["resid_post", layer]

    X = resid[:, -1, :].cpu().numpy()
    belief_state = proc.mixed_state_presentation(test_sequences)
    y = belief_state[:, -1, :]

    r2_score = regressors[layer].score(X, y)
    mse = np.mean((regressors[layer].predict(X) - y)**2)
    return r2_score, mse

def get_directions(regressor, num_random_directions=4):
    coef = regressor.coef_
    U, S, Vt = np.linalg.svd(coef, full_matrices=True)

    # Para subspace: first min(3, rank) directions
    n_para = min(3, Vt.shape[0])
    para_subspace = [v.squeeze() for v in np.split(Vt[:n_para], n_para, axis=0)]

    # Orth subspace: remaining directions (if any)
    orth_subspace = []
    if Vt.shape[0] > 3:
        n_orth = Vt.shape[0] - 3
        orth_subspace = [v.squeeze() for v in np.split(Vt[3:], n_orth, axis=0)]

    # Random directions
    rand_directions = np.random.randn(num_random_directions, Vt.shape[1])
    rand_directions = rand_directions / np.linalg.norm(rand_directions, axis=1, keepdims=True)
    rand_directions = [v.squeeze() for v in np.split(rand_directions, num_random_directions, axis=0)]

    return para_subspace, orth_subspace, rand_directions


def ablate_direction(
    actvs: Float[t.Tensor, "batch seq_len d_model"],
    direction: Float[t.Tensor, "d_model"],
    method: str = "mean",
    position: int | None = -1
):
    """
    Ablate a direction from activations.

    Args:
        actvs: Activations tensor (batch, seq_len, d_model)
        direction: Direction vector to ablate (d_model,)
        method: "mean" (mean ablation) or "zero" (zero ablation)
        position: Position to ablate. -1 for last position (default), None for all positions
    """
    direction = direction / direction.norm()

    if position is None:
        # Ablate at all positions
        actvs_para_B = actvs @ direction  # (batch, seq_len)
        actvs_para_BM = actvs_para_B.unsqueeze(-1) * direction  # (batch, seq_len, d_model)
        if method == "mean":
            actvs_para_mean_M = actvs_para_BM.mean(dim=0)  # (seq_len, d_model)
            actvs = actvs - actvs_para_BM + actvs_para_mean_M.unsqueeze(0)
        elif method == "zero":
            actvs = actvs - actvs_para_BM
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        # Ablate at specific position
        actvs_para_B = actvs[:, position, :] @ direction  # (batch,)
        actvs_para_BM = actvs_para_B.unsqueeze(-1) * direction  # (batch, d_model)
        if method == "mean":
            actvs_para_mean_M = actvs_para_BM.mean(dim=0)  # (d_model,)
            actvs[:, position, :] = actvs[:, position, :] - actvs_para_BM + actvs_para_mean_M.unsqueeze(0)
        elif method == "zero":
            actvs[:, position, :] = actvs[:, position, :] - actvs_para_BM
        else:
            raise ValueError(f"Invalid method: {method}")

    return actvs


def get_clean_logits(model, test_sequences):
    logits, cache = model.model.run_with_cache(test_sequences)
    return logits[:, -1, :]

def get_ablated_logits(model, test_sequences, layer, direction, method, device=DEFAULT_DEVICE, position=-1):
    # Convert direction to tensor if it's numpy
    if isinstance(direction, np.ndarray):
        direction = t.from_numpy(direction).float().to(device)

    logits_abl = model.model.run_with_hooks(
        t.tensor(test_sequences) if isinstance(test_sequences, np.ndarray) else test_sequences,
        fwd_hooks=[
            (f"blocks.{layer}.hook_resid_post", lambda resid, hook: ablate_direction(resid, direction, method, position))
        ]
    )[:, -1, :]
    return logits_abl

def ablate_experiment(model, test_sequences, model_dir_path, device=DEFAULT_DEVICE, position=-1):
    n_ctx = model.model.cfg.n_ctx
    d_model = model.model.cfg.d_model
    n_layers = model.model.cfg.n_layers

    # Move test sequences to device once
    test_sequences = test_sequences.to(device)
    model = model.to(device)

    # Get clean logits and compute log_softmax once
    clean_logits = get_clean_logits(model, test_sequences)
    log_probs_clean = F.log_softmax(clean_logits, dim=-1)

    # Load regressors from the model directory path
    regressors = load_regressors(model_dir_path)

    all_kl_divs = []
    for layer in range(n_layers):
        reg = regressors[layer]
        para_directions, orth_directions, rand_directions = get_directions(reg)

        kl_divergence_at_layer = []

        # Process all direction types (order: para, orth, rand)
        for direction in para_directions + orth_directions + rand_directions:
            logits_abl = get_ablated_logits(model, test_sequences, layer, direction, "mean", device, position)
            log_probs_abl = F.log_softmax(logits_abl, dim=-1)

            # Compute KL(clean || ablated) - how much info is lost by ablation
            kl_divergence = F.kl_div(log_probs_abl, log_probs_clean,
                                    reduction='none', log_target=True).sum(dim=-1)
            kl_divergence_at_layer.append(kl_divergence)

        kl_divergence_at_layer = t.stack(kl_divergence_at_layer)
        all_kl_divs.append(kl_divergence_at_layer)

    all_kl_divs_LDB = t.stack(all_kl_divs)  # (n_layers, n_directions, num_test_sequences)
    return all_kl_divs_LDB.detach().cpu().numpy()

def get_model_list():
    metadata_path = '/mnt/extraspace/clin/records/metadata.parquet'
    df = pd.read_parquet(metadata_path)
    df_completed = df[df['status'] == 'completed']
    
    df_completed = df_completed.sort_values(by='final_val_loss', ascending=True) 
    model_dir_path_list = df_completed['run_dir'].tolist()
    return model_dir_path_list

def main():
    # Configuration
    position = None  # None for all positions, -1 for last position only

    model_dir_path_list = get_model_list()
    all_stats = {}
    # Fixed: seq_len=15, num_sequences=1000 (was backwards)
    sample, _ = generate_test_sequences(seq_len=15, num_sequences=1000)
    for model_dir_path in tqdm(model_dir_path_list):
        try:
            model = load_model(model_dir_path)
            test_sequences = sample[:, :model.model.cfg.n_ctx]
            # Pass model_dir_path to ablate_experiment
            all_kl_divs_LDB = ablate_experiment(model, test_sequences, model_dir_path, position=position)
            print(f"{model_dir_path}: {all_kl_divs_LDB.shape}")
            sys.stdout.flush()
            all_stats[model_dir_path] = all_kl_divs_LDB
        except Exception as e:
            print(f"Error processing {model_dir_path}: {e}")
            sys.stdout.flush()
            continue

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    suffix = 'all' if position is None else f'pos{position}'
    with open(f"data/ablation_stats_{suffix}.pkl", "wb") as f:
        pickle.dump(all_stats, f)
    print(f"Saved results for {len(all_stats)} models to data/ablation_stats_{suffix}.pkl")

if __name__ == "__main__":
    main()