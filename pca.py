from hmm import RRXOR, Z1R, Mess3Proc
from model import HookedTransformerModel, MaskedHeadTransformerModel
import yaml
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math
from jaxtyping import Int, Float
import numpy as np
import einops
from utils import print_shape

from nnsight import NNsight
import transformer_lens

def load_model_from_weights(model_dir: str, file_name: str = 'final_model_mess3.pt'):
    """Load a model from a weights file, load with nnsight"""
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config["model"]["name"]
    try:
        model = globals()[model_name](
            config_path=os.path.join(model_dir, "config.yaml")
        )
    except KeyError:
        raise ValueError(f"Model {model_name} not found in globals, possibly due to an absent import")
    
    model.load_state_dict(t.load(os.path.join(model_dir, file_name)))
    model.eval()
    if t.cuda.is_available():
        model.to("cuda")
        device = "cuda" # Todo: the model does not have a device argument.
    else:
        model.to("cpu")
        device = "cpu"
    print(f"Loaded model {model_name} from {os.path.join(model_dir, file_name)} on {device}")
    return model

def extract_residual_stream(
    model: nn.Module,
    sequences: Int[t.Tensor, 'batch seq_len'],
    device: str = "cuda",
) -> Float[t.Tensor, 'layer batch seq_len d_model']:
    model.eval()
    model.to(device)
    with t.no_grad():
        # at the moment only have two variants of model architectures, they behave similarly loss-wise.
        # Follow the grammar of NNsight
        if model.__class__.__name__ == "MaskedHeadTransformerModel":
            actv_list = []
            with model.trace(sequences) as tracer:
                for layer in model.model.layers:
                    actv = layer.output[0].save()
                    actv_list.append(actv.cpu())
        elif model.__class__.__name__ == "HookedTransformerModel":
            # Follow the grammar of TransformerLens
            actv_list = []
            logits, cache = model.model.run_with_cache(sequences)
            for layer_idx in range(model.model.cfg.n_layers):
                actv = cache["resid_post", layer_idx]
                actv_list.append(actv.cpu())
        
    actvs = t.stack(actv_list)
    return actvs

def learn_affine_mapping(
    actvs: Float[t.Tensor, 'batch d_actv'],
    belief_states: Float[t.Tensor, 'batch d_vocab'],
):
    # Fit a linear regression model to map actvs to belief states
    actvs = actvs.cpu().numpy()
    belief_states = belief_states.cpu().numpy()
    regressor = LinearRegression()
    regressor.fit(actvs, belief_states)
    
    predictions = regressor.predict(actvs)
    mse = np.mean((predictions - belief_states)**2)
    return regressor, predictions,mse

def pca_concat_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True, # for some reason it's better to use the last position of the sequences for the PCA?
    save_dir: str = "pca_results",
    run_id: str = "",
):
    actvs = einops.rearrange(
        raw_actvs,
        'layer batch seq_len d_model -> batch seq_len (layer d_model)',
    )
    if use_last_pos:
        actvs = actvs[:, -1, :]
        belief_states = belief_states[:, -1, :]
    else:
        actvs = einops.rearrange(
            actvs,
            'batch seq_len concat_dim -> (batch seq_len) concat_dim',
        )
        belief_states = einops.rearrange(
            belief_states,
            'batch seq_len d_vocab -> (batch seq_len) d_vocab',
        )
    
    regressor, predictions, mse = learn_affine_mapping(actvs, belief_states)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{run_id}_regressor.pkl"), "wb") as f:
        pickle.dump(regressor, f)
    with open(os.path.join(save_dir, f"{run_id}_predictions.pkl"), "wb") as f:
        pickle.dump(predictions, f)
    return regressor, predictions, mse

def pca_layer_wise_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True,
    save_dir: str = "pca_results",
    run_id: str = "",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    regressors = []
    mses = []
    predictions = []
    
    if use_last_pos:
        belief_states = belief_states[:, -1, :]
    else:
        belief_states = einops.rearrange(
            belief_states,
            'batch seq_len d_vocab -> (batch seq_len) d_vocab',
        )
        
    print_shape(belief_states)
    
    for i in range(raw_actvs.shape[0]):
        actvs = raw_actvs[i, :, :, :]
        if use_last_pos:
            actvs = actvs[:, -1, :]
        else:
            actvs = einops.rearrange(
                actvs,
                'batch seq_len d_model -> (batch seq_len) d_model',
            )
        print_shape(actvs)
        
        regressor, preds, mse = learn_affine_mapping(actvs, belief_states)
        
        regressors.append(regressor)
        predictions.append(preds)
        mses.append(mse)
        print(f"Layer {i} MSE: {mse}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{run_id}_regressors.pkl"), "wb") as f:
        pickle.dump(regressors, f)
    with open(os.path.join(save_dir, f"{run_id}_predictions.pkl"), "wb") as f:
        pickle.dump(predictions, f)
    with open(os.path.join(save_dir, f"{run_id}_mses.pkl"), "wb") as f:
        pickle.dump(mses, f)
    
    return regressors, predictions, mses


def visualize_simplex_3D(
    predictions: Float[t.Tensor, 'batch d_vocab'],
):
    pass

if __name__ == "__main__":
    model = load_model_from_weights(
        model_dir="records/20250801_225134",
        file_name="final_model_mess3.pt"
    )
    print(model)
    
    process = Mess3Proc()
    batch_obs = process.generate_data(
        batch_size=100_000, length=10, use_tqdm=True,
    )
    
    # get the belief states
    belief_states = process.mixed_state_presentation(batch_obs)
    print(belief_states.shape)
    actvs = extract_residual_stream(
        model=model,
        sequences=batch_obs,
    )
    print(actvs.shape)
    
    regressors, predictions, mses = pca_layer_wise_actvs(
        raw_actvs=actvs,
        belief_states=belief_states,
        run_id="20250803_main",
    )
    
    print(predictions[0].shape)
    print(f"MSE: {mses[0]}")
    
    