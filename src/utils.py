from transformer_lens import HookedTransformer, HookedTransformerConfig
import yaml
import inspect
from hmm import HMM
from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM, CylinderGraphHMM, DirectedCycleHMM

# Process Registry
PROCESS_REGISTRY = {
    "rrxor": RRXOR,
    "z1r": Z1R,
    "mess3": Mess3Proc,
    "psl7": PSL7HMM,
    "cylinder_graph": CylinderGraphHMM,
    "directed_cycle": DirectedCycleHMM,
}


# Config Utilities
def initialize_transformer_from_yaml(config_path: str, model_cfg: dict = None) -> HookedTransformer:
    if model_cfg is None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg['model']

    hooked_config = HookedTransformerConfig(
        n_layers=model_cfg['n_layer'],
        d_model=model_cfg['n_embd'],
        d_head=model_cfg['d_head'],
        n_heads=model_cfg['n_head'],
        d_vocab=model_cfg['vocab_size'],
        n_ctx=model_cfg['n_ctx'],
        d_mlp=model_cfg.get('d_mlp'),
        act_fn=model_cfg['activation_function'],
        normalization_type=model_cfg.get('normalization_type', 'LN'),
        attn_only=model_cfg.get('attn_only', False),
        tie_word_embeddings=model_cfg.get('tie_word_embeddings', False),
    )

    return HookedTransformer(hooked_config)


def create_process_from_dict(process_config: dict) -> HMM:
    """
    Factory function to create HMM process from config dictionary.

    Automatically filters parameters to only those accepted by the process class,
    so different processes can have different constructor signatures without errors.

    Args:
        process_config: Dict with 'name' and 'params' keys

    Returns:
        Instantiated HMM process

    Example:
        config = {
            "name": "cylinder_graph",
            "params": {"n": 6, "depth": 3, "tokens_per_cluster": 16}
        }
        proc = create_process_from_dict(config)
    """
    process_name = process_config["name"]
    all_params = process_config.get("params", {})

    # Get process class from registry
    if process_name not in PROCESS_REGISTRY:
        available = ', '.join(PROCESS_REGISTRY.keys())
        raise ValueError(f"Unknown process '{process_name}'. Available: {available}")

    process_cls = PROCESS_REGISTRY[process_name]

    # Get valid parameters for this class's __init__
    sig = inspect.signature(process_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}

    # Filter kwargs to only include valid ones
    filtered_params = {k: v for k, v in all_params.items() if k in valid_params}

    # Warn about ignored parameters (helpful for debugging config typos)
    ignored_params = set(all_params.keys()) - valid_params
    if ignored_params:
        print(f"Note: Process '{process_name}' ignoring parameters: {ignored_params}")

    return process_cls(**filtered_params)