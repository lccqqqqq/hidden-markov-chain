"""
Shared test fixtures for the linear probing pipeline.
"""

import sys
import os
import pytest
import numpy as np
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Add src/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def tiny_model():
    """1-layer, d_model=4, vocab=2, n_ctx=6, attn_only HookedTransformer."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        d_head=4,
        n_heads=1,
        d_vocab=2,
        n_ctx=6,
        attn_only=True,
        normalization_type=None,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def z1r_data():
    """200 sequences of length 6 from Z1R, with belief states."""
    from hmm import Z1R

    process = Z1R()
    batch_obs = process.generate_data(batch_size=200, length=6)
    belief_states = process.mixed_state_presentation(batch_obs.numpy())
    return {
        "process": process,
        "sequences": batch_obs,          # (200, 6) int tensor
        "belief_states": belief_states,   # (200, 6, 3) numpy
    }
