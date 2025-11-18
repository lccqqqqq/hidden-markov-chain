import torch as t
import torch.nn as nn
import yaml
from nnsight import NNsight
from transformers import GPT2LMHeadModel, GPT2Config

from transformer_lens import HookedTransformer, HookedTransformerConfig

class MaskedHeadTransformerModel(nn.Module):
    def __init__(self, config_path):
        super(MaskedHeadTransformerModel, self).__init__()
        
        # Load config from YAML file
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        model_cfg = cfg['model']
        
        # Store desired parameters
        self.desired_n_head = model_cfg['n_head']
        self.d_head = model_cfg['d_head']
        self.n_embd = model_cfg['n_embd']
        
        # Calculate expected n_head from dimensions
        expected_n_head = self.n_embd // self.d_head
        
        # Configure GPT2 model (use expected_n_head to satisfy dimensional constraints)
        config = GPT2Config(
            vocab_size=model_cfg['vocab_size'],
            n_embd=model_cfg['n_embd'],
            n_head=expected_n_head,  # Use calculated value
            n_layer=model_cfg['n_layer'],
            n_positions=model_cfg['n_positions'],
            n_ctx=model_cfg['n_ctx'],
            activation_function=model_cfg['activation_function'],
        )
        
        self.tsfm_model = GPT2LMHeadModel(config)
        
        # Apply head masking if needed
        self._mask_attention_heads(expected_n_head, self.desired_n_head)
    
    def _mask_attention_heads(self, expected_n_head, desired_n_head):
        """
        Mask attention heads if there's a dimensional mismatch.
        Keep first 'desired_n_head' heads active, zero out the rest.
        """
        if expected_n_head == desired_n_head:
            print(f"No masking needed: {expected_n_head} heads match desired {desired_n_head}")
            return
        
        heads_to_mask = expected_n_head - desired_n_head
        print(f"Masking {heads_to_mask} heads: keeping heads 0-{desired_n_head-1}, masking heads {desired_n_head}-{expected_n_head-1}")
        
        # For each transformer layer
        for layer_idx, layer in enumerate(self.tsfm_model.transformer.h):
            # Create mask for attention output projection
            mask = t.ones(self.n_embd, self.n_embd, dtype=t.float32)
            
            # Calculate head boundaries and mask unwanted heads
            head_size = self.n_embd // expected_n_head  # d_head
            
            for head_idx in range(desired_n_head, expected_n_head):
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                
                # Zero out columns corresponding to this head in the mask
                mask[:, start_idx:end_idx] = 0
            
            # Register mask as a buffer (non-parameter, part of model state)
            self.register_buffer(f'attn_mask_{layer_idx}', mask)
    
    def _apply_attention_masks(self):
        """Apply masks to attention projection weights before forward pass."""
        for layer_idx, layer in enumerate(self.tsfm_model.transformer.h):
            mask_name = f'attn_mask_{layer_idx}'
            if hasattr(self, mask_name):
                mask = getattr(self, mask_name)
                # Apply mask to the attention output projection weights
                layer.attn.c_proj.weight.data = layer.attn.c_proj.weight.data * mask
    
    def forward(self, src):
        # Apply attention masks before forward pass
        self._apply_attention_masks()
        
        # GPT2LMHeadModel handles embedding and lm_head internally
        transformer_output = self.tsfm_model(src)
        # Get the logits (already includes lm_head)
        logits = transformer_output.logits
        return logits


class HookedTransformerModel(nn.Module):
    def __init__(self, config_path):
        super(HookedTransformerModel, self).__init__()
        
        # Load config from YAML file
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        model_cfg = cfg['model']
        
        # Create HookedTransformerConfig from YAML config
        hooked_config = HookedTransformerConfig(
            n_layers=model_cfg['n_layer'],
            d_model=model_cfg['n_embd'],
            d_head=model_cfg['d_head'],
            n_heads=model_cfg['n_head'],
            d_vocab=model_cfg['vocab_size'],
            n_ctx=model_cfg['n_ctx'],
            act_fn=model_cfg['activation_function'],
            normalization_type="LN",  # Layer norm
            device="cpu"  # Will be moved to correct device later
        )
        
        # Create the HookedTransformer
        self.model = HookedTransformer(hooked_config)
    
    def forward(self, src):
        """Forward pass compatible with your training loop"""
        # HookedTransformer expects input tokens and returns logits
        return self.model(src)
    
    def to(self, device):
        """Override to() to properly move HookedTransformer to device"""
        super().to(device)
        self.model = self.model.to(device)
        return self
    
    def state_dict(self):
        """Return the HookedTransformer's state dict"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into HookedTransformer"""
        return self.model.load_state_dict(state_dict)
    
    def parameters(self):
        """Return HookedTransformer parameters for optimizer"""
        return self.model.parameters()
    
    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        self.model.eval()
        return self

if __name__ == "__main__":
    # Test with YAML config
    model = HookedTransformerModel("config/hook_transformer_config.yaml")
    # model = NNsight(model)
    print(model)
