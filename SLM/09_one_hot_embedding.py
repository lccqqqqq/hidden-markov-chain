"""
09_one_hot_embedding.py

Quadratic autoregressive model using one-hot embeddings instead of
scalar token values. Tested on the original Z₂ data (blocks of 3:
two random bits + XOR) to verify equivalence with the scalar model.

With vocab size q=2, one-hot gives x_t ∈ {(1,0), (0,1)} ∈ ℝ².
The linear term W[t,i] becomes a (q_out, q_in) matrix.
The quadratic term C[t,i,j] becomes a (q_out, q_in, q_in) tensor —
crucially, it must produce per-output-class logits, not a shared
scalar, since softmax is shift-invariant (adding the same constant
to all logits has no effect).

Companion to version2.md §6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_z2_data(batch_size, T=12):
    """Sample sequences as integer tokens."""
    assert T % 3 == 0
    tokens = torch.zeros(batch_size, T, dtype=torch.long)
    for block in range(T // 3):
        i = block * 3
        tokens[:, i]     = torch.randint(0, 2, (batch_size,))
        tokens[:, i + 1] = torch.randint(0, 2, (batch_size,))
        tokens[:, i + 2] = (tokens[:, i] + tokens[:, i + 1]) % 2
    return tokens


class OneHotQuadraticAutoregressive(nn.Module):
    """
    Quadratic autoregressive model with one-hot token representations.

    Each token x_t ∈ {0, ..., q-1} is one-hot encoded as e_{x_t} ∈ ℝ^q.

    Logit for output class a at position t:
        Δ_t[a] = b[t, a]
                 + Σ_{i<t} Σ_c  W[t, i, a, c] · x_i[c]
                 + Σ_{i<j<t} Σ_{c,d}  C[t, i, j, a, c, d] · x_i[c] · x_j[d]

    The output dimension 'a' on C is essential: softmax is shift-invariant,
    so a quadratic term that adds the same value to all output logits would
    have zero effect. Each output class must get a different contribution
    from the pairwise context.
    """

    def __init__(self, T, q=2):
        super().__init__()
        self.T = T
        self.q = q

        # b[t, a]: bias per position per output class
        self.bias = nn.Parameter(torch.zeros(T, q))

        # W[t, i, a, c]: linear weight — output class a, input token c at position i
        self.W = nn.Parameter(torch.zeros(T, T, q, q))

        # C[t, i, j, a, c, d]: quadratic weight — output class a,
        #   input token c at position i, input token d at position j
        self.C = nn.Parameter(torch.zeros(T, T, T, q, q, q))

        # Position masks
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

    def forward(self, tokens):
        """
        tokens: (B, T) long tensor of token indices
        returns: logits (B, T, q)
        """
        B, T = tokens.shape
        q = self.q

        # One-hot encode: (B, T, q)
        x = F.one_hot(tokens, num_classes=q).float()

        # Bias: (T, q) → (B, T, q)
        logits = self.bias.unsqueeze(0).expand(B, -1, -1).clone()

        # Linear: Σ_{i<t} Σ_c W[t,i,a,c] · x_i[c] → (B, T, q)
        W_masked = self.W * self.linear_mask[:, :, None, None]
        linear = torch.einsum('bic, tiac -> bta', x, W_masked)
        logits = logits + linear

        # Quadratic: Σ_{i<j<t} Σ_{c,d} C[t,i,j,a,c,d] · x_i[c] · x_j[d] → (B, T, q)
        C_masked = self.C * self.quad_mask[:, :, :, None, None, None]
        quadratic = torch.einsum('bic, bjd, tijacd -> bta', x, x, C_masked)
        logits = logits + quadratic

        return logits


def main():
    T = 12
    q = 2
    model = OneHotQuadraticAutoregressive(T, q=q)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    theoretical_min = (2 / 3) * torch.log(torch.tensor(2.0)).item()
    print(f"T = {T}, vocab size q = {q}")
    print(f"Theoretical minimum loss: {theoretical_min:.4f}")
    print()

    for step in range(5001):
        tokens = sample_z2_data(batch_size=512, T=T)
        logits = model(tokens)  # (B, T, q)

        # Cross-entropy loss (softmax over vocab dimension)
        loss = F.cross_entropy(logits.view(-1, q), tokens.view(-1), reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Gap: {loss.item() - theoretical_min:.4f}")

    # --- Check XOR predictions at position 2 ---
    print("\n--- XOR predictions (position 2) ---")
    with torch.no_grad():
        test_tokens = torch.tensor([
            [0, 0, 0] * (T // 3),
            [0, 1, 1] * (T // 3),
            [1, 0, 1] * (T // 3),
            [1, 1, 0] * (T // 3),
        ], dtype=torch.long)
        logits = model(test_tokens)
        probs = torch.softmax(logits[:, 2, :], dim=-1)  # (4, q)
        for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            target = (a + b) % 2
            print(f"  ({a},{b}) → P(x₂=0)={probs[i, 0].item():.4f}, "
                  f"P(x₂=1)={probs[i, 1].item():.4f}  (target: {target})")


if __name__ == '__main__':
    main()
