"""
Two-state two-token HMM and bilinear Deep Sets model.

The Deep Sets architecture follows section 10.2-10.9 of the LLM embeddings note:
    log p(w_{1:L}) = c - 1/2 (s - d)^T M (s - d),   s = sum_t e_{w_t}
with learnable embeddings e_i, symmetric matrix M, bias d, and constant c.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hmm import HMM


class TwoStateTwoTokenHMM(HMM):
    """
    Two-state, two-token HMM with tunable parameters.

    Transition:  T[i,k] = P(next=k | current=i)
        T = [[1-a, a],
             [a, 1-a]]

    Observation: O[j,i] = P(emit token j | state i)
        O = [[1-eps, eps],
             [eps, 1-eps]]

    Combined:    E[j,i,k] = O[j,i] * T[i,k]
    """

    def __init__(self, eps=0.1, a=0.3):
        self.eps = eps
        self.a = a
        self._E = self._build_emission_matrices()

    def _build_emission_matrices(self):
        a = self.a
        eps = self.eps
        T = np.array([[1 - a, a],
                       [a, 1 - a]])
        O = np.array([[1 - eps, eps],
                       [eps, 1 - eps]])
        E = np.einsum('ji,ik->jik', O, T)
        return E

    @property
    def emission_matrices(self):
        return self._E


class BilinearDeepSets(nn.Module):
    """
    Generic bilinear Deep Sets model for sequence log-probability.

    Architecture:
        1. Embedding lookup: token i -> e_i in R^r
        2. Sum pooling: s = sum_{t=1}^L e_{w_t}
        3. Quadratic energy head: log p = c - 1/2 (s - d)^T M (s - d)

    M is unconstrained symmetric (not forced PSD).
    """

    def __init__(self, V=2, r=1):
        super().__init__()
        self.V = V
        self.r = r
        self.embedding = nn.Embedding(V, r)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        self.M = nn.Parameter(torch.eye(r) * 0.1)
        self.d = nn.Parameter(torch.zeros(r))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Args:
            x: (batch_size, L) integer token sequences
        Returns:
            log_p: (batch_size,) predicted log-probabilities
        """
        emb = self.embedding(x)         # (batch, L, r)
        s = emb.sum(dim=1)              # (batch, r)
        M_sym = 0.5 * (self.M + self.M.T)
        diff = s - self.d               # (batch, r)
        energy = torch.einsum('bi,ij,bj->b', diff, M_sym, diff)
        return self.c - 0.5 * energy

    def init_from_ground_truth(self, embeddings, M, d, c, noise=0.0):
        """Initialize parameters near known ground truth values."""
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor(embeddings, dtype=torch.float32))
            self.M.copy_(torch.tensor(M, dtype=torch.float32))
            self.d.copy_(torch.tensor(d, dtype=torch.float32))
            self.c.fill_(c)
            if noise > 0:
                for p in self.parameters():
                    p.add_(torch.randn_like(p) * noise)
