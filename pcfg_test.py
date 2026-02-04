#%% Import HMM
import numpy as np
import networkx as nx
import itertools

def from_cyllinder_graph(n: int, depth: int, int_range: int = 2, p: float = 0.1, seed: int = 42) -> np.ndarray:
    # Construct the cylinder graph with unidirectional "height" edges and bidirectional "flat" edges
    if seed is not None:
        np.random.seed(seed)
    G = nx.DiGraph()
    for i, (l, j) in enumerate(itertools.product(range(depth), range(n))):
        G.add_node(i, depth=l, position=j)
    
    for l in range(depth):
        for j in range(n):
            # NN and NNN transitions
            current_node = l * n + j
            NN_node = l * n + (j + 1) % n
            NNN_node = l * n + (j + 2) % n
            next_layer_node = ((l + 1) % depth) * n + j
            random_weight = np.random.uniform(0, 1-p)
            G.add_edge(current_node, NN_node, weight=random_weight)
            G.add_edge(current_node, NNN_node, weight=1 - p - random_weight)
            G.add_edge(current_node, next_layer_node, weight=p)
    
    return G

G = from_cyllinder_graph(n=6, depth=3, p=0.1, seed=42)

#%%
import matplotlib.pyplot as plt

# Draw the graph with node labels and edge weights
pos = nx.circular_layout(G)  # Circular layout for better visibility of directed edges

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', arrows=True, font_size=10)

# Draw edge labels (weights)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title('Cylinder Graph Visualization')
plt.axis('off')
plt.show()

#%% Set tokens and transition amplitudes
# Take Dirichlet alpha 1/2 for round metrics on Categorical distributions
def _add_tokens_and_transition_amps(G: nx.DiGraph, depth: int, n: int, tokens_per_cluster: int = 16, dirichlet_alpha: float = 0.5) -> nx.DiGraph:
    assert G.number_of_nodes() == depth * n, "Input graph not consistent with metadata"
    d_vocab = tokens_per_cluster * depth
    for node, attrs in G.nodes(data=True):
        token_dist = np.random.dirichlet(np.ones(tokens_per_cluster) * dirichlet_alpha)
        token_dist_all = np.zeros(d_vocab)
        token_dist_all[attrs['depth'] * tokens_per_cluster:(attrs['depth'] + 1) * tokens_per_cluster] = token_dist
        attrs['token_dist'] = token_dist_all
    
    return G

G = _add_tokens_and_transition_amps(G, depth=3, n=6, tokens_per_cluster=16, dirichlet_alpha=0.3)

#%%
G.nodes()[0]


#%% Get the emission matrices from graph...
from jaxtyping import Float
import einops
def emission_mat_from_graph(G: nx.DiGraph) -> Float[np.ndarray, "d_vocab num_hidden_states num_hidden_states"]:
    _state_transition = nx.adjacency_matrix(G, weight='weight').todense()
    d_vocab = G.nodes(data=True)[0]['token_dist'].shape[0]
    _emission = np.zeros((d_vocab, G.number_of_nodes()))
    for i in range(G.number_of_nodes()):
        _emission[:, i] = G.nodes(data=True)[i]['token_dist']
    
    tot_emission_matrix = einops.einsum(
        _state_transition, _emission,
        "current_state next_state, d_vocab current_state -> d_vocab current_state next_state"
    )
    return tot_emission_matrix

print(emission_mat_from_graph(G).shape)
#%% Generate sample data from HMM
from src.hmm import HMM

class CylinderGraphHMM(HMM):
    """HMM based on a cylinder graph structure with depth and nodes per level."""

    def __init__(
        self,
        n: int,
        depth: int,
        tokens_per_cluster: int = 16,
        dirichlet_alpha: float = 0.5,
        p: float = 0.1,
        int_range: int = 2,
        seed: int = None
    ):
        """
        Initialize a Cylinder Graph HMM.

        Args:
            n: Number of nodes per level (width of cylinder)
            depth: Number of levels (height of cylinder)
            tokens_per_cluster: Number of tokens per depth level
            dirichlet_alpha: Concentration parameter for Dirichlet prior on token distributions
            p: Probability of transitioning to next layer (vs staying in current layer)
            int_range: Integer range parameter (currently unused in cylinder graph)
            seed: Random seed for reproducibility
        """
        self.n = n
        self.depth = depth
        self.tokens_per_cluster = tokens_per_cluster
        self.dirichlet_alpha = dirichlet_alpha
        self.p = p
        self.seed = seed

        # Build the complete HMM
        self.graph = self._build_cylinder_graph()

    def _build_cylinder_graph(self) -> nx.DiGraph:
        """Construct the cylinder graph with token distributions."""
        if self.seed is not None:
            np.random.seed(self.seed)

        G = nx.DiGraph()

        # Add nodes
        for i, (l, j) in enumerate(itertools.product(range(self.depth), range(self.n))):
            G.add_node(i, depth=l, position=j)

        # Add edges with weights
        for l in range(self.depth):
            for j in range(self.n):
                # NN and NNN transitions within layer
                current_node = l * self.n + j
                NN_node = l * self.n + (j + 1) % self.n
                NNN_node = l * self.n + (j + 2) % self.n
                next_layer_node = ((l + 1) % self.depth) * self.n + j

                random_weight = np.random.uniform(0, 1 - self.p)
                G.add_edge(current_node, NN_node, weight=random_weight)
                G.add_edge(current_node, NNN_node, weight=1 - self.p - random_weight)
                G.add_edge(current_node, next_layer_node, weight=self.p)

        # Add token distributions to nodes
        self._add_token_distributions(G)

        return G

    def _add_token_distributions(self, G: nx.DiGraph) -> None:
        """Add token emission distributions to each node."""
        d_vocab = self.tokens_per_cluster * self.depth

        for node, attrs in G.nodes(data=True):
            # Sample from symmetric Dirichlet for this cluster
            token_dist = np.random.dirichlet(np.ones(self.tokens_per_cluster) * self.dirichlet_alpha)

            # Embed into full vocabulary (only non-zero for this depth's tokens)
            token_dist_all = np.zeros(d_vocab)
            start_idx = attrs['depth'] * self.tokens_per_cluster
            end_idx = (attrs['depth'] + 1) * self.tokens_per_cluster
            token_dist_all[start_idx:end_idx] = token_dist

            attrs['token_dist'] = token_dist_all

    def _get_emission_matrix(self) -> Float[np.ndarray, "d_vocab num_hidden_states num_hidden_states"]:
        """Compute emission matrix from graph structure."""
        # Get state transition matrix
        state_transition = nx.adjacency_matrix(self.graph, weight='weight').todense()

        # Get emission matrix (d_vocab x num_states)
        d_vocab = self.graph.nodes(data=True)[0]['token_dist'].shape[0]
        emission = np.zeros((d_vocab, self.graph.number_of_nodes()))
        for i in range(self.graph.number_of_nodes()):
            emission[:, i] = self.graph.nodes(data=True)[i]['token_dist']

        # Combine: P(token, next_state | current_state)
        tot_emission_matrix = einops.einsum(
            state_transition, emission,
            "current_state next_state, d_vocab current_state -> d_vocab current_state next_state"
        )
        return tot_emission_matrix
    
    @property
    def emission_matrices(self):
        return self._get_emission_matrix()

proc = CylinderGraphHMM(n=6, depth=3, tokens_per_cluster=16, dirichlet_alpha=0.3, p=0.1, seed=42)

#%%
# stationary_distribution = proc.get_stationary_distribution()
proc.entropy_rate_empirical_estimate(10000, burn_in=1000)


#%% Load some data...

import torch as t
data = t.load(f"data/datasets/cylinder_graph_hmm/train/observations.pt")
print(data.shape)