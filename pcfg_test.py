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
def _add_tokens_and_transition_amps(G: nx.DiGraph, tokens_per_cluster: int = 16, dirichlet_alpha: float = 1.0) -> nx.DiGraph:
    pass
