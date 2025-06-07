import numpy as np
import networkx as nx
from math import sqrt, floor, cos, pi, ceil
from numpy.linalg import eigvals

def proximity(G):
    dist = nx.floyd_warshall_numpy(G)
    return np.min(np.sum(dist, axis=1)) / (G.number_of_nodes() - 1)

def dist_eigenvalue(G, n):
    D = nx.floyd_warshall_numpy(G)
    eigenvals = sorted(np.linalg.eigvals(D).real, reverse=True)
    return eigenvals[n - 1]

def graph_spectrum(G):
    A = nx.adjacency_matrix(G).astype(float).todense()
    return sorted(np.linalg.eigvals(A).real, reverse=True)

def matching_size(G):
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)

# --- Score Functions ---

def Conj1_score(G):
    n = G.number_of_nodes()
    return sqrt(n - 1) + 1 - max(graph_spectrum(G)) - matching_size(G)

def Conj2_score(G):
    return -(proximity(G) + dist_eigenvalue(G, ceil(2 * nx.diameter(G) / 3)))