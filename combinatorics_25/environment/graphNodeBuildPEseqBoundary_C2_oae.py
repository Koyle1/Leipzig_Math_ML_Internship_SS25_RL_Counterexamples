import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
import torch

# Constants
N = 30
MAX_EDGES = int(N * (N - 1) / 2)
OBSERVATION_SIZE = N * N + 2  # adjacency matrix + current_node_scalar + edge_index_scalar



import networkx as nx
import numpy as np
import math
import datetime
from math import log1p
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

import os


def boundary_function(boundary: float = 0.0, fielder_score: float = 0.0):
    return np.exp((boundary - fielder_score) * 10.0)

def fiedler_value_path_graph(N):
    # Create a path graph with N nodes
    G = nx.path_graph(N)
    
    # Get the Laplacian matrix (as a dense NumPy array)
    L = nx.laplacian_matrix(G).toarray()
    
    # Compute all eigenvalues of the Laplacian
    eigenvalues = eigh(L, eigvals_only=True)
    
    # Sort eigenvalues to get the second smallest (Fiedler value)
    fiedler_val = sorted(eigenvalues)[1]
    
    return fiedler_val

'''
def calc_reward_nx(G: nx.Graph, fiedler_score: dict[int, float], penalty: float = 0.0, save_dir: str = "saved_states_c2", end_of_note: bool = False):
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return 0.0

    try:
        # Laplacian eigenvalues
        L = nx.laplacian_matrix(G).astype(float).todense()
        lap_eigvals = np.linalg.eigvalsh(L)
        fiedler_value = lap_eigvals[1]
    except np.linalg.LinAlgError:
        return -5.0

    try:
        dist_matrix = nx.floyd_warshall_numpy(G)
        dist_matrix += 1e-6 * np.eye(N_graph)  # Regularization

        avg_dists = np.sum(dist_matrix, axis=1) / (N_graph - 1)
        proximity = np.min(avg_dists)

        dist_eigvals = eigh(dist_matrix, eigvals_only=True)
        dist_eigvals = np.sort(dist_eigvals)[::-1]

        D = nx.diameter(G)
        k = min(math.floor(2 * D / 3), len(dist_eigvals) - 1)
        partial_eig = dist_eigvals[k]
    except Exception:
        return -5.0  # Gracefully degrade if eigs fail

    # Use provided fiedler bound or 0.0 if missing
    min_fied = fiedler_score[N_graph]
    boundary = boundary_function(min_fied, fiedler_value)

    alpha = 1.0
    if fiedler_value < 1e-12 and end_of_note:
        alpha += 0.05

    reward = - (proximity + partial_eig) - alpha * boundary

    if reward > 1e-6 and fiedler_value > 1e-12:
        reward = 10

    if reward == 10 and save_dir and N_graph > 4:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N_graph}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return reward
'''


#Version 2

'''
def safe_fiedler_value(L, N_graph):
    try:
        if N_graph >= 6:
            # Use sparse eigsh for speed
            lap_eigvals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
            return float(sorted(lap_eigvals)[1])
        else:
            # Fallback to full dense eigval
            lap_dense = L.toarray()
            lap_eigvals = np.linalg.eigvalsh(lap_dense)
            return float(sorted(lap_eigvals)[1])
    except Exception:
        return None

def calc_reward_nx(
    G: nx.Graph,
    fiedler_score: dict[int, float],
    penalty: float = 0.0,
    save_dir: str = "saved_states_c2",
    end_of_note: bool = False
):
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return 0.0

    # --- Fiedler value ---
    try:
        L = nx.laplacian_matrix(G).astype(float)
        fiedler_value = safe_fiedler_value(L, N_graph)
        if fiedler_value is None:
            return -5.3
    except Exception:
        return -5.1

    # --- Distance-based features ---
    try:
        dist_matrix = nx.floyd_warshall_numpy(G)
        dist_matrix += 1e-6 * np.eye(N_graph)

        avg_dists = np.sum(dist_matrix, axis=1) / (N_graph - 1)
        proximity = np.min(avg_dists)

        dist_eigvals = eigh(dist_matrix, eigvals_only=True)
        dist_eigvals = np.sort(dist_eigvals)[::-1]

        D = nx.diameter(G)
        k = min(math.floor(2 * D / 3), len(dist_eigvals) - 1)
        partial_eig = dist_eigvals[k]
    except Exception:
        return -5.2

    # --- Reward Calculation ---
    min_fied = fiedler_score[N_graph]
    boundary = boundary_function(min_fied, fiedler_value)

    alpha = 1.0
    if fiedler_value < 1e-12 and end_of_note:
        alpha += 0.05

    reward = - (proximity + partial_eig) - alpha * boundary

    if reward > 1e-6 and fiedler_value > 1e-12:
        reward = 10

    if reward == 10 and save_dir and N_graph > 4:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N_graph}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return reward
'''

# Version 3

def soft_boundary_transition(x, threshold=1e-6, sharpness=100.0):
    # Smooth transition from 0 to 1 around threshold
    return 1 / (1 + np.exp(-sharpness * (x - threshold)))

def calc_reward_nx(
    G: nx.Graph,
    fiedler_score: dict[int, float],
    penalty: float = 0.0,
    save_dir: str = "saved_states_c2",
    end_of_note: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    
    
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return alpha, beta, 0.0, 0.0

    if not nx.is_connected(G):
        # Smooth penalty for disconnected graphs
        if end_of_note:
            alpha += 0.01
    
        return alpha, beta, -15, -100

    

    # Laplacian Fiedler value (2 smallest eigenvalues)
    L = nx.laplacian_matrix(G).astype(float)
    eigvals = eigsh(L, k=2, which="SM", return_eigenvectors=False)
    fiedler_value = eigvals[1] if eigvals.size > 1 else 0.0

    # Distance matrix from adjacency
    adj_matrix = nx.adjacency_matrix(G)
    dist_matrix = csgraph.floyd_warshall(adj_matrix, directed=False)
    '''
    if np.isinf(dist_matrix).any():
        return alpha, beta, -np.tanh(N_graph / 2.0) * beta, -np.tanh(N_graph / 2.0) * beta
    '''

    np.fill_diagonal(dist_matrix, 1e-6)  # Regularization

    avg_dists = dist_matrix.sum(axis=1) / (N_graph - 1)
    proximity = np.min(avg_dists)

    # Eigenvalues of distance matrix
    dist_eigvals = eigh(dist_matrix, eigvals_only=True)
    dist_eigvals = np.sort(dist_eigvals)[::-1]

    D = nx.diameter(G)
    k = min(math.floor(2 * D / 3), len(dist_eigvals) - 1)
    partial_eig = dist_eigvals[k] if k >= 0 else 0.0

    # Boundary penalty (could use e.g. quadratic loss)
    min_fied = fiedler_score[N_graph]
    boundary = np.exp(fiedler_value - min_fied)

    reward = - (proximity + partial_eig) - (4 - 3 * np.exp(-alpha)) * boundary

    true_reward = - (proximity + partial_eig)

    # Soft transition to high reward if base_reward is good and fiedler is non-zero
    

    if true_reward > 0:
        reward = 1000.0

    # Optional save
    if reward > 999.0 and save_dir and N_graph > 4:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N_graph}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return alpha, beta, reward, true_reward


class GraphNodeBuildEnv(gym.Env):
    """
    Builds a graph by adding one node at a time.
    For each new node, decides whether to connect it to each previous node.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, use_surrogate=False, surrogate_model=None):
        super().__init__()

        self.render_mode = render_mode
        self.use_surrogate = use_surrogate
        self.surrogate_model = surrogate_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graph = nx.Graph()
        self.current_node = 1
        self.current_edge_idx = 0

        self.max_nodes = N
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.penalty = -22
        self.average = -22
        self.max_reward = -np.inf
        self.cumulative_reward = 0.0
        self._update_observation()
        
        self.min_fiedler = [0.0] * (N + 1)

        #increase parameters
        self.alpha = 1.0
        self.beta = 1.0

        for i in range(2, N + 1):
            self.min_fiedler[i] = fiedler_value_path_graph(i)
         
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.graph = nx.Graph()
        self.graph.add_node(0)
        self.current_node = 1
        self.current_edge_idx = 0
        self.max_reward = -np.inf
        self.cumulative_reward = 0.0
        self.average = -22
        self._update_observation()
        
        return self.obs, {}

    def _update_observation(self):
        # Step 1: Create padded N x N adjacency matrix
        adj_padded = np.zeros((N, N), dtype=np.float32)
        for u, v in self.graph.edges():
            adj_padded[u, v] = 1.0
            adj_padded[v, u] = 1.0
        adj_flat = adj_padded.flatten()  # Shape: (361,)

        # Step 2: Encode current node index and edge index as normalized scalars
        node_scalar = np.array([self.current_node / N], dtype=np.float32)
        edge_scalar = np.array([self.current_edge_idx / (N - 1)], dtype=np.float32)

        # Step 3: Concatenate into final observation
        self.obs = np.concatenate([adj_flat, node_scalar, edge_scalar])  # Final shape: (363,)

    def step(self, action):
        assert self.action_space.contains(action)

        if self.current_node < N:
            target_node = self.current_edge_idx
            if action == 1:
                self.graph.add_edge(self.current_node, target_node)

            self.current_edge_idx += 1

            if self.current_edge_idx == self.current_node:
                self.graph.add_node(self.current_node)
                self.current_node += 1
                self.current_edge_idx = 0

        terminated = self.current_node == N
        truncated = False

        # Calculate reward at every step (partial graphs)
        if self.use_surrogate and self.surrogate_model is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                reward = self.surrogate_model(obs_tensor).item()
        else:
            self.alpha, self.beta, reward,true_reward = calc_reward_nx(self.graph, fiedler_score = self.min_fiedler, end_of_note=terminated, alpha=self.alpha, beta=self.beta)

        self.cumulative_reward += reward

        if reward == 10:
            terminated = True

        # Track max reward but only after step 4 (i.e., current_node >= 4)
        if self.current_node >= 4:
            if reward > self.max_reward:
                self.max_reward = reward

        self.average = reward if self.average == -22 else 0.1 * reward + 0.9 * self.average

        self._update_observation()

        info = {}
        if terminated:
            info["final_graph_reward"] = reward
            info["alpha"] = self.alpha
            info["beta"] = self.beta
            info["true_reward"] = true_reward
            
            info["episode"] = {
                "r": self.cumulative_reward,  # cumulative reward for entire episode
                "l": self.current_node,
                "average": (self.cumulative_reward / self.current_node)
            }
            info["final_step_reward"] = self.max_reward if self.current_node >= 4 else reward

            # Reset for next episode
            self.max_reward = -np.inf
            self.cumulative_reward = 0.0

        return self.obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step: Node {self.current_node}, Edge {self.current_edge_idx}")
        print(f"Edges: {self.graph.edges()}")

    def close(self):
        return self.alpha, self.beta
