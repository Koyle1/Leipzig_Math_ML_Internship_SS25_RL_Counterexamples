import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
import torch

# Constants
N = 19
MAX_EDGES = int(N * (N - 1) / 2)
OBSERVATION_SIZE = N * N + 2  # adjacency matrix + current_node_scalar + edge_index_scalar

alpha = 1

import networkx as nx
import numpy as np
import math
import datetime
from math import log1p
from scipy.linalg import eigh

def boundary_function(boundary: float = 0.0, fielder_score: float = 0.0):
    return 1 / (10 * fielder_score + 1e-2)

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

def calc_reward_nx(G: nx.Graph, fiedler_score: dict[int, float], penalty: float = 0.0, save_dir: str = "saved_states", action: int = 0, end_of_note: bool = False, last_reward: int = -20):
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return last_reward, last_reward

    if action < 1e-3 and not end_of_note:
        return last_reward, last_reward

    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_eigvals = np.linalg.eigvalsh(A) #sorted list of eigenvalues
    lambda_1 = adj_eigvals[-1] if len(adj_eigvals) > 0 else 0.0

    L = nx.laplacian_matrix(G).astype(float).todense()
    lap_eigvals = np.linalg.eigvalsh(L)
    fiedler_value = lap_eigvals[1]

    alpha = 1.0
    boundary = boundary_function(fiedler_score[N_graph], fiedler_value)

    try:
        mu = len(nx.max_weight_matching(G, maxcardinality=True))
    except Exception:
        mu = 0

    reward = math.sqrt(N_graph - 1) + 1 - lambda_1 - mu - alpha * boundary
    true_reward = math.sqrt(N_graph - 1) + 1 - lambda_1 - mu
    
    if (math.sqrt(N_graph - 1) + 1 - lambda_1 - mu) > 0.000000000001 and N_graph > 4 and fiedler_value > 0.000000000001:
        reward = 10

    if reward == 10 and save_dir is not None and N_graph > 4:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return reward, true_reward

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

        self.last_reward = -20
        
        self.min_fiedler = [0.0] * (N + 1)

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
        self.last_reward = -20
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
        end_of_note = self.current_edge_idx == (self.current_node - 1)
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
        
        reward, true_reward = calc_reward_nx(self.graph, fiedler_score = self.min_fiedler, last_reward=self.last_reward, action=action, end_of_note = end_of_note)
        self.last_reward = reward

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
        pass
