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

import networkx as nx
import numpy as np
import math
import datetime
from math import log1p

def calc_reward_nx(G: nx.Graph, penalty: float = 0.0, save_dir: str = "saved_states"):
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return 0

    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_eigvals = np.linalg.eigvalsh(A) #sorted list of eigenvalues
    lambda_1 = adj_eigvals[-1] if len(adj_eigvals) > 0 else 0.0

    L = nx.laplacian_matrix(G).astype(float).todense()
    lap_eigvals = np.linalg.eigvalsh(L)
    fiedler_value = lap_eigvals[1]

    if lambda_1 == 0.0:
        return 0

    if fiedler_value == 0.0:
        return -10

    try:
        mu = len(nx.max_weight_matching(G, maxcardinality=True))
    except Exception:
        mu = 0

    reward = math.sqrt(N_graph - 1) + 1 - lambda_1 - mu + log1p(1 + fiedler_value)
    
    if (math.sqrt(N_graph - 1) + 1 - lambda_1 - mu) > 0 and N_graph == N:
        reward = 10

    if reward == 10 and save_dir is not None and N_graph == N:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return reward

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
            reward = calc_reward_nx(self.graph, self.penalty)

        self.cumulative_reward += reward

        if reward > 900_000:
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
