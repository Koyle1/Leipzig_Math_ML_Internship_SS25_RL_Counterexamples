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

def calc_reward_nx(G: nx.Graph, penalty: float = 0.0):
    if len(G) == 0:
        return penalty

    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)

    D = np.diag(A.sum(axis=1))
    L = D - A

    lap_eigvals = np.linalg.eigvalsh(L)
    lambda_2 = lap_eigvals[1] if len(lap_eigvals) > 1 else 0.0  # algebraic connectivity

    adj_eigvals = np.linalg.eigvalsh(A)
    lambda_1 = adj_eigvals[-1] if len(adj_eigvals) > 0 else 0.0

    try:
        mu = len(nx.max_weight_matching(G, maxcardinality=True))
    except Exception:
        mu = 0

    connected_boost = np.log1p(lambda_2)

    if (-lambda_1 - mu) > 0:
        return 1_000_000

    reward = -lambda_1 - mu + connected_boost
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

        # Reward for each partial graph
        if self.use_surrogate and self.surrogate_model is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                reward = self.surrogate_model(obs_tensor).item()
        else:
            reward = calc_reward_nx(self.graph, self.penalty)

        self.average = reward if self.average == -22 else 0.1 * reward + 0.9 * self.average

        self._update_observation()

        info = {}
        if terminated:
            info["episode"] = {
                "r": reward,
                "l": self.current_node,
                "average": self.average
            }

        return self.obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step: Node {self.current_node}, Edge {self.current_edge_idx}")
        print(f"Edges: {self.graph.edges()}")

    def close(self):
        pass
