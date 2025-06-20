import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
import torch
import os
import math
import datetime
from scipy.linalg import eigh


# Constants
N = 19
MAX_EDGES = int(N * (N - 1) / 2)
OBSERVATION_SIZE = N * N + 2  # adjacency + current_node_scalar + edge_index_scalar


def fiedler_value_path_graph(n):
    return 2 * (1 - math.cos(math.pi / n))


def calc_reward_nx(G: nx.Graph, penalty: float = 0.0, save_dir: str = None,
                   end_of_note: bool = False, action: int = 0, last_reward: int = -20):
    N_graph = G.number_of_nodes()
    if N_graph < 4:
        return 0

    if not end_of_note and action < 0.1:
        return last_reward

    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_eigvals = np.linalg.eigvalsh(A)
    lambda_1 = adj_eigvals[-1] if len(adj_eigvals) > 0 else 0.0

    try:
        mu = len(nx.max_weight_matching(G, maxcardinality=True))
    except Exception:
        mu = 0

    reward = math.sqrt(N_graph - 1) + 1 - lambda_1 - mu

    if reward > 1e-12:
        reward = 10

    if reward == 10 and save_dir is not None and N_graph > 4:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/graph_reward_{reward:.3f}_n{N_graph}_e{G.number_of_edges()}_{timestamp}.graphml"
        nx.write_graphml(G, filename)

    return reward


class GraphNodeBuildEnv(gym.Env):
    def __init__(self, render_mode=None, use_surrogate=False, surrogate_model=None, save_dir="saved_states"):
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

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.min_fiedler = [0.0] * (N + 1)
        for i in range(2, N + 1):
            self.min_fiedler[i] = fiedler_value_path_graph(i)

        # Lagrange multiplier for double descent
        self.lambda_dual = 1.0

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

        if self.use_surrogate and self.surrogate_model is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                reward = self.surrogate_model(obs_tensor).item()
                self.last_reward = reward
                base_reward = reward
        else:
            base_reward = calc_reward_nx(
                self.graph,
                save_dir=self.save_dir,
                action=action,
                end_of_note=(self.current_edge_idx == self.current_node),
                last_reward=self.last_reward
            )

        # --- penalty based on inverse of Fiedler value (1 / λ₂) ---
        try:
            L = nx.laplacian_matrix(self.graph).astype(float).todense()
            lap_eigvals = np.linalg.eigvalsh(L)
            fiedler = lap_eigvals[1] if len(lap_eigvals) > 1 else 0.0
        except Exception:
            fiedler = 0.0

        P = 1.0 / (fiedler + 1e-3) if fiedler > 0 else 1e3  # capped penalty

        # --- shaped reward ---
        reward = base_reward - self.lambda_dual * P
        self.last_reward = reward
        self.cumulative_reward += reward

        if reward == 10:
            terminated = True

        if self.current_node >= 4 and reward > self.max_reward:
            self.max_reward = reward

        self.average = reward if self.average == -22 else 0.1 * reward + 0.9 * self.average
        self._update_observation()

        info = {
            "P": P,
            "lambda": self.lambda_dual,
            "lambda_2": fiedler,
        }

        if terminated:
            info["final_graph_reward"] = base_reward
            info["episode"] = {
                "r": self.cumulative_reward,
                "l": self.current_node,
                "average": (self.cumulative_reward / self.current_node)
            }
            info["final_step_reward"] = self.max_reward if self.current_node >= 4 else reward

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/final_graph_n{self.graph.number_of_nodes()}_e{self.graph.number_of_edges()}_{timestamp}.graphml"
            nx.write_graphml(self.graph, filename)

            self.max_reward = -np.inf
            self.cumulative_reward = 0.0

        return self.obs, reward, terminated, truncated, info

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
        adj_padded = np.zeros((N, N), dtype=np.float32)
        for u, v in self.graph.edges():
            adj_padded[u, v] = 1.0
            adj_padded[v, u] = 1.0
        adj_flat = adj_padded.flatten()

        node_scalar = np.array([self.current_node / N], dtype=np.float32)
        edge_scalar = np.array([self.current_edge_idx / (N - 1)], dtype=np.float32)
        self.obs = np.concatenate([adj_flat, node_scalar, edge_scalar])

    def render(self, mode="human"):
        print(f"Step: Node {self.current_node}, Edge {self.current_edge_idx}")
        print(f"Edges: {self.graph.edges()}")

    def close(self):
        pass
