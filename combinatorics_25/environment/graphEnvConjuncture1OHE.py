from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random

# Constants
N = 19
MYN = int(N * (N - 1) / 2)
INF = 1_000_000
OBSERVATION_SPACE = MYN * 3  # state + one-hot position + one-hot step index

def calc_score_conjecture_2_1(state, step, penalty):
    if step != MYN:
        return 0

    adjMatG = np.zeros((N, N), dtype=np.int8)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
            count += 1

    G = nx.from_numpy_array(adjMatG)

    if not nx.is_connected(G):
        return penalty  # Reduced penalty for the sake of exploration

    # Largest eigenvalue of adjacency matrix
    eigenvalues = np.linalg.eigvalsh(adjMatG)
    lambda_1 = eigenvalues[-1]

    # Maximum matching size (matching number)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    mu = len(matching)

    reward = -(lambda_1 + mu)
    return reward


class GraphConstructionEnv(gym.Env):
    """Environment for combinatorial graph construction for Conjecture 2.1"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None, **kwargs):
        super(GraphConstructionEnv, self).__init__()

        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(OBSERVATION_SPACE,),
            dtype=np.int8
        )

        self.action_space = spaces.Discrete(2)  # binary: 0 or 1

        self.episode_reward = 0
        self.episode_length = 0

        self.MYN = MYN
        self.average = -22

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.state = np.zeros(MYN, dtype=np.int8)
        self.current_step = 0
        self._update_observation()
        info = {} if options is None else options
        return self.obs, info

    def _update_observation(self):
        # Observation = [state (MYN), one-hot position (MYN), one-hot step (MYN)]
        position = np.zeros(MYN, dtype=np.int8)
        step_one_hot = np.zeros(MYN, dtype=np.int8)
        if self.current_step < MYN:
            position[self.current_step] = 1
            step_one_hot[self.current_step] = 1
        self.obs = np.concatenate([self.state, position, step_one_hot])

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self.state[self.current_step] = action
        self.current_step += 1
        terminated = self.current_step == MYN
        truncated = False
        reward = 0

        if terminated:
            reward = calc_score_conjecture_2_1(self.state, self.current_step, self.average)
            if self.average == -22:
                self.average = reward
            else:
                self.average = 0.1 * reward + 0.9 * self.average

        self._update_observation()
        info = {}
        if reward > 0:
            print(self.obs)

        if terminated or truncated:
            episode_reward = reward
            episode_length = N
            info["episode"] = {
                "r": episode_reward,
                "l": episode_length
            }

        return self.obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}: {self.state[:self.current_step]}")

    def close(self):
        pass
