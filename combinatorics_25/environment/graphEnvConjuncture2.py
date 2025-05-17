from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
import random

    

# Constants (same as in original code)
N = 30
MYN = int(N * (N - 1) / 2)
DECISIONS = MYN
OBSERVATION_SPACE = 2 * MYN
INF = 1000000

@njit
def bfs(Gdeg, edgeListG):
    '''
        Calculates shortest path matrix
    '''
    distMat1 = np.zeros((N, N))
    conn = True
    for s in range(N):
        visited = np.zeros(N, dtype=np.int8)
        myQueue = np.zeros(N, dtype=np.int8)
        dist = np.zeros(N, dtype=np.int8)
        startInd = 0
        endInd = 0

        myQueue[endInd] = s
        endInd += 1
        visited[s] = 1

        while endInd > startInd:
            pivot = myQueue[startInd]
            startInd += 1
            for i in range(Gdeg[pivot]):
                if visited[edgeListG[pivot][i]] == 0:
                    myQueue[endInd] = edgeListG[pivot][i]
                    dist[edgeListG[pivot][i]] = dist[pivot] + 1
                    endInd += 1
                    visited[edgeListG[pivot][i]] = 1

        if endInd < N:
            conn = False

        for i in range(N):
            distMat1[s][i] = dist[i]

    return distMat1, conn

@njit
def jitted_calcScore(state, step):
    '''
        Returns the reward for the graph -> positive counterexample
    '''
    if step != MYN:
        return 0
    
    adjMatG = np.zeros((N, N), dtype=np.int8)
    edgeListG = np.zeros((N, N), dtype=np.int8)
    Gdeg = np.zeros(N, dtype=np.int8)
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if state[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
                edgeListG[i][Gdeg[i]] = j
                edgeListG[j][Gdeg[j]] = i
                Gdeg[i] += 1
                Gdeg[j] += 1
            count += 1

    distMat, conn = bfs(Gdeg, edgeListG)
    if not conn:
        return -INF

    diam = np.amax(distMat)
    sumLengths = np.sum(distMat, axis=0)
    evals = -np.sort(-np.linalg.eigvalsh(distMat))
    proximity = np.amin(sumLengths) / (N - 1.0)

    ans = -(proximity + evals[int(2 * diam / 3) - 1])
    if ans > 0:
        print("Graph satisfies conditions")

    return ans


class GraphConstructionEnv(gym.Env):
    """Environment for combinatorial graph construction for Conjecture 2.3"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None, **kwargs):
        super(GraphConstructionEnv, self).__init__()
        
        self.render_mode = render_mode
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBSERVATION_SPACE,), dtype=np.int8)
        self.action_space = spaces.Discrete(2)  # binary: 0 or 1

        self.episode_reward = 0
        self.episode_length = 0

        self.reset()
    
    #TBU FOR EVALUATION
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
        # Observation = [state (MYN), one-hot position (MYN)]
        position = np.zeros(MYN, dtype=np.int8)
        if self.current_step < MYN:
            position[self.current_step] = 1
        self.obs = np.concatenate([self.state, position])

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self.state[self.current_step] = action
        self.current_step += 1
        terminated = self.current_step == MYN
        truncated = False
        reward = 0

        if terminated:
            reward = jitted_calcScore(self.state, self.current_step)

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