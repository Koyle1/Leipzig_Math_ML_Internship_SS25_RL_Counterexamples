import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
import random

# Constants
N = 30
MYN = int(N * (N - 1) / 2)
DECISIONS = MYN
OBSERVATION_SPACE = 2 * MYN
INF = 1000000

@njit
def bfs(Gdeg, edgeListG):
    '''
    Calculates shortest path matrix and connectivity
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
                nxt = edgeListG[pivot][i]
                if visited[nxt] == 0:
                    myQueue[endInd] = nxt
                    dist[nxt] = dist[pivot] + 1
                    endInd += 1
                    visited[nxt] = 1

        if endInd < N:
            conn = False

        for i in range(N):
            distMat1[s][i] = dist[i]

    return distMat1, conn

@njit
def jitted_calcScore(state, current_node):
    '''
    Progressive and final reward for graph built up to current_node
    '''
    adjMatG = np.zeros((N, N), dtype=np.int8)
    edgeListG = np.zeros((N, N), dtype=np.int8)
    Gdeg = np.zeros(N, dtype=np.int8)
    count = 0

    for i in range(N):
        for j in range(i + 1, N):
            if state[count] == 1:
                adjMatG[i][j] = 1
                adjMatG[j][i] = 1
                edgeListG[i][Gdeg[i]] = j
                edgeListG[j][Gdeg[j]] = i
                Gdeg[i] += 1
                Gdeg[j] += 1
            count += 1

    active_nodes = current_node

    if active_nodes <= 1:
        return 0.0

    try:
        subGdeg = Gdeg[:active_nodes]
        subEdgeList = edgeListG[:active_nodes]
        distMat, conn = bfs(subGdeg, subEdgeList)
    except:
        return -10.0  # fallback penalty

    if not conn:
        return -50.0  # disconnected penalty

    diam = np.amax(distMat[:active_nodes, :active_nodes])
    sumLengths = np.sum(distMat[:active_nodes, :active_nodes], axis=0)
    proximity = np.amin(sumLengths) / (active_nodes - 1.0)
    evals = -np.sort(-np.linalg.eigvalsh(distMat[:active_nodes, :active_nodes]))

    if active_nodes == N:
        satisfies = (diam <= 5) and (evals[0] - evals[1] > 0.5)
        if satisfies:
            print("Graph satisfies conditions")
            return 1000.0
        else:
            return -100.0

    shaping = -proximity + (evals[0] if evals.shape[0] > 0 else 0)
    return shaping * 0.1

class SeqGraphConstructionEnv(gym.Env):
    """Environment for sequential graph construction"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()

        self.render_mode = render_mode
        
        self.total_edges = MYN
        self.node = 1
        self.edge_step = 0

        self.max_steps = MYN
        self.state = np.zeros(MYN, dtype=np.int8)
        self.edge_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SPACE,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # binary: 0 or 1

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.state = np.zeros(self.total_edges, dtype=np.int8)
        self.node = 1
        self.edge_step = 0
        self._update_observation()
        info = {} if options is None else options
        return self.obs, info

    def _update_observation(self):
        position = np.zeros(MYN, dtype=np.int8)
        if self.node < N:
            idx = self._current_edge_index()
            position[idx] = 1
        self.obs = np.concatenate([self.state, position]).astype(np.float32)

    def _current_edge_index(self):
        i = self.node
        j = self.edge_step
        # Ensure i > j since edges stored with i < j
        if i < j:
            i, j = j, i
        return int(i * (i - 1) / 2 + j)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if self.node >= N:
            raise Exception("All nodes have already been added!")

        idx = self._current_edge_index()
        self.state[idx] = action

        self.edge_step += 1

        terminated = False
        truncated = False
        reward = 0

        if self.edge_step == self.node:
            self.node += 1
            self.edge_step = 0

        if self.node == N:
            terminated = True
        
        reward = jitted_calcScore(self.state, self.node)

        self._update_observation()
        if reward > 0:
            print(self.obs)
        info = {}
        return self.obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Building Node {self.node}, Edge Step {self.edge_step}")
        print(f"Current state (partial graph): {self.state}")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def close(self):
        pass
