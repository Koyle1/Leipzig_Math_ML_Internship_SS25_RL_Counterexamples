import numpy as np
import gym
from gym import spaces
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
def jitted_calcScore(state, current_node):
    '''
    Returns progressive reward for partially built graph,
    and final reward if graph is complete.
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

    # Only include nodes that have been added so far
    active_nodes = current_node

    # If less than 2 nodes, no reward yet
    if active_nodes <= 1:
        return 0.0

    # BFS over subgraph with active_nodes
    distMat = np.zeros((N, N))
    try:
        subGdeg = Gdeg[:active_nodes]
        subEdgeList = edgeListG[:active_nodes]
        distMat, conn = bfs(subGdeg, subEdgeList)
    except:
        return -10.0  # fallback small penalty

    if not conn:
        return -50.0  # disconnected = bad

    diam = np.amax(distMat[:active_nodes, :active_nodes])
    sumLengths = np.sum(distMat[:active_nodes, :active_nodes], axis=0)
    proximity = np.amin(sumLengths) / (active_nodes - 1.0)
    evals = -np.sort(-np.linalg.eigvalsh(distMat[:active_nodes, :active_nodes]))

    # If final graph:
    if active_nodes == N:
        # Placeholder for conjecture check
        satisfies = (diam <= 5) and (evals[0] - evals[1] > 0.5)
        if satisfies:
            return 1000.0
        else:
            return -100.0

    # Progressive reward: encourage low diameter and high connectivity
    shaping = -proximity + (evals[0] if evals.shape[0] > 0 else 0)
    return shaping * 0.1  # scale to keep rewards smaller during build-up

class SeqGraphConstructionEnv(gym.Env):
    """Environment for node-by-node graph construction"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(SeqGraphConstructionEnv, self).__init__()

        self.total_edges = MYN
        self.node = 1
        self.edge_step = 0

        self.max_steps = MYN  # stays same
        self.state = np.zeros(MYN, dtype=np.int8)
        self.edge_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]

        # Maximum number of decisions is N*(N-1)/2, still
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBSERVATION_SPACE,), dtype=np.int8)
        self.action_space = spaces.Discrete(2)  # binary: 0 or 1

        self.reset()

    def reset(self):
        self.state = np.zeros(self.total_edges, dtype=np.int8)
        self.node = 1
        self.edge_step = 0
        self._update_observation()
        return self.obs

    def _update_observation(self):
        # Encode the current node and edge step as one-hot for simplicity
        position = np.zeros(MYN, dtype=np.int8)
        if self.node < N:
            idx = self._current_edge_index()
            position[idx] = 1
        self.obs = np.concatenate([self.state, position])

    def _current_edge_index(self):
        # Map (i, j) to index in edge list
        return self.edge_indices.index((self.edge_step, self.node))

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if self.node >= N:
            raise Exception("All nodes have already been added!")

        idx = self._current_edge_index()
        self.state[idx] = action

        self.edge_step += 1

        done = False
        reward = 0

        if self.edge_step == self.node:
            self.node += 1
            self.edge_step = 0

        if self.node == N:
            done = True
            reward = jitted_calcScore(self.state, idx)

        self._update_observation()
        return self.obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Building Node {self.node}, Edge Step {self.edge_step}")
        print(f"Current state (partial graph): {self.state}")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def close(self):
        pass