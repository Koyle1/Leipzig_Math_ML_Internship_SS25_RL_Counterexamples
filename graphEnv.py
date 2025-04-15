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
def jitted_calcScore(state):
    '''
        Returns the reward for the graph -> positive counterexample
    '''
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
    return ans


class GraphConstructionEnv(gym.Env):
    """Environment for combinatorial graph construction for Conjecture 2.3"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(GraphConstructionEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(OBSERVATION_SPACE,), dtype=np.int8)
        self.action_space = spaces.Discrete(2)  # binary: 0 or 1

        self.reset()
    
    #TBU FOR EVALUATION
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.state = np.zeros(MYN, dtype=np.int8)
        self.current_step = 0
        self._update_observation()
        return self.obs

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
        done = self.current_step == MYN
        reward = 0

        if done:
            reward = jitted_calcScore(self.state)

        self._update_observation()
        return self.obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.current_step}: {self.state[:self.current_step]}")

    def close(self):
        pass