import gym
from gym import spaces

class CombinatorialEnv(gym.Env):
    def __init__(self, N):
        self.N = N
        self.DECISIONS = int(N * (N - 1) / 2)
        self.state = np.zeros(self.DECISIONS)
        self.step_count = 0

        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * self.DECISIONS,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.DECISIONS)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(2 * self.DECISIONS)
        obs[:self.DECISIONS] = self.state
        obs[self.DECISIONS + self.step_count] = 1  # One-hot position
        return obs

    def step(self, action):
        self.state[self.step_count] = action
        self.step_count += 1

        done = self.step_count == self.DECISIONS
        reward = 0
        if done:
            reward = calc_score(self.state.astype(int))

        return self._get_obs(), reward, done, {}