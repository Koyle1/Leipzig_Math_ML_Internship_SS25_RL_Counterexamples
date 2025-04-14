from gymnasium import gym
from abc import ABC, abstractmethod
import numpy as np

class SimpleEnv(ABC, gym.Env):
    def __init__(self, N : int):
        super(SimpleEnv, self).__init__()

        # Here we need to specify the differnet possible observations

        self.N = N   #number of vertices in the graph. Change if needed
        self.decisions = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length
        observation_space = 2*self.decisions

        self.observation_space = spaces.MultiBinary(observation_space) #Binärere Vektor der Länge "observation_space" für One-hot encoding

        # Discrete Actions
        self.action_space = spaces.Discrete(2) #Gibt nur zwei discrete Handlungen


        #Update later
        self.state = None
        self.max_steps = 20
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #update later
        self.state = None
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        self.take_action(action)

        done = self.current_step >= self.max_steps or self.state[0] == 0
        reward = self.calc_reward(self.state, self.action)

        return self.state, reward, done, False, {}

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def calc_reward(self, state, action):
        pass

    def render(self):
        print(f"State: {self.state[0]}")

    def close(self):
        pass
