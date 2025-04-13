from gymnasium import gym

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()

        # Here we need to specify the differnet possible observations

        N = 19   #number of vertices in the graph. Change if needed
        MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length
        observation_space = 2*MYN

        self.observation_space = spaces.MultiBinary(observation_space) #Binärere Vektor der Länge "observation_space" für One-hot encoding

        # Discrete Actions
        self.action_space = spaces.Discrete(2) #Gibt nur zwei discrete Handlungen

        self.state = None
        self.max_steps = 20
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.random.randint(-10, 10)], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        if action == 0:
            self.state[0] -= 1
        elif action == 1:
            self.state[0] += 1

        done = self.current_step >= self.max_steps or self.state[0] == 0
        reward = 1.0 if self.state[0] == 0 else -0.1  # Belohnung für "Ziel erreicht"

        return self.state, reward, done, False, {}

    def render(self):
        print(f"State: {self.state[0]}")

    def close(self):
        pass
