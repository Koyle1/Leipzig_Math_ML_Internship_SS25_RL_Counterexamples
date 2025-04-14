from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from model import model
from graphEnv import GraphConstructionEnv


print("Training starts...")
env = make_vec_env(GraphConstructionEnv, n_envs=1)
m = model("MlpPolicy", env, verbose=1)
try:
    m.load_weights()
except:
    print("Prior weight not found")
m.model_train()
m.save("ppo_graph_constructor")