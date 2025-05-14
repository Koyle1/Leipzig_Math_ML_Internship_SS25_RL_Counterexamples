from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from model import Model_PPO
from graphEnv import GraphConstructionEnv
from seqGraphEnv import SeqGraphConstructionEnv


print("Training starts...")
env = make_vec_env(SeqGraphConstructionEnv, n_envs=8)
m = Model_PPO("MlpPolicy", env, verbose=1)
try:
    m.load_weights(source="seq_weights.pth")
except:
    print("Prior weight not found")
m.model_train(save_path="seq_weights.pth")
m.save("ppo_graph_constructor")