from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from model import Model
import gymnasium as gym
import enviroment.registration

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--save_path", type=str, default="seq_weights.pth", help="Path to save the model")
    parser.add_argument("--seed_nr", type=int, default=0, help="Choose a seed between [0-19]")
    parser.add_argument("--enviroment", type=str, default="GraphEnv-v0", help="Choose an enviorment")
    parser.add_argument("--model", type=str, default="PPO", help="Choose an algorithm")
    
    args = parser.parse_args()
    
    print("Training starts...")
    #Added aditional 
    env = make_vec_env(args.enviroment,
                       n_envs=8)

    m = Model.create(args.model, 
                     "MlpPolicy",
                     env,
                     verbose=1)
    try:
        m.load_weights(source=args.save_path)
    except:
        print("Prior weight not found")
    m.model_train(save_path=args.save_path)
    m.save("ppo_graph_constructor")

def seed(number: int = 0):
    pass
    
if __name__ == "__main__":
    main()
