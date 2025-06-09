import os
os.environ['WANDB_DIR'] = '/tmp/wandb_logs'
os.environ['WANDB_SAVE_CODE'] = 'false'
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ['WANDB_SAVE_CODE'] = 'false'
os.environ['WANDB_CONSOLE'] = 'off'        # kein CLI-Output speichern

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from model.model import Model
from model.explors import ExplorerModel
import gymnasium as gym
import environment.registration

import argparse

#code added for wandb
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--save_path", type=str, default="seq_weights.pth", help="Path to save the model")
    parser.add_argument("--seed_nr", type=int, default=0, help="Choose a seed between [0-19]")
    parser.add_argument("--enviroment", type=str, default="c2-graphNodeBuildPEseqB-v0", help="Choose an enviorment")
    parser.add_argument("--model", type=str, default="PPO", help="Choose an algorithm")
    parser.add_argument("--n_env", type=int, default=4, help="number of parallel enviroments")
    
    args = parser.parse_args()
    
    print("Training starts...")
    #Added aditional 

    #Create wandb session
    wandb.init(project="math_ml",
               config={"env":args.enviroment,
                      "algo":args.model,
                      "seed_nr":args.seed_nr},
               monitor_gym=True,
               save_code=True,
    )
    
    seed = get_seed(args.seed_nr)
    
    env = make_vec_env(args.enviroment,
                       n_envs=args.n_env)

    policy_kwargs = dict(net_arch=dict(pi=[128, 64, 4], vf=[128, 64, 4])) #Use custom net_arch
    
    m = ExplorerModel(model_name=args.model,
                     env=env,
                     seed=seed)
    try:
        m.load_weights(source=args.save_path)
    except:
        print("Prior weight not found")
    m.model_train()

    #Close wandb session
    wandb.finish()

def get_seed(number: int):
    seeds= [278485391,
            282931393,
            255812181,
            222190534,
            123027592,
            265054478,
            194916960,
            163829584,
            197921026,
            765436071,
            676906395,
            316983907,
            306006899,
            304396019,
            135212709,
            597077742,
            214792385,
            170222376,
            472807066,
            248379215]
    
    return seeds[number]
    
if __name__ == "__main__":
    main()
