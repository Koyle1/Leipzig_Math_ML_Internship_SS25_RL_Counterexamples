import os

# Configuration of WandB: No logs are saved
os.environ['WANDB_DIR'] = '/tmp/wandb_logs'
os.environ['WANDB_SAVE_CODE'] = 'false'
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'        #

from stable_baselines3.common.env_util import make_vec_env # For enviorment parallelisation
from model.explors import ExplorerModel # Reinforcement learning algortihm
import environment.registration # Enviroment
import argparse
import wandb #For training visualisation on WanDB

def main():
    # Define parser arguments for training configuration
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--checkpoint_path", type=str, default="seq_weights.pth", help="Path to a previous model checkpoint.")
    parser.add_argument("--seed_nr", type=int, default=0, help="Choose a seed between [0-19]")
    parser.add_argument("--enviroment", type=str, default="conjecture_1", help="Choose an enviorment")
    parser.add_argument("--model", type=str, default="PPO", help="Choose an algorithm")
    parser.add_argument("--n_env", type=int, default=4, help="number of parallel enviroments")
    parser.add_argument("--stop_on_solution", type=bool, default=False, help="stop the training once a solution is found")
    parser.add_argument("--save_path", type=str, default="./training_run_logs", help="Directory for saving training artifacts")
    
    args = parser.parse_args()

    # Debug print statement
    print("Training starts...")

    #Create wandb session
    wandb.init(project="math_ml",
               config={"env":args.enviroment,
                      "algo":args.model,
                      "seed_nr":args.seed_nr},
               monitor_gym=True,
               save_code=True,
    )

    # Choose one of the twenty seeds used for model evaluation
    seed = get_seed(args.seed_nr)

    #Initialise parallel enviorments with class, number and seed as arguments
    env = make_vec_env(args.enviroment,
                       n_envs=args.n_env,
                       seed=seed)

    # Initialise the model
    m = ExplorerModel(model_name=args.model,
                     env=env,
                     seed=seed,
                     stop_on_solution=args.stop_on_solution,
                     log_dir=args.save_path)
    try:
        # Train from prior weights if available
        m.load_weights(source=args.checkpoint_path)
    except:
        #Otherwise use a newly initialised policy
        print("Prior weight not found")

    # Use the custom training loop defined in ExplorerModel class
    m.model_train()

    #Close wandb session
    wandb.finish()

def get_seed(number: int):
    '''
        The twenty different seeds that were used for testing our implementation.
    '''
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
