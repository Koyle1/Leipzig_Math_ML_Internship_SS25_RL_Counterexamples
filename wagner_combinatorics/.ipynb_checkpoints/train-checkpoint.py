# Code to accompany the paper "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner
# Code for conjecture 2.1, without the use of numba 
#
# Please keep in mind that I am far from being an expert in reinforcement learning. 
# If you know what you are doing, you might be better off writing your own code.
#
# This code works on tensorflow version 1.14.0 and python version 3.6.3
# It mysteriously breaks on other versions of python.
# For later versions of tensorflow there seems to be a massive overhead in the predict function for some reason, and/or it produces mysterious errors.
# Debugging these was way above my skill level.
# If the code doesn't work, make sure you are using these versions of tf and python.
#
# I used keras version 2.3.1, not sure if this is important, but I recommend this just to be safe.

## External imports

import numpy as np
import pickle
import time
import datetime
import argparse

## Internal imports
from src.model import get_model
from src.environment import generate_session
from src.util import select_percentile, select_super_sessions, RANDOM_SEEDS

## External parameters
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", help="run number", type=int)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

## Internal parameters


N = 19   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.0001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
N_SESSIONS =1000 #number of new sessions per iteration
PERCENTILE = 93 #top 100-X percentile we are learning from
SUPER_PERCENTILE = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

N_ACTIONS = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
			  #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
			  
OBSERVATION_SPACE = 2*MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
						  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
						  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
						  #Is there a better way to format the input to make it easier for the neural network to understand things?
						  
LEN_GAME = MYN 
STATE_DIM = (OBSERVATION_SPACE,)
REWARD_THRESHOLD = 0

INF = 1000000

# TODO: All required output, correct ending criteria

def train():
    tock = time.time()
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_name_identifier = f"{datetime_string}_run{args.run}"

    super_states =  np.empty((0,LEN_GAME,OBSERVATION_SPACE), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0
    max_reward = REWARD_THRESHOLD - 1 # Initialize max_reward smaller than reward threshold

    # get numpy rng
    try:
        rng = np.random.default_rng(seed=RANDOM_SEEDS[args.run])
    except:
        print("no random seed found for given run number, proceeding with default seed")
        rng = np.random.default_rng(seed=42)
    
    model = get_model(FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS, OBSERVATION_SPACE, LEARNING_RATE, verbose=args.verbose)

    for i in range(1000000): #1000000 generations should be plenty
        #generate new sessions
        #performance can be improved with joblib
        tic = time.time()
        sessions = generate_session(model, N_SESSIONS, OBSERVATION_SPACE, LEN_GAME, MYN, N, rng, args.verbose) #change 0 to 1 to print out how much time each step in generate_session takes 
        sessgen_time = time.time()-tic
        tic = time.time()
        
        states_batch = np.array(sessions[0], dtype = int)
        actions_batch = np.array(sessions[1], dtype = int)
        rewards_batch = np.array(sessions[2])
        states_batch = np.transpose(states_batch,axes=[0,2,1])
        
        states_batch = np.append(states_batch, super_states, axis=0)

        if i>0:
            actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
        rewards_batch = np.append(rewards_batch,super_rewards)
            
        randomcomp_time = time.time()-tic 
        tic = time.time()

        elite_states, elite_actions, _ = select_percentile(states_batch, actions_batch, rewards_batch, percentile=PERCENTILE) #pick the sessions to learn from
        select1_time = time.time()-tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=SUPER_PERCENTILE) #pick the sessions to survive
        select2_time = time.time()-tic
        
        tic = time.time()
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
        select3_time = time.time()-tic
        
        tic = time.time()
        model.fit(elite_states, elite_actions) #learn from the elite sessions
        fit_time = time.time()-tic
        
        tic = time.time()
        
        super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
        super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
        super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
        
        rewards_batch.sort()
        mean_all_reward = np.mean(rewards_batch[-100:])	
        mean_best_reward = np.mean(super_rewards)
        max_reward = np.max(super_rewards)

        score_time = time.time()-tic

        print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
        if args.verbose: print("Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
        
        
        if (i%20 == 1): #Write all important info to files every 20 iterations
            with open(f'out/{file_name_identifier}_best_species_pickle.pkl', 'wb') as fp:
                pickle.dump(super_actions, fp)
            with open(f'out/{file_name_identifier}_best_species.txt', 'w') as f:
                for item in super_actions:
                    f.write(str(item))
                    f.write("\n")
            with open(f'out/{file_name_identifier}_best_species_rewards.txt', 'w') as f:
                for item in super_rewards:
                    f.write(str(item))
                    f.write("\n")
            with open(f'out/{file_name_identifier}_best_100_rewards.txt', 'a') as f:
                f.write(str(mean_all_reward)+"\n")
            with open(f'out/{file_name_identifier}_best_elite_rewards.txt', 'a') as f:
                f.write(str(mean_best_reward)+"\n")
            with open(f'out/{file_name_identifier}_interation_runtime.txt', 'a') as f:
                f.write(f"{i}: {str(sessgen_time+randomcomp_time+select1_time+select2_time+select3_time+fit_time+score_time)}\n")
        if (i%200 == 2): # To create a timeline, like in Figure 3
            with open(f'out/{file_name_identifier}_best_species_timeline.txt', 'a') as f:
                f.write(str(super_actions[super_rewards.index(max_reward)]))
                f.write("\n")
        
        if REWARD_THRESHOLD <= max_reward: ## Break if a solution has been found.
            soltution_state = super_states[super_rewards.index(max_reward)]
            with open(f'out/{file_name_identifier}_solution.txt', 'w') as f:
                f.write("Solution state:\n")
                f.write(str(soltution_state))
                f.write(f"\nSolution reward: {max_reward}\n")
                f.write(f"Generations needed: {N_SESSIONS*i}\n")
                f.write(f"Total runtime: {str(time.time()-tock)}")
            model.save_weights('out/{file_name_identifier}_best_policy.pkl')
            print("Solution found!")
            break

if __name__ == "__main__":
	train()