This project tries to improve upon the approach by Adam Zsolt Wagner (2021) for constructing counterexamples to conjectures in combinatorics using reinforcement learning.

## Project Overview

Wagner showed that reinforcement learning can be used to generate constructions and counterexamples to mathematical conjectures in combinatorics.
For that, he used the *cross-entropy method*. 
In this project, we:

- reproduced and improved Wagner's code on efficiency
- implemented an RL approach based on PPO (Proximal Policy Optimization) and natural graph building
- implemented new evaluation metrics
- compared the results with Wagner's approach and Adaptive Monte Carlo Search (Vito & Stefanus, 2023)

## Wagner's Approach

### The Cross-Entropy Method
Wagne applies the cross-entropy method to generate solutions for combinatorics problems.
The method works iteratively and follows the following process:

For graph problems, graphs are represented as a binary string of all possible edges (1 = edge exists, 0 = no edge exists).
In the first step, a neural network (NN) predicts the first decision in the construction.
It outputs a probabilty distribution for the next decision based on previous ones. 
The sample process continuis step-by-step until a full graph structure is generated.
After generating a full structure, a reward function evaluates the graph structure based on a reward function.
This process is repeated n times, resulting in the selection of the top-performing constructions (e.g. top 5%).
These top-performing constructions are used to update the NN.
The process is repeated until a valid counterexample is found.

### Key Results

## Our Improvements

### Algorithmic Changes

We tested and compared different RL algorithms with the cross-entropy method as baseline:

- Proximal Policy Optimization (PPO): actor-critic architecture. It clips gradients resulting in more stable training
- DQN
- DDQN
- A2C
- AMCS

-> we chose two actor-critic models (PPO and A2C) and two value-based models (DQN and DDQN). Additionally, we chose one heuristic search algorithm (AMCS)

### Problem encoding
Can we improve on the graph representation as a flattened adjacency matrix?

### Reward shaping

## Our results

Evaluation of approaches over 20 runs having defined random seeds.

Conjecture 1:

| Approach | Success rate | Avg. sample efficiency | Avg. time till counterexample is found | Stability over 1-3 |
|-------------|-------------|-------------|-------------|-------------|
| Wagner | | | | 
| AMCS | 100% | 12.90 steps | 1.30s | Variance efficiency: 0.09; Variance time: 0.004 |

Conjecture 2: 

| Approach | Success rate | Avg. sample efficiency | Avg. time till counterexample is found | Stability over 1-3 |
|-------------|-------------|-------------|-------------|-------------|
| Wagner | | | | |
| AMCS | | | | |


