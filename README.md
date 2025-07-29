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
Wagner applies the cross-entropy method to generate solutions for combinatorics problems.
The method works iteratively and follows the following process:

For graph problems, graphs are represented as a binary string of all possible edges (1 = edge exists, 0 = no edge exists).
In the first step, a neural network (NN) predicts the first decision in the construction.
It outputs a probability distribution for the next decision based on previous ones. 
The sample process continuis step-by-step until a full graph structure is generated.
After generating a full structure, a reward function evaluates the graph structure based on a reward function.
This process is repeated n times, resulting in the selection of the top-performing constructions (e.g. top 5%).
These top-performing constructions are used to update the NN.
The process is repeated until a valid counterexample is found.

## Our Improvements

### Constructing the environment

Wagner uses a simple graph encoding based on a flatted representation of the adjacency matrix.
The first half of the arrray represents the edges of the graph and can be interpreted as teh upper traingle of the graph adjacency matrix.
The second half one-hot encodes the currend action index.

We use two rational numbers to represent the node and edge indices in a more compact form than Wagner.
The graph is not build along the rows of the adjacency matrix but along the columns equating to building the graph node-by-node.
This approach gives the model the ability to find counterexamples to the conjectures smaller than a pre-defined N.

### Considerations regarding the reward function

Our reward function is a combination of four factors: the distance metric, a side condition, a repetition penalty & exploration bonus and an intrinstic reward.
As Wagner does, we take the distance metric as the base of our reward function. To sepcifically reward counterexamples, we give a constant reward of 10 in this case.
A side condition we use is the algebraic connectivity (Fiedler value) using to obtain differentiable reward values that are strictly positive and tend to zero for graphs that are more highly connected than the path graph.
By using repetition penalty & exploration bonus we encourage exploration and penalize repetition.
The intrinsic reward is updated after every complete graph construction and learns to provide a meaningful reward component based on previous graph construction.

## Our results

We evaluated our approach, Wagners approach and the Adaptive Monte Carlos Search approach over 20 runs having defined random seeds.
Our model improves on Wagner

| Approach | Success rate | Avg. sample efficiency | Avg. time till counterexample is found | Stability over 1-3 |
|-------------|-------------|-------------|-------------|-------------|
| Wagner | | | | 
| AMCS | 100% | 12.90 steps | 1.30s | Variance efficiency: 0.09; Variance time: 0.004 |




