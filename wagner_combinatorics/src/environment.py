import networkx as nx
import numpy as np
import math
import time


BIG_INTEGER = 1000000 # A big integer

def calcScore(state, graph_size):
	"""
	Calculates the reward for a given word. 
	This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
	:param state: the first MYN letters of this param are the word that the neural network has constructed.


	:returns: the reward (a real number). Higher is better, the network will try to maximize this.
	"""	
	
	#Example reward function, for Conjecture 2.1
	#Given a graph, it minimizes lambda_1 + mu.
	#Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
	#There is a lot of run-to-run variance.
	#Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.
	
	#Construct the graph 
	G= nx.Graph()
	G.add_nodes_from(list(range(graph_size)))
	count = 0
	for i in range(graph_size):
		for j in range(i+1,graph_size):
			if state[count] == 1:
				G.add_edge(i,j)
			count += 1
	
	#G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
	if not (nx.is_connected(G)):
		return -BIG_INTEGER
		
	#Calculate the eigenvalues of G
	# Takes approx. O(N^3) time
	# Could be sped up with numba
	evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
	evalsRealAbs = np.zeros_like(evals)
	for i in range(len(evals)):
		evalsRealAbs[i] = abs(evals[i])
	lambda1 = max(evalsRealAbs)
	
	#Calculate the matching number of G
	# Takes O(N^3) time
	# Not feasible to replace this function. 
	maxMatch = nx.max_weight_matching(G)
	mu = len(maxMatch)
		
	#Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
	#We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
	myScore = math.sqrt(graph_size-1) + 1 - lambda1 - mu
	if myScore > 0:
		#You have found a counterexample. Do something with it.
		print("A counterexample has been found! State: ")
		print(state)
		
	return myScore

def generate_session(
		agent,
		n_sessions,
		observation_space,
		len_game,
		MYN,
		graph_size,
		rng,
		verbose=True):
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
	actions = np.zeros([n_sessions, len_game], dtype = int)
	state_next = np.zeros([n_sessions,observation_space], dtype = int)
	prob = np.zeros(n_sessions)
	scores = np.zeros([n_sessions])
	
	
	pred_time = play_time = scorecalc_time = recordsess_time = total_time = 0 # timers

	tock = time.time()

	states[:,MYN,0] = 1 # initialize first action index
	step = 0
	while (step < len_game): # TODO might need index +-1
		# get prediction for next index
		tic = time.time()
		prob = agent.predict(states[:,:,step], batch_size = n_sessions) 
		pred_time += time.time() - tic

		# derive action from model distribution
		tic = time.time()
		actions[:,step] = rng.binomial(1, prob).flatten()
		
		# inizialize next state with current state
		state_next = states[:,:,step]

		# Add current action to next state
		state_next[:,step] = actions[:,step]
		play_time += time.time() - tic

		# Bitshift of action index in state array
		tic = time.time()
		state_next[:,MYN + step] = 0
		if (step < MYN - 1): # regular iteration: update  next state
			state_next[:, MYN + step + 1] = 1
			states[:,:,step + 1] = state_next
			recordsess_time += time.time() - tic
		else: # final iteration
			scores = np.array([calcScore(session, graph_size) for session in state_next], dtype=float)
			scorecalc_time = time.time() - tic
		step += 1

	#If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.	
	total_time = time.time() - tock	
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time) + "; total: " + str(total_time))
	return states, actions, scores