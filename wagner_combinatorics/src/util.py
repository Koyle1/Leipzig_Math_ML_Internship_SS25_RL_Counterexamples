import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# somewhat cumbersome original implementation
def select_elites(states_batch, actions_batch, rewards_batch, n_sessions, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]

	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]:
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = int)	
	elite_actions = np.array(elite_actions, dtype = int)	
	return elite_states, elite_actions

# somewhat cumbersome original implementation
def select_super_sessions(states_batch, actions_batch, rewards_batch, n_sessions, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = int)
	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards

def select_session_percentile(
		states,
		actions,
		rewards,
		n_sessions,
		percentile=50
		):
	"""
	Pythonic version using numpy operations to do all the filtering. Replaces select_elites.
	I don't yet understand select_super_sessions well enought to replace it as well.
	"""
	# We want to select the percentile based only on the number of session generated per generation (e.g. out of 1000)
	# In order to use numpys convenice methods, we therefore have to recalculate our boundary.
	# Example
	# What percentile of n = 1200 has the same size of the 95%ile on n = 1000?, in this case 50?
	session_based_percentile = (1 - (n_sessions * (100 - percentile) / 100) / states.shape[0]) * 100

	reward_threshold = np.percentile(rewards, session_based_percentile)
	filter_index = rewards >= reward_threshold+0.0000001

	filtered_states = states[filter_index]
	filtered_actions = actions[filter_index]

	old_shape = filtered_states.shape
	new_first_dimension = old_shape[0] * old_shape[1]

	return (
		filtered_states.reshape(new_first_dimension, old_shape[2]),
		filtered_actions.reshape(new_first_dimension)
    )
	
def visualize_state_as_graph(state, n_nodes):
    """Print state edge array as a graph visual."""
    G= nx.Graph()
    G.add_nodes_from(list(range(n_nodes)))
    count = 0
    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            if state[count] == 1:
                G.add_edge(i,j)
            count += 1

    nx.draw_kamada_kawai(G)
    plt.show()
	
RANDOM_SEEDS = [
	278485391165970724333648292573411832171,
    2829313939463750814984884989431368559,
    255812181389433800451327842441650200264,
    222190534461520838614358741292077253444,
    123027592070920660996488309663833729338,
    265054478718630852523838193044225179144,
    194916960477164744241242668455265749050,
    163829584347856677719605943273818718754,
    19792102686818527222542248056776694981,
    76543607132690056818837546675621478121,
    67690639562908739701960554995164189103,
    316983907325689536334355486753666674827,
    306006899000145009617743176148662272252,
    304396019548523974838072433834229120258,
    135212709492921225206824863101937531601,
    59707774288202406839869396943770428081,
    214792385022860995250559709259616130020,
    170222376936427941355471218879611434393,
    47280706601717796827643415157352020930,
    248379215267471694824648201709914284579,
]
