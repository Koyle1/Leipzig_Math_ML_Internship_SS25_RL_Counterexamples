import argparse
import datetime
import logging
import os
import random
import time
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
from nmcs import NMCS_connected_graphs, NMCS_trees
import scores

STAGNATION_THRESHOLD = 50
MAX_DEPTH = 500
MAX_LEVEL = 500

def create_output_dir():
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')

def get_seed(run_number):
    seeds = [
        278485391, 282931393, 255812181, 222190534, 123027592,
        265054478, 194916960, 163829584, 197921026, 765436071,
        676906395, 316983907, 306006899, 304396019, 135212709,
        597077742, 214792385, 170222376, 472807066, 248379215
    ]
    return seeds[run_number % len(seeds)]

# Graph utilities
def remove_randleaf(G):
    '''Removes random leaf node from graph.

    Args:
        G (nx.Graph): input graph.

    Returns:
        leaf: removed node or None if no leaf exists.
    '''
    leaves = [v for v in G.nodes if G.degree[v] == 1]
    if not leaves:
        return None
    leaf = random.choice(leaves)
    G.remove_node(leaf)
    return leaf

def remove_subdiv(G):
    '''Removes a subdivision node (degree 2) or a leaf if none exists.

    Args:
        G (nx.Graph): input graph.

    Returns:
        v: removed node or None.
    '''
    deg_2 = [v for v in G.nodes if G.degree[v] == 2]
    if not deg_2:
        return remove_randleaf(G)
    v = random.choice(deg_2)
    neighbors = list(G.neighbors(v))
    if len(neighbors) == 2:
        G.add_edge(neighbors[0], neighbors[1])
    G.remove_node(v)
    return v

def relabel_graph(G):
    '''Relabels nodes in graph to maintain order continuity.

    Args:
        G (nx.Graph): graph to relabel.

    Returns:
        nx.Graph: relabeled graph.
    '''
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes))}
    return nx.relabel_nodes(G, mapping)

def random_tree(n):
    '''Generates random tree with n nodes
    Args:
        n (int): Number of nodes in the tree.

    Returns:
        nx.Graph: random tree.
    '''
    prufer = [random.randint(0, n - 1) for _ in range(n - 2)]
    degree = [1] * n
    for node in prufer:
        degree[node] += 1

    edges = []
    for node in prufer:
        for i in range(n):
            if degree[i] == 1:
                edges.append((i, node))
                degree[i] -= 1
                degree[node] -= 1
                break
    u, v = [i for i in range(n) if degree[i] == 1]
    edges.append((u, v))
    return nx.Graph(edges)

def mutate_graph(G: nx.Graph, depth: int):
    '''Applies a mutation to the graph based on current depth.

    Args:
        G (nx.Graph): input graph.
        depth (int): Current recursion depth.

    Returns:
        nx.Graph: mutated graph.
    '''
    if random.random() < depth / (depth + 1):
        mutator = remove_randleaf if random.random() < 0.5 else remove_subdiv
        removed = mutator(G)
        if removed is not None:
            G = relabel_graph(G, removed)
    return G

def log_progress(G, score, iteration, depth, level):
    '''Logs progress of the AMCS algorithm.

    Args:
        G (nx.Graph): Current graph.
        score (float): Score of the current graph.
        iteration (int): Iteration count.
        depth (int): Current depth.
        level (int): Current level.
    '''
    adj_matrix = nx.to_numpy_array(G, dtype=int)
    half_adj = [int(adj_matrix[i, j]) for i in range(len(adj_matrix)) for j in range(i + 1, len(adj_matrix))]
    logging.info(f"Iteration {iteration} | Score: {score:.3f} | Depth: {depth} | Level: {level} | Adjacency: {half_adj}")

def AMCS(score_function, initial_graph = None, trees_only = False):
    '''Adaptive Monte Carlo Search
    '''
    NMCS = NMCS_trees if trees_only else NMCS_connected_graphs
    start_time = time.time()

    graph = initial_graph or random_tree(5)
    current_score = score_function(graph)
    logging.info(f"Initial score: {current_score:.3f}")

    depth = 0
    level = 1
    min_order = graph.number_of_nodes()
    stagnation_counter = 0
    iteration = 0

    while current_score <= 0 and level <= MAX_LEVEL:
        next_graph = graph.copy()

        while next_graph.number_of_nodes() > min_order:
            next_graph = mutate_graph(next_graph, depth)

        next_graph = NMCS(next_graph, depth, level, score_function)
        next_score = score_function(next_graph)
        iteration += 1

        log_progress(graph, current_score, iteration, depth, level)

        if next_score > current_score:
            graph = next_graph
            current_score = next_score
            depth = 0
            level = 1
            stagnation_counter = 0
        elif depth < MAX_DEPTH:
            depth += 1
            stagnation_counter += 1
        else:
            depth = 0
            level += 1
            stagnation_counter += 1

        if stagnation_counter >= STAGNATION_THRESHOLD:
            logging.info("Resetting graph due to stagnation.")
            graph = random_tree(min_order)
            depth = 0
            level = 1
            stagnation_counter = 0

    if current_score > 0:
        logging.info("Counterexample found.")
    else:
        logging.info("No valid counterexample found.")

    logging.info(f"Final Score: {current_score:.3f} | Iterations: {iteration} | Nodes: {graph.number_of_nodes()}")
    logging.info(f"Search time: {time.time() - start_time:.2f} seconds")
    return graph

def visualize_graph(G, path):
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True)
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", help="Run number", type=int, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    seed = get_seed(args.run)
    random.seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    identifier = f"{timestamp}_run{args.run}_seed{seed}"

    out_dir = create_output_dir()
    log_path = os.path.join(out_dir, f"{identifier}_log.txt")
    setup_logging(log_path)

    final_graph = AMCS(scores.Conj2_score)
    visualize_graph(final_graph, os.path.join(out_dir, f"{identifier}_graph.png"))

if __name__ == "__main__":
    main()