import networkx as nx
import random

def add_randleaf(G):
    '''Adds a random leaf to G'''
    n = max(G.nodes, default=-1) + 1
    random_vertex = random.choice(list(G.nodes))
    G.add_node(n)
    G.add_edge(random_vertex, n)

def add_leaf(G, v):
    '''Adds a leaf adjacent to a vertex v of G'''
    n = max(G.nodes, default=-1) + 1
    G.add_node(n)
    G.add_edge(v, n)

def add_randsubdiv(G):
    '''Subdivide a random edge of G by inserting a vertex'''
    if G.number_of_edges() == 0:
        return
    u, v = random.choice(list(G.edges))
    new_node = max(G.nodes, default=-1) + 1
    G.remove_edge(u, v)
    G.add_node(new_node)
    G.add_edge(u, new_node)
    G.add_edge(new_node, v)

def NMCS_trees(current_graph, depth, level, score_function, is_parent=True):
    '''NMCS for trees'''
    best_graph = current_graph
    best_score = score_function(current_graph)

    if level == 0:
        next_graph = current_graph.copy()
        for _ in range(depth):
            if random.random() < 0.5:
                add_randleaf(next_graph)
            else:
                add_randsubdiv(next_graph)
        if score_function(next_graph) > best_score:
            best_graph = next_graph.copy()
    else:
        for x in list(current_graph.nodes) + list(current_graph.edges):
            next_graph = current_graph.copy()
            if isinstance(x, tuple):  # edge
                u, v = x
                new_node = max(next_graph.nodes, default=-1) + 1
                next_graph.remove_edge(u, v)
                next_graph.add_node(new_node)
                next_graph.add_edge(u, new_node)
                next_graph.add_edge(new_node, v)
            else:  # node
                add_leaf(next_graph, x)

            next_graph = NMCS_trees(next_graph, depth, level - 1, score_function, False)
            score = score_function(next_graph)
            if score > best_score:
                best_graph = next_graph.copy()
                best_score = score
                if current_graph.number_of_nodes() > 20 and is_parent:
                    break
    return best_graph

def NMCS_connected_graphs(current_graph, depth, level, score_function, is_parent=True):
    '''NMCS for connected graphs'''
    best_graph = current_graph
    best_score = score_function(current_graph)

    if level == 0:
        next_graph = current_graph.copy()
        for _ in range(depth):
            rand = random.random()
            possible_edges = list(nx.non_edges(next_graph))
            if rand < 0.5 and possible_edges:
                next_graph.add_edge(*random.choice(possible_edges))
            elif rand < 0.8:
                add_randleaf(next_graph)
            else:
                add_randsubdiv(next_graph)

        if score_function(next_graph) > best_score:
            best_graph = next_graph.copy()
    else:
        all_elements = list(current_graph.nodes) + list(current_graph.edges) + list(nx.non_edges(current_graph))
        for x in all_elements:
            next_graph = current_graph.copy()
            if isinstance(x, tuple):
                if x in current_graph.edges:
                    # subdivide edge
                    u, v = x
                    new_node = max(next_graph.nodes, default=-1) + 1
                    next_graph.remove_edge(u, v)
                    next_graph.add_node(new_node)
                    next_graph.add_edge(u, new_node)
                    next_graph.add_edge(new_node, v)
                else:
                    # add new edge from complement
                    u, v = x
                    next_graph.add_edge(u, v)
            else:
                add_leaf(next_graph, x)

            next_graph = NMCS_connected_graphs(next_graph, depth, level - 1, score_function, False)
            score = score_function(next_graph)
            if score > best_score:
                best_graph = next_graph.copy()
                best_score = score
                if current_graph.number_of_nodes() > 20 and is_parent:
                    break
    return best_graph
