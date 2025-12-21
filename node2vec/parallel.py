import random
from tqdm import tqdm
import numpy as np


def parallel_precompute_probabilities(source, graph, p, q, weight_key, 
                                      sampling_strategy, PROBABILITIES_KEY):
    """
    Precomputes transition probabilities for a single source node.
    This function is designed to be called in parallel for multiple nodes.
    
    :param source: The source node to compute probabilities for
    :param graph: The NetworkX graph
    :param p: Return hyper parameter
    :param q: Input parameter
    :param weight_key: Key for edge weights
    :param sampling_strategy: Node-specific sampling strategies
    :param PROBABILITIES_KEY: Key for probabilities in d_graph (for reference)
    :return: Dictionary with computed probabilities for this source node
    """
    result = {}
    
    for current_node in graph.neighbors(source):
        unnormalized_weights = list()
        
        # Calculate unnormalized weights
        for destination in graph.neighbors(current_node):
            p_val = sampling_strategy[current_node].get('p', p) if current_node in sampling_strategy else p
            q_val = sampling_strategy[current_node].get('q', q) if current_node in sampling_strategy else q
            
            try:
                if graph[current_node][destination].get(weight_key):
                    weight = graph[current_node][destination].get(weight_key, 1)
                else:
                    # Example: AtlasView({0: {'type': 1, 'weight':0.1}}) - when we have edge weight
                    edge = list(graph[current_node][destination])[-1]
                    weight = graph[current_node][destination][edge].get(weight_key, 1)
            except:
                weight = 1
            
            if destination == source:  # Backwards probability
                ss_weight = weight * 1 / p_val
            elif destination in graph[source]:  # If the neighbor is connected to the source
                ss_weight = weight
            else:
                ss_weight = weight * 1 / q_val
            
            # Assign the unnormalized sampling strategy weight, normalize during random walk
            unnormalized_weights.append(ss_weight)
        
        # Normalize
        unnormalized_weights = np.array(unnormalized_weights)
        result[current_node] = unnormalized_weights / unnormalized_weights.sum()
    
    # Calculate first_travel weights for source
    first_travel_weights = []
    for destination in graph.neighbors(source):
        try:
            weight = graph[source][destination].get(weight_key, 1)
        except:
            weight = 1
        first_travel_weights.append(weight)
    
    first_travel_weights = np.array(first_travel_weights)
    result['first_travel'] = first_travel_weights / first_travel_weights.sum()
    
    # Save neighbors
    result['neighbors'] = list(graph.neighbors(source))
    
    return result


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
