import random
from tqdm import tqdm
import numpy as np


def validate_positive_number(value, name):
    """Return value as float if it is finite and positive."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc

    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {number!r}")

    return number


def get_sampling_value(sampling_strategy, node, key, default):
    """Return a validated node-specific sampling value or the default."""
    if node in sampling_strategy and key in sampling_strategy[node]:
        return validate_positive_number(sampling_strategy[node][key], f"sampling_strategy[{node!r}][{key!r}]")

    return default


def get_edge_weight(graph, source, destination, weight_key):
    """Return a finite positive edge weight, defaulting missing weights to 1.

    For MultiGraphs, all parallel edge weights are validated and summed so the
    effective transition weight is deterministic and uses every parallel edge.
    """
    edge_data = graph[source][destination]

    if graph.is_multigraph():
        if not edge_data:
            return 1.0

        return sum(
            validate_positive_number(
                attrs.get(weight_key, 1),
                f"Edge weight for ({source!r}, {destination!r}, {edge_key!r})",
            )
            for edge_key, attrs in edge_data.items()
        )

    return validate_positive_number(
        edge_data.get(weight_key, 1),
        f"Edge weight for ({source!r}, {destination!r})",
    )


def normalize_probabilities(weights, context):
    """Normalize transition weights and reject invalid probability totals."""
    weights = np.array(weights, dtype=np.float64)
    if weights.size == 0:
        return weights

    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Transition weights for {context} must sum to a finite positive value")

    return weights / total


def parallel_precompute_probabilities(source, graph, p, q, weight_key, sampling_strategy, PROBABILITIES_KEY):
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
            p_val = get_sampling_value(sampling_strategy, current_node, "p", p)
            q_val = get_sampling_value(sampling_strategy, current_node, "q", q)

            weight = get_edge_weight(graph, current_node, destination, weight_key)

            if destination == source:  # Backwards probability
                ss_weight = weight * 1 / p_val
            elif destination in graph[source]:  # If the neighbor is connected to the source
                ss_weight = weight
            else:
                ss_weight = weight * 1 / q_val

            # Assign the unnormalized sampling strategy weight, normalize during random walk
            unnormalized_weights.append(ss_weight)

        # Normalize
        result[current_node] = normalize_probabilities(
            unnormalized_weights,
            context=f"transition from {source!r} through {current_node!r}",
        )

    # Calculate first_travel weights for source
    first_travel_weights = []
    for destination in graph.neighbors(source):
        weight = get_edge_weight(graph, source, destination, weight_key)
        first_travel_weights.append(weight)

    result["first_travel"] = normalize_probabilities(
        first_travel_weights,
        context=f"first travel from {source!r}",
    )

    # Save neighbors
    result["neighbors"] = list(graph.neighbors(source))

    return result


def parallel_generate_walks(
    d_graph: dict,
    global_walk_length: int,
    num_walks: int,
    cpu_num: int,
    sampling_strategy: dict = None,
    num_walks_key: str = None,
    walk_length_key: str = None,
    neighbors_key: str = None,
    probabilities_key: str = None,
    first_travel_key: str = None,
    quiet: bool = False,
) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc="Generating walks (CPU: {})".format(cpu_num))

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
            if (
                source in sampling_strategy
                and num_walks_key in sampling_strategy[source]
                and sampling_strategy[source][num_walks_key] <= n_walk
            ):
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
