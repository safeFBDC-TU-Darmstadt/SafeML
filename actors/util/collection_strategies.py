import configuration


def round_robin_collection_multiple_groups(iteration, group_sizes):
    """
    Round-robin collection of shares of a list of groups.

    Parameters
    ----------
    iteration: int
        The iteration of the round-robin collection.
    group_sizes:
        A list of the sizes of the groups to sub-sample.

    Returns
    -------
    collection_ids:
        A dictionary of the like mapping each group number to a list of node ids to collect from.
    """
    collection_ids = {}
    for i, g_size in enumerate(group_sizes):
        if configuration.optimize_communication:
            opt_range = range(configuration.max_byzantine_nodes_per_group + 1)
            collection_ids[i] = [(iteration + j) % g_size for j in opt_range]
        else:
            collection_ids[i] = range(g_size)
    return collection_ids


def round_robin_collection_single_group(iteration, g_size):
    """
    Round-robin collection of shares of a one group.

    Parameters
    ----------
    iteration: int
        The iteration of the round-robin collection.
    g_size:
        The size of the group to sub-sample from.

    Returns
    -------
    collection_ids:
        A list of node ids to collect from.
    """
    if configuration.optimize_communication:
        opt_range = range(configuration.max_byzantine_nodes_per_group + 1)
        opt_range = [(iteration + j) % g_size for j in opt_range]
    else:
        opt_range = range(g_size)
    return opt_range


def round_robin(iteration, group_size, collection_mode):
    if collection_mode == 'single':
        return round_robin_collection_single_group(iteration, group_size)
    elif collection_mode == 'mult':
        return round_robin_collection_multiple_groups(iteration, group_size)


def next_node(node_ids: list, group_size: int):
    for i in range(group_size):
        if i not in node_ids:
            return i
    return None
