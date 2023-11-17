import ray

import configuration


def consensus_confirmation(x: []):
    """
    Consensus confirmation for input x.
    """
    byzantine_values = 0
    value = None
    for val, byzantine in x:
        if byzantine:
            byzantine_values += 1
        else:
            value = val

    if byzantine_values == 0 \
            or (configuration.optimize_communication and (len(x)-byzantine_values) >= (configuration.max_byzantine_nodes_per_group + 1))\
            or (not configuration.optimize_communication and byzantine_values < int(len(x) / 2) + 1):
        result = value
    else:
        result = None

    return result


def consensus_confirmation_receiver(x: []):
    """
    Consensus confirmation for input x.
    """
    byzantine_values = 0
    value = None
    receive_next = None
    for val, _receive_next, byzantine in x:
        if byzantine:
            byzantine_values += 1
        else:
            value = val
            receive_next = _receive_next

    if byzantine_values == 0 \
            or (configuration.optimize_communication and (len(x)-byzantine_values) >= (configuration.max_byzantine_nodes_per_group + 1))\
            or (not configuration.optimize_communication and byzantine_values < int(len(x) / 2) + 1):
        result = value, receive_next
    else:
        result = None, None

    return result


def accept_byzantine(x: []):
    for val, byzantine in x:
        if byzantine:
            return val


def collect_next(x: [], sim_byzantine, node, receiver):
    _shares, _receive_next = None, None
    if receiver:
        _shares, _receive_next, _ = x[0]
    else:
        _shares, _ = x[0]

    shapes = []
    length = 1
    if type(_shares) == tuple:
        for _share in _shares:
            if type(_share) == list:
                shapes.append(_share[0].shape)
                length = len(_share)
            else:
                shapes.append(_share.shape)
    elif type(_shares) == list:
        shapes.append(_shares[0].shape)
        length = len(_shares)
    else:
        shapes.append(_shares.shape)

    ray.get(node.collect.remote(shapes, length))

    if receiver:
        return _shares, _receive_next, sim_byzantine
    else:
        return _shares, sim_byzantine
