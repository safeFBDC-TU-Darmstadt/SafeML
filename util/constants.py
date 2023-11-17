import torch

import configuration as cfg

tensor_package = 'torch'
torch.set_default_tensor_type(torch.DoubleTensor)

update_collection_strategy = False

share_min = - 2 ** 6
share_max = 2 ** 6 - 1

num_workers = cfg.replication_factor
num_auxiliaries = cfg.replication_factor
num_mediators = cfg.replication_factor
group_sizes = [num_workers for _ in range(cfg.num_groups)]
total_nodes = cfg.num_groups * num_workers + num_auxiliaries + num_mediators + 2

if cfg.threat_model == 'malicious':
    byzantine_setup = {
        'auxiliary': {i: i < cfg.max_byzantine_nodes_per_group if num_auxiliaries >= 2 * cfg.max_byzantine_nodes_per_group + 1 else False for i in range(num_auxiliaries)},
        'mediator': {i: i < cfg.max_byzantine_nodes_per_group if num_mediators >= 2 * cfg.max_byzantine_nodes_per_group + 1 else False for i in range(num_mediators)},
        'worker': {i: {j: j < cfg.max_byzantine_nodes_per_group if group_sizes[i] >= 2 * cfg.max_byzantine_nodes_per_group + 1 else False for j in range(group_sizes[i])} for i in range(cfg.num_groups)}
    }
elif cfg.threat_model == 'semi-honest':
    byzantine_setup = {
        'auxiliary': {i: False for i in range(num_auxiliaries)},
        'mediator': {i: False for i in range(num_mediators)},
        'worker': {i: {j: False for j in range(group_sizes[i])} for i in range(cfg.num_groups)}
    }

lr = None
if cfg.network == 'SecureML':
    if cfg.batch_size <= 1:
        lr = 2 ** (-7)
    elif cfg.batch_size <= 10:
        lr = (2 ** (-7)) * cfg.batch_size
    else:
        lr = (2 ** (-7)) * (cfg.batch_size/2)
elif cfg.batch_size == 'Chameleon':
    if cfg.batch_size <= 1:
        lr = 0.01
    elif cfg.batch_size <= 10:
        lr = 0.05
    else:
        lr = 0.1
else:
    raise Exception(f'Unsupported network \'{cfg.network}\'.')
