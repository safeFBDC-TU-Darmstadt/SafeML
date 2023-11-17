import configuration as cfg

share_min = - 2 ** 6
share_max = 2 ** 6 - 1

num_workers = cfg.replication_factor
num_auxiliaries = cfg.replication_factor
num_mediators = cfg.replication_factor
group_sizes = [num_workers for _ in range(cfg.num_groups)]
total_nodes = cfg.num_groups * num_workers + num_auxiliaries + num_mediators + 2
