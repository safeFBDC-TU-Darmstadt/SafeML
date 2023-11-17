import sys

import ray
import torch
from tqdm import tqdm

import configuration as cfg
import util
from actors import AuxiliaryNode, DataOwner, MediatorNode, ModelOwner, WorkerNode
from actors.auxiliary_collection_node import Recollector
from actors.util import round_robin_collection_single_group
from util.networks import construct_network


def main():

    """ local ray start """
    resources = dict()
    for i in range(cfg.num_groups):
        resources[f"group{i}"] = cfg.replication_factor
    resources["auxiliary"] = util.constants.num_auxiliaries
    resources["mediator"] = util.constants.num_mediators
    if not ray.is_initialized(): ray.init(log_to_driver=cfg.log_warnings, resources=resources)

    network = 'SecureML' if len(sys.argv) < 2 else sys.argv[1]
    batch_size = 1
    iterations = 10

    lr = None
    if network == 'SecureML':
        if batch_size <= 1:
            lr = 2 ** (-7)
        elif batch_size <= 10:
            lr = (2 ** (-7)) * batch_size
        else:
            lr = (2 ** (-7)) * (batch_size / 2)
    elif network == 'Chameleon':
        if batch_size <= 1:
            lr = 0.01
        elif batch_size <= 10:
            lr = 0.05
        else:
            lr = 0.1

    print(f'network={network}, batch_size={batch_size}, lr={lr}, iterations={iterations}')

    device = 'cpu'

    random_fct = lambda size: torch.rand(size, dtype=torch.float64, device=device)

    ran_group = 0

    recollector = Recollector.remote()

    output_mask = torch.rand(10)
    model_owner = ModelOwner.remote(cfg.num_groups, torch.exp, torch.empty_like, torch.abs, util.constants.num_mediators,
                                    output_mask, random_fct, recollector)

    data_owner = DataOwner(cfg.num_groups, random_fct, torch.zeros)

    auxiliaries = []
    for i in range(util.constants.num_auxiliaries):
        auxiliaries.append(AuxiliaryNode.options(resources={"auxiliary": 1}).remote(util.constants.group_sizes, node_id=i, recollector=recollector))

    mediators = []
    senders_m = round_robin_collection_single_group(iteration=0, g_size=util.constants.num_mediators)
    for i in range(util.constants.num_mediators):
        mediators.append(MediatorNode.options(resources={"mediator": 1}).remote(util.constants.group_sizes, model_owner, node_id=i, receive_next=i not in senders_m, recollector=recollector))

    ray.get([model_owner.set_mediators.remote(mediators)])

    distributed_nn = construct_network(name=network, type='distributed', batch_size=batch_size, lr=lr,
                                       auxiliary_nodes=auxiliaries, mediator_nodes=mediators, ran_group=ran_group,
                                       output_mask=output_mask)

    workers = []
    workers_dict = {}
    group_nums = []
    models = ray.get(model_owner.create_models.remote(distributed_nn, cfg.num_groups))
    init_senders_w = round_robin_collection_single_group(iteration=0, g_size=util.constants.num_workers)
    for i in range(cfg.num_groups):
        workers_dict[i] = []
        for j in range(util.constants.num_workers):
            worker = WorkerNode.options(resources={f"group{i}": 1}).remote(group_num=i, auxiliary_nodes=auxiliaries,
                                                                           mediator_nodes=mediators,
                                                                           model_owner=model_owner, nn=models[i],
                                                                           node_id=j,
                                                                           receive_next=j not in init_senders_w,
                                                                           recollector=recollector)
            workers.append(worker)
            group_nums.append(i)
            workers_dict[i].append(worker)

    ray.get([a.set_workers.remote(workers_dict) for a in auxiliaries])
    ray.get([m.set_workers.remote(workers_dict) for m in mediators])

    train = True if len(sys.argv) < 3 else (sys.argv[2] == 'train')
    times = []
    X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]
    for i in tqdm(range(iterations * batch_size)):
        if (i + 1) % batch_size == 0 and i > 0:
            get_runtime(X_shares, Y_shares, group_nums, times, train, workers)
            X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]

        _X_shares, _Y_shares = data_owner.mnist_train_shares(i)
        for j, (_X_share, _Y_share) in enumerate(zip(_X_shares, _Y_shares)):
            X_shares[j].append(_X_share)
            Y_shares[j].append(_Y_share)

    get_runtime(X_shares, Y_shares, group_nums, times, train, workers)
    total_runtime = sum(times)
    avg_time_per_iteration = total_runtime / len(times)
    total_runtime_ex_0 = sum(times[1:])
    avg_time_per_iteration_ex_0 = total_runtime_ex_0 / (len(times) - 1)
    tqdm.write(f'Total runtime: {total_runtime} ns = {total_runtime / (10 ** 6)} ms = {total_runtime / (10 ** 9)} s')
    tqdm.write(f'Runtime per iteration (ns): {times}')
    tqdm.write(f'Average runtime per iteration: {avg_time_per_iteration} ns = {avg_time_per_iteration / (10 ** 6)} ms = {avg_time_per_iteration / (10 ** 9)} s')
    tqdm.write(f'Average runtime per iteration (excluding first iteration): {avg_time_per_iteration_ex_0} ns = {avg_time_per_iteration_ex_0 / (10 ** 6)} ms = {avg_time_per_iteration_ex_0 / (10 ** 9)} s')


def get_runtime(X_shares, Y_shares, group_nums, times, train, workers):
    res_dict = ray.get(
        [w.iterate.remote(X_shares[j], Y_shares[j], train=train, return_result=False, return_runtime=True) for w, j in
         zip(workers, group_nums)])
    start_times = [res[0] for res in res_dict]
    end_times = [res[1] for res in res_dict]
    times.append(max(end_times) - min(start_times))


if __name__ == '__main__':
    main()
