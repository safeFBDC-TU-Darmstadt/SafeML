import ray
import torch
from tqdm import tqdm

import configuration as cfg
import util
from actors import AuxiliaryNode, DataOwner, MediatorNode, ModelOwner, WorkerNode
from actors.auxiliary_collection_node import Recollector
from actors.util import round_robin_collection_single_group
from util.networks import construct_network


def get_accumulated_comm_cost(workers, auxiliaries, mediators):
    comm_costs = []
    comm_costs.extend(ray.get([w.get_comm_cost.remote() for w in workers]))
    comm_costs.extend(ray.get([a.get_comm_cost.remote() for a in auxiliaries]))
    comm_costs.extend(ray.get([m.get_comm_cost.remote() for m in mediators]))

    msgs, msg_size = 0, 0
    for _msgs, _msg_size in comm_costs:
        msgs += _msgs
        msg_size += _msg_size

    return msgs, msg_size


def main():

    """ local ray start """
    resources = dict()
    for i in range(cfg.num_groups):
        resources[f"group{i}"] = cfg.replication_factor
    resources["auxiliary"] = util.constants.num_auxiliaries
    resources["mediator"] = util.constants.num_mediators
    if not ray.is_initialized(): ray.init(log_to_driver=cfg.log_warnings, resources=resources)

    print(f'network={cfg.network}, batch_size={cfg.batch_size}, lr={util.constants.lr}, iterations={cfg.iterations}, train={cfg.train}')

    random_fct = lambda size: torch.rand(size, dtype=torch.float64, device='cpu')

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

    distributed_nn = construct_network(name=cfg.network, type='distributed', batch_size=cfg.batch_size, lr=util.constants.lr, auxiliary_nodes=auxiliaries, mediator_nodes=mediators, ran_group=ran_group, output_mask=output_mask)

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

    X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]
    for i in tqdm(range(cfg.iterations * cfg.batch_size)):
        if (i+1) % cfg.batch_size == 0 and i > 0:
            ray.get([w.iterate.remote(X_shares[j], Y_shares[j], train=cfg.train, return_result=False, return_runtime=False) for w, j in zip(workers, group_nums)])
            X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]

        _X_shares, _Y_shares = data_owner.mnist_train_shares(i)
        for j, (_X_share, _Y_share) in enumerate(zip(_X_shares, _Y_shares)):
            X_shares[j].append(_X_share)
            Y_shares[j].append(_Y_share)

    ray.get([w.iterate.remote(X_shares[j], Y_shares[j], train=cfg.train, return_result=False, return_runtime=False) for w, j in zip(workers, group_nums)])

    msgs, msg_size = get_accumulated_comm_cost(workers, auxiliaries, mediators)
    tqdm.write(f'Total # of messages: {msgs}')
    tqdm.write(f'Total communication cost: {msg_size} B = {msg_size / (10**6)} MB = {msg_size / (10**9)} GB')


if __name__ == '__main__':
    main()
