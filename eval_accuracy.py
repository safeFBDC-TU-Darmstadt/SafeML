import os

import ray
import torch
from tqdm import tqdm

import configuration as cfg
import util.constants
from actors import AuxiliaryNode, DataOwner, MediatorNode, ModelOwner, WorkerNode
from actors.auxiliary_collection_node import Recollector
from actors.util import round_robin_collection_single_group
from data.preparation.mnist import mnist
from neural_network import CentralizedNeuralNetwork
from secret_sharing.consensus import consensus_confirmation
from util.networks import construct_distributed_from_centralized


def main():

    """ local ray start """
    resources = dict()
    for i in range(cfg.num_groups):
        resources[f"group{i}"] = cfg.replication_factor
    resources["auxiliary"] = util.constants.num_auxiliaries
    resources["mediator"] = util.constants.num_mediators
    if not ray.is_initialized(): ray.init(log_to_driver=cfg.log_warnings, resources=resources)

    tqdm.write(f'network={cfg.network}, batch_size={cfg.batch_size}, lr={util.constants.lr}, epochs={cfg.epochs}')

    device = 'cpu'

    random_fct = lambda size: torch.rand(size, dtype=torch.float64, device=device)

    ran_group = 0

    recollector = Recollector.remote()

    output_mask = torch.rand(10)
    model_owner = ModelOwner.remote(cfg.num_groups, torch.exp, torch.empty_like, torch.abs, util.constants.num_mediators,
                                    output_mask, random_fct, recollector)

    data_owner = DataOwner(cfg.num_groups, random_fct, torch.zeros)
    loaders = mnist.prepare_loaders(batch_size=cfg.batch_size)

    auxiliaries = []
    for i in range(util.constants.num_auxiliaries):
            auxiliaries.append(AuxiliaryNode.options(resources={"auxiliary": 1}).remote(util.constants.group_sizes, node_id=i, recollector=recollector))

    mediators = []
    senders_m = round_robin_collection_single_group(iteration=0, g_size=util.constants.num_mediators)
    for i in range(util.constants.num_mediators):
            mediators.append(MediatorNode.options(resources={"mediator": 1}).remote(util.constants.group_sizes, model_owner, node_id=i, receive_next=i not in senders_m, recollector=recollector))

    ray.get([model_owner.set_mediators.remote(mediators)])

    centralized_nn = CentralizedNeuralNetwork(batch_size=cfg.batch_size, lr=util.constants.lr)
    centralized_nn.load_from_files(torch.load, f'models/{cfg.network.lower()}/parameters.db', f'models/{cfg.network.lower()}/layers.txt')
    distributed_nn = construct_distributed_from_centralized(other_nn=centralized_nn, batch_size=cfg.batch_size, lr=util.constants.lr,
                                                            auxiliary_nodes=auxiliaries, mediator_nodes=mediators,
                                                            ran_group=ran_group, output_mask=output_mask)

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

    if not os.path.exists('results'):
        os.makedirs('results')

    f = open(f'results/{cfg.network}_epochs_{cfg.epochs}_bs_{cfg.batch_size}.txt', 'w')
    f.write('Epochs Centralized SafeML\n')

    train_size = data_owner.get_mnist_train_size()
    test_size = data_owner.get_mnist_test_size()
    test_interval = 10000
    bar1 = tqdm(total=train_size, desc=f'Training (Epoch 1/{cfg.epochs})', position=0, leave=True)
    bar2 = tqdm(total=test_size, desc='Testing', position=1, leave=False)
    for e in range(cfg.epochs):
        bar1.reset(total=train_size)
        bar1.set_description(f'Training (Epoch {e+1}/{cfg.epochs})')

        X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]
        X, Y = [], []

        for i in range(train_size):

            if i % test_interval == 0:
                bar2.reset(test_size)
                bar2.set_description('Centralized Tests')
                accuracy_centralized = run_tests_centralized(centralized_nn, loaders, cfg.batch_size, test_size=test_size, tqdm_bar=bar2)
                bar2.reset(test_size)
                bar2.set_description('SafeML Tests')
                accuracy_distributed = run_tests_distributed(data_owner, group_nums, workers, test_size=test_size, tqdm_bar=bar2)
                bar2.reset(test_size)
                bar2.set_description('Testing')
                bar2.update()
                tqdm.write(f'Latest Accuracy (Epoch, Centralized, SafeML): {(i + e * train_size) / train_size} {accuracy_centralized * 100} {accuracy_distributed * 100}')
                f.write(f'{(i + e * train_size) / train_size} {accuracy_centralized * 100} {accuracy_distributed * 100}\n')

            if i % cfg.batch_size == 0 and i > 0 or i == train_size - 1:
                ray.get([w.iterate.remote(X_shares[j], Y_shares[j]) for w, j in zip(workers, group_nums)])

                centralized_nn.forward_pass(X)
                centralized_nn.backward_pass(Y)

                bar1.update(cfg.batch_size)

                X_shares, Y_shares = [[] for _ in range(cfg.num_groups)], [[] for _ in range(cfg.num_groups)]
                X, Y = [], []

            _X_shares, _Y_shares = data_owner.mnist_train_shares(i)
            for j, (_X_share, _Y_share) in enumerate(zip(_X_shares, _Y_shares)):
                X_shares[j].append(_X_share)
                Y_shares[j].append(_Y_share)

            image, label = loaders['train'].dataset[i]
            label_tensor = torch.zeros(10)
            label_tensor[label] = 1
            X.append(image)
            Y.append(label_tensor)

    bar2.reset(test_size)
    bar2.set_description('Centralized Tests')
    accuracy_centralized = run_tests_centralized(centralized_nn, loaders, cfg.batch_size, test_size=test_size, tqdm_bar=bar2)
    bar2.reset(test_size)
    bar2.set_description('SafeML Tests')
    accuracy_distributed = run_tests_distributed(data_owner, group_nums, workers, test_size=test_size, tqdm_bar=bar2)
    bar2.reset(test_size)
    tqdm.write(f'Latest Accuracy (Epoch, Centralized, SafeML): {float(cfg.epochs)} {accuracy_centralized * 100} {accuracy_distributed * 100}')
    f.write(f'{float(cfg.epochs)} {accuracy_centralized * 100} {accuracy_distributed * 100}\n')


def run_tests_distributed(data_owner, group_nums, workers, test_size, tqdm_bar):
    correct = 0
    for i in range(test_size):
        X_shares, label = data_owner.mnist_test_shares(i, test_size)
        results = ray.get([w.inference.remote(X_shares[j]) for w, j in zip(workers, group_nums)])

        result_dict = {}
        for j in range(cfg.num_groups):
            for k in range(util.constants.num_workers):
                if j not in result_dict.keys():
                    result_dict[j] = []
                result_dict[j].append(results[j * util.constants.num_workers + k])

        cons_results = []
        for k, v in result_dict.items():
            cons_results.append(consensus_confirmation(v))

        prediction = torch.argmax(sum(cons_results))

        if prediction == label:
            correct += 1

        tqdm_bar.update(1)

    return correct / test_size


def run_tests_centralized(nn, loaders, batch_size, test_size, tqdm_bar):
    correct = 0

    for i, (images, labels) in enumerate(loaders['test']):
        if i * batch_size >= test_size:
            return correct / test_size
        for j in range(batch_size):
            output = nn.forward_pass(images[j])
            result = torch.argmax(output)
            if result == labels[j]:
                correct = correct + 1

        tqdm_bar.update(1)

    return correct / test_size


if __name__ == '__main__':
    main()
