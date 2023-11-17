import time

import ray
from ray.util.queue import Queue

import configuration as cfg
import util
from actors.abstract import AbstractNode, Receiver, ReceiverManager
from actors.util import round_robin
from neural_network import DistributedNeuralNetwork
from secret_sharing import SecComp

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class WorkerNode(AbstractNode, ReceiverManager):

    nn: DistributedNeuralNetwork

    _auxiliary_nodes: []
    _sec_comp: SecComp
    _group_num: int
    _result_queues: [Queue]

    def __init__(self, group_num: int, auxiliary_nodes: [], mediator_nodes: [], model_owner, nn: DistributedNeuralNetwork,
                 node_id: int, receive_next: bool, recollector) -> None:
        # set up receivers
        auxiliary_receiver = Receiver(max_results=cfg.max_byzantine_nodes_per_group + 1, receive_next=receive_next, node_id=node_id)
        mediator_receiver = Receiver(max_results=cfg.max_byzantine_nodes_per_group + 1, receive_next=receive_next, node_id=node_id)
        receiver_dict = {'auxiliary': auxiliary_receiver, 'mediator': mediator_receiver}
        ReceiverManager.__init__(self, receiver_dict)

        AbstractNode.__init__(self, node_id, util.constants.byzantine_setup['worker'][group_num][node_id], byzantine_dict=None)
        self.auxiliary_byzantine_dict = util.constants.byzantine_setup['auxiliary']
        self.mediator_byzantine_dict = util.constants.byzantine_setup['mediator']

        self.iteration_auxiliaries = 0
        self.receive_from_auxiliaries = round_robin(self.iteration_auxiliaries, util.constants.num_auxiliaries, 'single')
        self.iteration_mediators = 0
        self.receive_from_mediators = round_robin(self.iteration_auxiliaries, util.constants.num_mediators, 'single')

        self._auxiliary_nodes = auxiliary_nodes
        self._mediator_nodes = mediator_nodes
        self._sec_comp = SecComp(auxiliary_nodes, mediator_nodes, model_owner, group_num, node_id, self, recollector)
        self._group_num = group_num
        if nn is not None:
            self.nn = nn
            self.nn.sec_comp = self._sec_comp
            for layer in self.nn.layers:
                layer.sec_comp = self._sec_comp
        nn.set_worker_node(self)

    def update_model(self, nn: DistributedNeuralNetwork):
        self.nn = nn

    async def iterate(self, X_i, y_i, train=True, return_result=True, return_runtime=False):
        start = time.time_ns()

        output = await self.nn.forward_pass(X_i)

        if train:
            await self.nn.backward_pass(y_i)

        end = time.time_ns()

        if return_result:
            if return_runtime:
                return output, start, end
            else:
                return output
        elif return_runtime:
            return start, end
        else:
            return None

    async def inference(self, X_i):
        return await self.nn.forward_pass(X_i), self.byzantine

    def get_comm_cost(self):
        return self._sec_comp.online_comm_evaluator.msgs, self._sec_comp.online_comm_evaluator.msg_size

    def init_auxiliary_numbers(self, a_i, b_i, c_i):
        self._sec_comp.a_i = a_i
        self._sec_comp.b_i = b_i
        self._sec_comp.c_i = c_i

    def init_auxiliary_matrices(self, A_i: dict, B_i: dict, C_i: dict):
        self._sec_comp.A_i = A_i
        self._sec_comp.B_i = B_i
        self._sec_comp.C_i = C_i

    def increase_auxiliary_iteration(self):
        self.iteration_auxiliaries += 1
        if util.constants.update_collection_strategy:
            self.receive_from_auxiliaries = round_robin(self.iteration_auxiliaries, util.constants.num_auxiliaries, 'single')

    def increase_mediator_iteration(self):
        self.iteration_mediators += 1
        if util.constants.update_collection_strategy:
            self.receive_from_mediators = round_robin(self.iteration_mediators, util.constants.num_mediators, 'single')
