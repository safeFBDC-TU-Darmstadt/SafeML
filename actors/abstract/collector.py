from asyncio import Queue, Lock

import configuration
import util
from actors.abstract import AbstractNode
from actors.util import round_robin, next_node
from secret_sharing.consensus import consensus_confirmation, accept_byzantine, collect_next


class Collector(AbstractNode):
    """
    Abstract class to collect secret shares from groups of nodes.
    """

    def __init__(self, node_id, byzantine, group_sizes, nodes, comm_evaluator, byzantine_dict, recollector, collection_strategy='rr', collection_mode='mult') -> None:
        """
        Parameters
        ----------
        group_sizes:
            A list of group sizes if ``collection_mode=='mult'`` or the size of the single group to collect from if
            ``collection_mode=='single'``.
        """

        AbstractNode.__init__(self, node_id, byzantine, byzantine_dict)

        self.collector_comm_evaluator = comm_evaluator

        collection_strategies = ['rr']
        collection_modes = ['mult', 'single']

        self.sec_result = Queue(maxsize=1)
        self.sec_shares = {}
        self.sec_shares_cons = {}
        self.group_sizes = group_sizes
        if collection_strategy not in collection_strategies:
            raise ValueError(f"Collection strategy '{collection_strategy}' unknown.")
        self.collection_strategy = collection_strategy
        if collection_mode not in collection_modes:
            raise ValueError(f"Collection mode '{collection_mode}' unknown.")
        self.collection_mode = collection_mode
        self.collection_iteration = 0
        self.collection_ids = self.apply_collection_strategy(0)
        self.new_collection_ids = None
        self.nodes = nodes
        self.iteration_locks = {}
        self.iteration_locks_access = Lock()
        self.results_delivered = 0

        self.recollector = recollector

    def set_workers(self, worker_nodes):
        self.nodes = worker_nodes

    def apply_collection_strategy(self, iteration=None):
        if self.collection_strategy == 'rr':
            return round_robin(self.collection_iteration if iteration is None else iteration, self.group_sizes, self.collection_mode)
        else:
            raise ValueError(f"Collection strategy '{self.collection_strategy}' not supported yet.")

    async def _reset_or_wait(self, iteration):
        if iteration > self.collection_iteration:
            async with self.iteration_locks_access:
                if iteration not in self.iteration_locks:
                    self.iteration_locks[iteration] = Lock()
                    await self.iteration_locks[iteration].acquire()
            await self.iteration_locks[iteration].acquire()
            self.iteration_locks[iteration].release()

    async def _await_result(self):
        # wait until result queue is not empty (-> method can be called in parallel!)
        result = await self.sec_result.get()
        # re-queue result
        await self.sec_result.put(result)
        self.results_delivered = self.results_delivered + 1
        return result

    async def _optional_reset(self):
        if configuration.optimize_communication:
            results_to_deliver = sum([len(node_ids) for node_ids in self.collection_ids.values()])
        else:
            results_to_deliver = sum(self.group_sizes)

        if self.results_delivered == results_to_deliver:
            await self._reset_helper_vars()

    async def _reset_helper_vars(self):
        await self.sec_result.get()
        self.sec_shares = {}
        self.sec_shares_cons = {}
        self.results_delivered = 0
        self.collection_iteration += 1
        async with self.iteration_locks_access:
            if self.collection_iteration in self.iteration_locks and self.iteration_locks[self.collection_iteration].locked():
                self.iteration_locks[self.collection_iteration].release()

    async def _save_shares(self, shares, group_num, byzantine):
        if group_num not in self.sec_shares.keys():
            self.sec_shares[group_num] = []
        self.sec_shares[group_num].append((shares, byzantine))

    async def _check_ready(self, msg_size, sender_group, iteration):
        """
        Check if shares of every worker in every group have been received.
        """
        ready = True
        for i, g_size in enumerate(self.group_sizes):

            # check if group i is not consensus-confirmed yet but all shares of group i received
            if (i not in self.sec_shares_cons.keys()) \
                    and (i in self.sec_shares.keys()) \
                    and ((not configuration.optimize_communication and g_size == len(self.sec_shares[i]))
                         or (configuration.optimize_communication and len(self.collection_ids[i]) == len(self.sec_shares[i]))):
                # (try to) perform consensus confirmation for group i
                collection_ids = self.apply_collection_strategy(iteration if util.constants.update_collection_strategy else 0)
                await self._consensus_confirm(g_size, i, self.nodes[i], msg_size, sender_group, iteration, collection_ids[i])

            # check if group i hasn't been consensus-confirmed yet
            elif i not in self.sec_shares_cons.keys():
                ready = False

        return ready

    async def _consensus_confirm(self, g_size, i, nodes, msg_size, sender_group: str, iteration, collection_ids):
        cc_shares = consensus_confirmation(self.sec_shares[i])
        if cc_shares is None:
            _next_node_idx = next_node(collection_ids, g_size)
            if _next_node_idx is None:
                # no nodes left to collect from -> accept some Byzantine value
                self.sec_shares_cons[i] = accept_byzantine(self.sec_shares[i])
            else:
                self.collector_comm_evaluator.msgs += 1
                self.collector_comm_evaluator.msg_size += msg_size

                collected = collect_next(self.sec_shares[i], sim_byzantine=self.byzantine_dict[i][_next_node_idx], node=self.recollector, receiver=False)
                collection_ids.append(_next_node_idx)

                self.sec_shares[i].append(collected)
                await self._consensus_confirm(g_size, i, nodes, msg_size, sender_group, iteration, collection_ids)
        else:
            self.sec_shares_cons[i] = cc_shares
            self.sec_shares.pop(i)

    async def send_to_uncollected_nodes(self, to_send, shares: bool, sender_group: str, iteration, send_byzantine=False):
        # send result to all nodes not included in the first sub-sampling try

        current_collection_ids = self.apply_collection_strategy(iteration if util.constants.update_collection_strategy else 0)
        new_collection_ids = self.apply_collection_strategy(iteration + 1 if util.constants.update_collection_strategy else 0)

        if self.collection_mode == 'mult':
            for g_num, nodes in self.nodes.items():
                for node_id, node in enumerate(nodes):
                    if node_id not in current_collection_ids[g_num]:
                        receive_next = node_id not in new_collection_ids[g_num]
                        node.receive.remote(to_send[g_num] if shares else to_send, send_byzantine, receive_next, iteration, sender_group)

        elif self.collection_mode == 'single':
            for node_id, node in enumerate(self.nodes):
                if node_id not in current_collection_ids:
                    receive_next = node_id not in new_collection_ids
                    node.receive.remote(to_send, send_byzantine, receive_next, iteration, sender_group)
