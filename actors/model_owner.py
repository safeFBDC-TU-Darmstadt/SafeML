from asyncio import Queue

import ray

import configuration as cfg
import secret_sharing.additive_secret_sharing
import util
from actors.abstract import Collector
from actors.abstract.collector import next_node
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size
from neural_network import DistributedNeuralNetwork
from neural_network.distributed.layers import Convolution, FullyConnected, Softmax
from secret_sharing.additive_secret_sharing import create_shares
from secret_sharing.consensus import consensus_confirmation, accept_byzantine, collect_next

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class ModelOwner(Collector):

    def __init__(self, num_groups: int, exp, empty_like, abs, num_mediators: int, output_mask, random_fct, recollector):
        self.comm_evaluator = CommunicationEvaluator()
        # self.nodes is initialized to None and set via 'set_mediators' to resolve circular dependency
        Collector.__init__(self, node_id=None, byzantine=False, group_sizes=num_mediators, nodes=None, comm_evaluator=self.comm_evaluator, byzantine_dict=util.constants.byzantine_setup['mediator'], recollector=recollector, collection_strategy='rr', collection_mode='single')
        self.sec_shares = []
        self._exp = exp
        self._empty_like = empty_like
        self._abs = abs
        self.num_mediators = num_mediators
        self.num_groups = num_groups
        self._output_mask = output_mask
        self._random_fct = random_fct
        self.auxiliary_matrices = {}
        self.auxiliary_mul_nums = None
        self.auxiliary_cmp_num = None

        self.sec_result = Queue(maxsize=1)
        self.sec_shares_cons = None
        self.collection_iteration = 0
        self.collection_ids = self.apply_collection_strategy()

    def set_mediators(self, mediator_nodes):
        self.nodes = mediator_nodes

    def create_models(self, nn: DistributedNeuralNetwork, num_groups: int):
        """
        Function to create a list of :class:`neural_network.DistributedNeuralNetwork`, one for each group of
        :class:`actors.WorkerNode`.

        Duplicates the layers of the given :class:`neural_network.DistributedNeuralNetwork`. The weights of each layer
        are split into additive secret shares and used in the :class:`neural_network.DistributedNeuralNetwork` created for
        each group.

        Parameters
        ----------
        nn:
            A :class:`neural_network.DistributedNeuralNetwork` with initialized layers.
        num_groups:
            The number of groups used, i.e. the number of duplicates and secret shares to create.

        Returns
        -------
        models: list of :class:`neural_network.DistributedNeuralNetwork`
            A list of :class:`neural_network.DistributedNeuralNetwork` containing one network for each group (i.e. a total
            of ``num_groups`` elements). Each created network uses additive secret shares in the trainable layers
            (:class:`neural_network.distributed.layers.FullyConnected`, :class:`neural_network.distributed.layers.Convolution`),
            created from the parameters used in the passed network ``nn``.
        """

        models = []
        params = []

        for layer in nn.layers:
            if type(layer) == Convolution:
                F_shares = create_shares(layer.F, num_groups, self._random_fct)
                b_shares = create_shares(layer.b, num_groups, self._random_fct)
                params.append((F_shares, b_shares))
            elif type(layer) == FullyConnected:
                W_shares = create_shares(layer.W, num_groups, self._random_fct)
                b_shares = create_shares(layer.b, num_groups, self._random_fct)
                params.append((W_shares, b_shares))
            elif type(layer) == Softmax:
                output_mask_shares = create_shares(layer.output_mask, num_groups, self._random_fct)
                params.append(output_mask_shares)
            else:
                params.append(None)

        for i in range(num_groups):
            new_nn = DistributedNeuralNetwork(nn.auxiliary_nodes, nn.mediator_nodes, i, nn.ran_group, lr=nn.lr)
            for j, layer in enumerate(nn.layers):
                clone = layer.clone(i)
                if type(layer) == Convolution:
                    F_shares, b_shares = params[j]
                    clone.F = F_shares[i]
                    clone.b = b_shares[i]
                elif type(layer) == FullyConnected:
                    W_shares, b_shares = params[j]
                    clone.W = W_shares[i]
                    clone.b = b_shares[i]
                elif type(layer) == Softmax:
                    output_mask_shares = params[j]
                    clone.output_mask = output_mask_shares[i]
                new_nn.layers.append(clone)
            models.append(new_nn)

        return models

    async def support_softmax(self, node_id, x, iteration, batch=False, byzantine=False):
        """
        Support function for softmax activation (https://en.wikipedia.org/wiki/Softmax_function). Collects the masked
        outputs ``x`` from all mediator nodes and performs a consensus confirmation. The consensus-confirmed result is
        unmasked and used as the input to the softmax activation.

        Parameters
        ----------
        x:
            Masked share of an output of some layer.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``x`` has to be a list and the
            result will be list of softmax activations of the elements of ``x``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        softmax_activations:
            A tensor of softmax activations for the received layer output, or a list of the like if ``batch==True``.
        """

        self.comm_evaluator.msgs += 1
        if batch:
            msg_size = get_tensor_size(x[0]) * len(x)
        else:
            msg_size = get_tensor_size(x)
        self.comm_evaluator.msg_size += msg_size

        return await self._generic_support(node_id, x, iteration, self._output_mask, self._softmax_result_fct, msg_size, batch, byzantine)

    async def update_model(self, delta_W):
        return None

    def get_cmp_auxiliary_share(self, group_num: int):
        """
        Function to provide auxiliary shares for a secure comparison (additive secret shares of a positive number).

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.

        Returns
        -------
        share_random_pos:
            An additive secret share, belonging to group ``group_num``, of a random positive number.
        """

        # self.comm_evaluator.msgs += 1
        # self.comm_evaluator.msg_size += sys.getsizeof(group_num)

        if self.auxiliary_cmp_num is not None:
            return self.auxiliary_cmp_num[group_num]
        else:
            rand_pos_num = self._abs(self._random_fct(1))
            shares = secret_sharing.additive_secret_sharing.create_shares(rand_pos_num, self.num_groups, self._random_fct)
            self.auxiliary_cmp_num = shares
            return shares[group_num]

    def get_mul_auxiliary_shares(self, group_num: int):
        """
        Function to provide auxiliary shares for a secure multiplication (additive secret shares of a random beaver triple
        of numbers).

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.

        Returns
        -------
        share_random_num:
            Additive secret shares, belonging to group ``group_num``, of a random beaver triple of numbers.
        """

        # self.comm_evaluator.msgs += 1
        # self.comm_evaluator.msg_size += sys.getsizeof(group_num)

        if self.auxiliary_mul_nums is not None:
            return self.auxiliary_mul_nums[group_num]
        else:
            shares = self._create_auxiliary_shares(1, 1)
            self.auxiliary_mul_nums = shares
            return shares[group_num]

    def get_mat_mul_auxiliary_shares(self, group_num: int, X_dim, Y_dim):
        """
        Function to provide auxiliary shares for a secure matrix multiplication ``X @ Y`` with matrices of size ``X_dim``
        and ``Y_dim`` (additive secret shares of a random beaver triple of matrices).

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        X_dim:
            The shape of the left matrix multiplicand.
        Y_dim:
            The shape of the right matrix multiplicand.

        Returns
        -------
        share_random_num:
            Additive secret shares, belonging to group ``group_num``, of a random beaver triple of matrices.
        """

        # self.comm_evaluator.msgs += 1
        # self.comm_evaluator.msg_size += sys.getsizeof(group_num) + sys.getsizeof(X_dim) + sys.getsizeof(Y_dim)

        if (X_dim, Y_dim) in self.auxiliary_matrices.keys():
            return self.auxiliary_matrices[(X_dim, Y_dim)][group_num]
        else:
            shares = self._create_auxiliary_shares(X_dim, Y_dim)
            self.auxiliary_matrices[(X_dim, Y_dim)] = shares
            return shares[group_num]

    def get_comm_cost(self):
        return self.comm_evaluator.msgs, self.comm_evaluator.msg_size

    def _create_auxiliary_shares(self, X_dim, Y_dim):
        """
        Creates auxiliary secret shares (of a beaver triple) to perform multiplication / matrix multiplication.
        """

        A = self._random_fct(X_dim)
        B = self._random_fct(Y_dim)
        C = A @ B  # simple multiplication if X_dim == Y_dim == 1

        A_shares = create_shares(A, self.num_groups, self._random_fct)
        B_shares = create_shares(B, self.num_groups, self._random_fct)
        C_shares = create_shares(C, self.num_groups, self._random_fct)

        return [t for t in zip(A_shares, B_shares, C_shares)]

    async def _generic_support(self, node_id, secret, iteration, mask, result_fct, msg_size, batch=False, byzantine=False):
        """
        Generic support function to collect values sent by the mediator nodes (masked output of some layer or model update).
        The collected values are consensus confirmed and unmasked. The function ``result_fct`` is called, specifying the
        unmasked value(s) and ``batch`` (e.g. to apply the softmax activation function or a model update). The result
        of this call (e.g. shares of the softmax activation or shares of new model parameters) is stored in the result queue.

        This function will block if not all shares of all groups were received to avoid callbacks.

        Parameters
        ----------
        secret:
            A masked secret of a mediator node. Secrets of all mediators will be saved in the ``sec_shares`` dictionary
            inherited by :class:`actors.abstract.Collector`. If all secrets were collected, a consensus confirmation is performed,
            after which the function ``result_fct`` is called whose result is placed in the ``sec_result`` dictionary.
        mask:
            The mask used for the received secret.
        result_fct:
            Function to be applied to the unmasked value(s) (softmax activation function / new model parameters).
        batch:
            Toggle between single (False) and batched (True) support.

        Returns
        -------
        result:
            The result of the specified function ``result_fct``.
        """

        await self._reset_or_wait(iteration)

        await self._save_shares(secret, None, byzantine)

        ready = await self._check_ready(msg_size, sender_group='model_owner', _=None)

        # reconstruct secret (output of last neurons / weight update) if ready;
        if ready:
            if batch:
                secret = [sec_shares_cons + mask for sec_shares_cons in self.sec_shares_cons]
            else:
                secret = self.sec_shares_cons + mask

            result_shares = result_fct(secret, batch)

            if cfg.optimize_communication:
                await self.send_to_uncollected_nodes(result_shares, shares=False, sender_group='model_owner', iteration=iteration, send_byzantine=False)

            await self.sec_result.put(result_shares)

        result = await self._await_result()

        if cfg.optimize_communication:
            new_collection_ids = self.apply_collection_strategy(iteration + 1 if util.constants.update_collection_strategy else 0)
            receive_next = node_id not in new_collection_ids
        else:
            receive_next = None

        return result, receive_next

    async def _reset_or_wait(self, _):
        if cfg.optimize_communication:
            results_to_deliver = len(self.collection_ids)
        else:
            results_to_deliver = self.num_mediators

        if not self.sec_result.empty() and self.results_delivered < results_to_deliver:
            await self.sec_result.put(None)  # queue max size is 1 -> call will block until queue is empty
            await self.sec_result.get()
        elif not self.sec_result.empty():
            await self._reset_helper_vars()

    async def _await_result(self):
        # wait until result queue is not empty (-> method can be called in parallel!)
        result = await self.sec_result.get()
        # re-queue result
        await self.sec_result.put(result)
        self.results_delivered = self.results_delivered + 1

        return result

    async def _reset_helper_vars(self):
        await self.sec_result.get()
        self.sec_shares = []
        self.sec_shares_cons = None
        self.results_delivered = 0
        self.collection_iteration += 1
        if util.constants.update_collection_strategy:
            self.collection_ids = self.apply_collection_strategy()

    async def _save_shares(self, shares, _, byzantine):
        self.sec_shares.append((shares, byzantine))

    async def _check_ready(self, msg_size, sender_group, _):
        """
        Check if shares of all mediator nodes have been received.
        """
        if cfg.optimize_communication:
            ready = len(self.sec_shares) == len(self.collection_ids)
        else:
            ready = len(self.sec_shares) == self.num_mediators

        if ready:
            # do consensus confirmation
            await self._consensus_confirm(g_size=self.num_mediators, i=None, nodes=self.nodes, msg_size=msg_size, sender_group=sender_group)

        return ready

    async def _consensus_confirm(self, g_size, i, nodes, msg_size, sender_group, iteration=None, collection_ids=None):
        cc_shares = consensus_confirmation(self.sec_shares)
        if cc_shares is None:
            _next_node_idx = next_node(self.collection_ids, self.num_mediators)
            if _next_node_idx is None:
                # no nodes left to collect from -> accept some Byzantine value
                self.sec_shares_cons = accept_byzantine(self.sec_shares)
            else:
                self.comm_evaluator.msgs += 1
                self.comm_evaluator.msg_size += msg_size

                collected = collect_next(self.sec_shares, sim_byzantine=self.byzantine_dict[_next_node_idx], node=self.recollector, receiver=False)
                self.sec_shares.append(collected)
                await self._consensus_confirm(g_size, i, nodes, msg_size, sender_group, iteration, collection_ids)
        else:
            self.sec_shares_cons = cc_shares

    def _softmax_result_fct(self, output_neurons, batch=False):
        if batch:
            result = [[] for _ in range(self.num_groups)]
            for _out in output_neurons:
                s = self._exp(_out).reshape(-1).sum(0)
                softmax = self._exp(_out) / s
                softmax_shares = create_shares(softmax, self.num_groups, self._random_fct)
                for i in range(self.num_groups):
                    result[i].append(softmax_shares[i])
        else:
            s = self._exp(output_neurons).reshape(-1).sum(0)
            softmax = self._exp(output_neurons) / s
            result = create_shares(softmax, self.num_groups, self._random_fct)
        return result
