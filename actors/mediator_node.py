import sys

import ray

import configuration as cfg
import util
from actors.abstract import Collector, AbstractNode, Receiver, ReceiverManager
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class MediatorNode(Collector, AbstractNode, ReceiverManager):

    def __init__(self, group_sizes: list[int], model_owner, node_id: int, receive_next: bool, recollector) -> None:
        self.comm_evaluator = CommunicationEvaluator()
        # self.nodes is initialized to None and set via 'set_workers' to resolve circular dependency
        Collector.__init__(self, node_id, util.constants.byzantine_setup['mediator'][node_id], group_sizes, None, self.comm_evaluator, byzantine_dict=util.constants.byzantine_setup['worker'], recollector=recollector, collection_strategy='rr', collection_mode='mult')

        # set up receiver
        model_owner_receiver = Receiver(max_results=1, receive_next=receive_next, node_id=node_id)
        receivers = {'model_owner': model_owner_receiver}
        ReceiverManager.__init__(self, receivers)
        self.model_owner_iterations = 0

        self._model_owner = model_owner

    async def support_softmax(self, group_num: int, node_id: int, iteration, x_i, batch=False, byzantine=False, send_result=True):
        """
        Collects the shares ``x_i``, masked shares of the output of some layer, of all groups i. Consensus confirmation
        is performed for each group once every share has been received. The masked output is reconstructed from the
        consensus-confirmed shares and sent to the model owner.

        The model owner will unmask the output, calculate the softmax activation and send shares of this result back to
        the mediator nodes. Each mediator node will respond to the worker node with the share belonging to group ``group_num``.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        x_i:
            Masked share of the output of some layer or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``x_i`` has to be
            a list and the result will be list of support results of the elements of ``x_i``.

        Returns
        -------
        result:
            A share of the softmax activation, belonging to group ``group_num``, of the reconstructed layer output.
        """

        if batch:
            msg_size = get_tensor_size(x_i[0]) * len(x_i)
        else:
            msg_size = get_tensor_size(x_i)

        if group_num != 0 or node_id != 0:
            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        return await self._generic_support(group_num, node_id, iteration, x_i, 'support_softmax', msg_size, batch, byzantine, send_result)

    async def support_update(self, group_num: int, node_id: int, iteration, delta_W):
        self.comm_evaluator.msgs += 1
        msg_size = get_tensor_size(delta_W) + sys.getsizeof(group_num)
        self.comm_evaluator.msg_size += msg_size
        return await self._generic_support(group_num, node_id, iteration, delta_W, 'update_model', msg_size)

    def get_comm_cost(self):
        return self.comm_evaluator.msgs, self.comm_evaluator.msg_size

    async def _generic_support(self, group_num: int, node_id: int, iteration, share, model_owner_support_fct: str, msg_size, batch=False, byzantine=False, send_result=True):
        """
        Generic support function to collect secret shares and reconstruct secrets. The reconstructed secret (masked
        layer output or model update) is sent to the model owner using the specified function ``model_owner_support_fct``.
        The model owner will respond with shares (softmax activation or new model parameters) for all groups.

        This function will block if not all shares of all groups were received to avoid callbacks.

        Parameters
        ----------
        share:
            An additive secret share of group ``group_num``. Shares of all groups will be saved in the ``sec_shares``
            dictionary inherited by :class:`actors.abstract.Collector`. If all shares of a group were collected, a consensus
            confirmation is performed for this group, whose result is placed in the ``sec_shares_cons`` dictionary.
            If the shares of `all` groups were collected (and consensus-confirmed), the secret is reconstructed via
            a callable reconstruction function (see below). The reconstructed secret is sent to the model owner, whose
            response is placed in the result queue ``sec_result``.
        group_num:
            The group number of the node calling a support function.
        model_owner_support_fct:
            Specifies the support function to be called on the :class:`actors.ModelOwner` stub. One of {``'support_softmax'``,
            ``'update_model'``}.
        batch:
            Toggle between single (False) and batched (True) support on the :class:`actors.ModelOwner` stub.

        Returns
        -------
        result:
            The result of the specified function ``model_owner_support_fct`` belonging to group ``group_num``.
        """

        await self._reset_or_wait(iteration)

        await self._save_shares(share, group_num, byzantine)

        ready = await self._check_ready(msg_size, sender_group='mediator', iteration=iteration)

        # reconstruct secret (output of last neurons / weight update) if ready;
        # save result shares received from model owner
        if ready:
            secret = await self._reconstruct_out(batch)
            if model_owner_support_fct == 'support_softmax':
                out_shares = await self._send_or_receive_softmax(secret, batch, byzantine)
            elif model_owner_support_fct == 'update_model':
                out_shares = await self._send_or_receive_update_model(secret, byzantine)
            else:
                raise Exception('Unknown support function.')

            if cfg.optimize_communication and send_result:
                await self.send_to_uncollected_nodes(out_shares, shares=True, sender_group='mediator', iteration=iteration, send_byzantine=self.byzantine)

            await self.sec_result.put(out_shares)

        result = await self._await_result()

        await self._optional_reset()

        send_byzantine = self.byzantine
        if cfg.optimize_communication:
            new_collection_ids = self.apply_collection_strategy(iteration + 1 if util.constants.update_collection_strategy else 0)
            receive_next = node_id not in new_collection_ids[group_num]
        else:
            receive_next = None

        return (result[group_num], receive_next, send_byzantine) if send_result else None

    async def _send_or_receive_softmax(self, secret, batch, byzantine):
        if cfg.optimize_communication and self.receive_next('model_owner'):
            result = await self.await_result(secret, byzantine, self.model_owner_iterations, sender_group='model_owner')
            result = result[0]  # do not need consensus confirmation because result is sent from trusted actor (ModelOwner)
            result, self.receiver(sender_group='model_owner').receive_next, _ = result
        else:
            send_byzantine = self.byzantine
            result, self.receiver(sender_group='model_owner').receive_next = ray.get(self._model_owner.support_softmax.remote(self.node_id, secret, self.model_owner_iterations, batch, byzantine=send_byzantine))
        self.model_owner_iterations += 1
        return result

    async def _send_or_receive_update_model(self, secret, byzantine):
        if cfg.optimize_communication and self.receive_next:
            result = await self.await_result(secret, byzantine, self.model_owner_iterations, sender_group='model_owner')
        else:
            result, receive_next = ray.get(self._model_owner.update_model.remote(secret))
            self.receive_next = receive_next
        self.model_owner_iterations += 1
        return result

    async def _reconstruct_out(self, batch):
        """
        Reconstructs a tensor from the consensus-confirmed shares.

        TODO: support (list of) gradients
        """

        if batch:
            return self._reconstruct_out_batch()
        shares = []
        for k in self.sec_shares_cons.keys():
            shares.append(self.sec_shares_cons[k])
        return sum(shares)

    def _reconstruct_out_batch(self):
        """
        Reconstructs a list of tensors from the consensus-confirmed shares.

        TODO: support (list of) gradients
        """

        out_shares = None
        for k in self.sec_shares_cons.keys():
            out_shares_i = self.sec_shares_cons[k]
            if out_shares is None:
                out_shares = out_shares_i
            else:
                out_shares = [out_shares_elem + next_out_share for out_shares_elem, next_out_share in zip(out_shares, out_shares_i)]
        return out_shares
