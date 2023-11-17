from asyncio import Lock

import ray

import configuration
import util
from actors.util import next_node
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size
from secret_sharing.consensus import consensus_confirmation_receiver, collect_next


class SecComp:
    """
    A class used to provide secure computations (multiplication, matrix multiplication, comparison) over additive secret shares.
    """

    _auxiliary_nodes: []
    _group_num: int

    def __init__(self, auxiliary_nodes, mediator_nodes, model_owner, group_num, node_id, worker_node, recollector) -> None:
        """
        Parameters
        ----------
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        model_owner:
            The :class:`actors.ModelOwner` ray stub.
        group_num:
            The group number to use for the secure computations.
        worker_node:
            The worker node using this object.
        """

        self.online_comm_evaluator = CommunicationEvaluator()

        self._auxiliary_nodes = auxiliary_nodes
        self._mediator_nodes = mediator_nodes
        self._model_owner = model_owner
        self._group_num = group_num
        self.worker_node = worker_node
        self.node_id = node_id

        # auxiliary values for secure multiplication
        self.a_i = None
        self.b_i = None
        self.c_i = None

        # auxiliary matrices for secure matrix multiplication (for different matrix sizes)
        self.A_i = {}
        self.B_i = {}
        self.C_i = {}

        # auxiliary positive number for secure comparison
        self.t_i = None

        self.iteration_lock = Lock()

        self.recollector = recollector

    async def sec_mul(self, x_i, y_i, batch=False, byzantine=False):
        """
        Method to perform a secure multiplication with the shares ``x_i`` and ``y_i`` of some tensors `x` and `y`.

        This method uses the support function `actors.AuxiliaryNode.sec_mul_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain a share ``xy_i`` of `x*y`.

        Parameters
        ----------
        x_i:
            An additive share (tensor) of some other tensor `x` or a list of the like if ``batch==True``.
        y_i:
            An additive share (tensor) of some other tensor `y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure multiplications. If ``batch==True``, ``x_i`` and ``y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure multiplications of the elements of ``zip(x_i, y_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        xy_i:
            An additive share (tensor) of `x*y` or a list of the like if ``batch==True``.
        """

        if self.a_i is None:
            self.a_i, self.b_i, self.c_i = ray.get(self._model_owner.get_mul_auxiliary_shares.remote(self._group_num))

        if batch:
            e_i = [_x_i - self.a_i for _x_i in x_i]
            f_i = [_y_i - self.b_i for _y_i in y_i]
        else:
            e_i = x_i - self.a_i
            f_i = y_i - self.b_i

        results = await self.send_or_receive_mul(e_i, f_i, batch, byzantine)

        result, receive_next, extra_collections = self._get_auxiliary_results(results)

        (e, f, g), self.worker_node.receiver(sender_group='auxiliary').receive_next = result, receive_next

        if batch:
            xy_i = []
            for _e, _f, _g in zip(e, f, g):
                _xy_i = self.c_i + self.b_i * _e + self.a_i * _f
                if self._group_num == 0:
                    _xy_i = _xy_i + _g
                xy_i.append(_xy_i)
        else:
            xy_i = self.c_i + self.b_i * e + self.a_i * f
            if self._group_num == 0:
                xy_i = xy_i + g

        if batch:
            e_size = get_tensor_size(e[0]) * len(e)
            f_size = get_tensor_size(f[0]) * len(f)
        else:
            e_size = get_tensor_size(e)
            f_size = get_tensor_size(f)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * (e_size + f_size)

        return xy_i

    async def sec_mat_mul(self, X_i, Y_i, batch=False, byzantine=False):
        """
        Function to perform a secure matrix multiplication with the shares ``X_i`` and ``Y_i`` of some tensors `X` and `Y`.

        This function uses the support function `actors.AuxiliaryNode.sec_mat_mul_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain a share ``XY_i`` of `X@Y`.

        Parameters
        ----------
        X_i:
            An additive share (tensor) of some other tensor `X` or a list of the like if ``batch==True``.
        Y_i:
            An additive share (tensor) of some other tensor `Y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure matrix multiplications. If ``batch==True``, ``X_i`` and ``Y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure matrix multiplications of the elements of ``zip(X_i, Y_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        XY_i:
            An additive share (tensor) of `X@Y` or a list of the like if ``batch==True``.
        """

        if batch:
            X_i_shape = X_i[0].shape
            Y_i_shape = Y_i[0].shape
        else:
            X_i_shape = X_i.shape
            Y_i_shape = Y_i.shape

        if (X_i_shape, Y_i_shape) not in self.A_i.keys():
            _A_i, _B_i, _C_i = ray.get(self._model_owner.get_mat_mul_auxiliary_shares.remote(self._group_num, X_i_shape, Y_i_shape))
            self.A_i[(X_i_shape, Y_i_shape)] = _A_i
            self.B_i[(X_i_shape, Y_i_shape)] = _B_i
            self.C_i[(X_i_shape, Y_i_shape)] = _C_i
        else:
            _A_i = self.A_i[(X_i_shape, Y_i_shape)]
            _B_i = self.B_i[(X_i_shape, Y_i_shape)]
            _C_i = self.C_i[(X_i_shape, Y_i_shape)]

        if batch:
            E_i = [_X_i - _A_i for _X_i in X_i]
            F_i = [_Y_i - _B_i for _Y_i in Y_i]
        else:
            E_i = X_i - _A_i
            F_i = Y_i - _B_i

        results = await self.send_or_receive_mat_mul(E_i, F_i, batch, byzantine)

        result, receive_next, extra_collections = self._get_auxiliary_results(results)

        (E, F, G), self.worker_node.receiver(sender_group='auxiliary').receive_next = result, receive_next

        if batch:
            XY_i = []
            for _E, _F, _G in zip(E, F, G):
                _XY_i = _C_i + _E @ _B_i + _A_i @ _F
                if self._group_num == 0:
                    _XY_i = _XY_i + _G
                XY_i.append(_XY_i)
        else:
            XY_i = _C_i + E @ _B_i + _A_i @ F
            if self._group_num == 0:
                XY_i = XY_i + G

        if batch:
            E_size = get_tensor_size(E[0]) * len(E)
            F_size = get_tensor_size(F[0]) * len(F)
        else:
            E_size = get_tensor_size(E)
            F_size = get_tensor_size(F)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * (E_size + F_size)

        return XY_i

    async def sec_cmp(self, x_i, y_i, batch=False, byzantine=False):
        """
        Function to perform a secure element-wise comparison with the shares ``x_i`` and ``y_i`` of some tensors `x` and `y`.

        This function uses the support function `actors.AuxiliaryNode.sec_comp_support` of all :class:`actors.AuxiliaryNode` stubs via RPC to obtain `t * alpha = t * (x - y)`.

        Parameters
        ----------
        x_i:
            An additive share (tensor) of some other tensor `x` or a list of the like if ``batch==True``.
        y_i:
            An additive share (tensor) of some other tensor `y` or a list of the like if ``batch==True``.
        batch: bool
            Toggle between single (False) and batched (True) secure comparisons. If ``batch==True``, ``x_i`` and ``y_i`` have to be
            equal-sized lists and the result will be list of pair-wise secure comparisons of the elements of ``zip(x_i, y_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        t_alpha:
            An tensor `t * alpha = t * (x - y)` where `sign(t*alpha) = sign(x-y)` or a list of the like if ``batch==True``.
        """

        if self.t_i is None:
            self.t_i = ray.get(self._model_owner.get_cmp_auxiliary_share.remote(self._group_num))

        if batch:
            alpha_i = [_x_i - _y_i for _x_i, _y_i in zip(x_i, y_i)]
            t_i = [self.t_i for _ in range(len(alpha_i))]
        else:
            alpha_i = x_i - y_i
            t_i = self.t_i

        t_alpha_i = await self.sec_mul(t_i, alpha_i, batch, byzantine)

        results = await self.send_or_receive_comp(t_alpha_i, batch, byzantine)

        result, receive_next, extra_collections = self._get_auxiliary_results(results)

        t_alpha, self.worker_node.receiver(sender_group='auxiliary').receive_next = result, receive_next

        if batch:
            t_alpha_size = get_tensor_size(t_alpha[0]) * len(t_alpha)
        else:
            t_alpha_size = get_tensor_size(t_alpha)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * t_alpha_size

        return t_alpha

    async def send_or_receive_mul(self, e_i, f_i, batch, byzantine):
        async with self.iteration_lock:
            if configuration.optimize_communication and self.worker_node.receive_next(sender_group='auxiliary'):
                result = await self.worker_node.await_result((e_i, f_i), byzantine, self.worker_node.iteration_auxiliaries, sender_group='auxiliary')
            else:
                result = ray.get([a.sec_mul_support.remote(self._group_num, self.node_id, self.worker_node.iteration_auxiliaries, e_i, f_i, batch, byzantine, send_result=i in self.worker_node.receive_from_auxiliaries if configuration.optimize_communication else True) for i, a in enumerate(self._auxiliary_nodes)])
                result = [res for res in result if res is not None]
            return result

    async def send_or_receive_mat_mul(self, E_i, F_i, batch, byzantine):
        async with self.iteration_lock:
            if configuration.optimize_communication and self.worker_node.receive_next(sender_group='auxiliary'):
                result = await self.worker_node.await_result((E_i, F_i), byzantine, self.worker_node.iteration_auxiliaries, sender_group='auxiliary')
            else:
                result = ray.get([a.sec_mat_mul_support.remote(self._group_num, self.node_id, self.worker_node.iteration_auxiliaries, E_i, F_i, batch, byzantine, send_result=i in self.worker_node.receive_from_auxiliaries if configuration.optimize_communication else True) for i, a in enumerate(self._auxiliary_nodes)])
                result = [res for res in result if res is not None]
            return result

    async def send_or_receive_comp(self, t_alpha_i, batch, byzantine):
        async with self.iteration_lock:
            if configuration.optimize_communication and self.worker_node.receive_next(sender_group='auxiliary'):
                result = await self.worker_node.await_result(t_alpha_i, byzantine, self.worker_node.iteration_auxiliaries, sender_group='auxiliary')
            else:
                result = ray.get([a.sec_comp_support.remote(self._group_num, self.node_id, self.worker_node.iteration_auxiliaries, t_alpha_i, batch, byzantine, send_result=i in self.worker_node.receive_from_auxiliaries if configuration.optimize_communication else True) for i, a in enumerate(self._auxiliary_nodes)])
                result = [res for res in result if res is not None]
            return result

    def _get_auxiliary_results(self, results):
        result, receive_next = consensus_confirmation_receiver(results)
        extra_collections = 0
        while result is None:
            next_aux_id = next_node(self.worker_node.receive_from_auxiliaries, util.constants.num_auxiliaries)
            next_aux_res = collect_next(results, self.worker_node.auxiliary_byzantine_dict[next_aux_id], self.recollector, receiver=True)
            results.append(next_aux_res)
            extra_collections += 1
            result, receive_next = consensus_confirmation_receiver(results)
        self.worker_node.increase_auxiliary_iteration()
        return result, receive_next, extra_collections

    async def sec_softmax(self, x, output_mask, batch=False, byzantine=False):
        if batch:
            x_masked = [_x - output_mask for _x in x]
        else:
            x_masked = x - output_mask

        results = await self._send_or_receive_softmax(x_masked, batch, byzantine)

        softmax, self.worker_node.receiver(sender_group='mediator').receive_next, extra_collections = self._get_mediator_results(results)

        if batch:
            softmax_size = get_tensor_size(softmax[0]) * len(softmax)
        else:
            softmax_size = get_tensor_size(softmax)

        self.online_comm_evaluator.msgs += len(results) - 1
        self.online_comm_evaluator.msg_size += (len(results) - 1) * softmax_size

        return softmax

    async def _send_or_receive_softmax(self, x_masked, batch, byzantine):
        if configuration.optimize_communication and self.worker_node.receive_next(sender_group='mediator'):
            await self.worker_node.receiver(sender_group='mediator').await_entry()
        if configuration.optimize_communication and self.worker_node.receive_next(sender_group='mediator'):
            result = await self.worker_node.await_result(x_masked, byzantine, iteration=self.worker_node.iteration_mediators, sender_group='mediator')
        else:
            result = ray.get([m.support_softmax.remote(self._group_num, self.worker_node.node_id, self.worker_node.iteration_mediators, x_masked, batch, byzantine=byzantine, send_result=i in self.worker_node.receive_from_mediators if configuration.optimize_communication else True) for i, m in enumerate(self._mediator_nodes)])
            result = [res for res in result if res is not None]
        return result

    def _get_mediator_results(self, results):
        result, receive_next = consensus_confirmation_receiver(results)
        extra_collections = 0
        while result is None:
            next_med_id = next_node(self.worker_node.receive_from_mediators, util.constants.num_mediators)
            next_med_res = collect_next(results, self.worker_node.mediator_byzantine_dict[next_med_id], self.recollector, receiver=True)
            results.append(next_med_res)
            extra_collections += 1
            result, receive_next = consensus_confirmation_receiver(results)
        self.worker_node.increase_mediator_iteration()
        return result, receive_next, extra_collections
