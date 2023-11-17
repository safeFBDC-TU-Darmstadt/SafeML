import ray

import configuration as cfg
import util
from actors.abstract import Collector
from evaluation import CommunicationEvaluator
from evaluation.util import get_tensor_size

cpu_res = cfg.threads_available / util.constants.total_nodes


def _apply_filter(m, n, X, F, F_dim):
    X_part = X[0:F_dim[1], m:m+F_dim[2], n:n+F_dim[3]]
    mult = F * X_part
    return F, X_part, mult


def calc_update(m, n, o, layer_input, partials_prev, F_dim, in_dim):
    dim1 = in_dim[1] - F_dim[2] + 1
    dim2 = in_dim[2] - F_dim[3] + 1

    input_part = layer_input[m, n:n + dim1, o:o + dim2]

    mult = input_part * partials_prev

    return input_part, partials_prev, mult


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class AuxiliaryNode(Collector):

    def __init__(self, group_sizes: list[int], node_id: int, recollector) -> None:
        self.comm_evaluator = CommunicationEvaluator()
        # self.nodes is initialized to None and set via 'set_workers' to resolve circular dependency
        Collector.__init__(self, node_id, util.constants.byzantine_setup['auxiliary'][node_id], group_sizes, None, self.comm_evaluator, byzantine_dict=util.constants.byzantine_setup['worker'], recollector=recollector, collection_strategy='rr', collection_mode='mult')

    async def sec_mul_support(self, group_num: int, node_id: int, iteration, e_i, f_i, batch=False, byzantine=False, send_result=True):
        """
        Collects all additive shares ``e_i``, ``f_i`` from all groups i. ``e_i`` and ``f_i`` are masked shares of some
        tensors `x`, `y` to be multiplied. Consensus confirmation is performed for all ``e_i``, ``f_i`` received by the nodes
        of group i. `e`, `f` are reconstructed from all ``e_i``, ``f_i`` and `g = e * f` is calculated.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        e_i:
            Masked share of some multiplicand (tensor) `x` or a list of the like if ``batch==True``.
        f_i:
            Masked share of some multiplicand (tensor) `y` or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``e_i`` and ``f_i`` have to be
            equal-sized lists and the result will be list of pair-wise support results of the elements of ``zip(e_i, f_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        e, f, g:
            ``(e, f, g)`` where ``e`` is the masked tensor of multiplicand `x`, ``f`` is the masked tensor of multiplicand `y`
            and `g = e * f` or a list of the like if ``batch==True``.
        """

        if batch:
            msg_size = 0
            for _e_i, _f_i in zip(e_i, f_i):
                msg_size += get_tensor_size(_e_i) + get_tensor_size(_f_i)
        else:
            msg_size = get_tensor_size(e_i) + get_tensor_size(f_i)

        if group_num != 0 or node_id != 0:
            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        return await self._generic_support((e_i, f_i), group_num, node_id, iteration, self._reconstruct_mul if not batch else self._reconstruct_mul_batch, msg_size, byzantine, send_result)

    async def sec_mat_mul_support(self, group_num: int, node_id: int, iteration, E_i, F_i, batch=False, byzantine=False, send_result=True):
        """
        Collects all additive shares ``E_i``, ``F_i`` from all groups i. ``E_i`` and ``F_i`` are masked shares of some
        tensors `X`, `Y` for which the matrix multiplication is to be performed. Consensus confirmation is performed for all
        ``E_i``, ``F_i`` received by the nodes of group i. `E`, `F` are reconstructed from all ``E_i``, ``F_i`` and `G = E @ F`
         is calculated.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        E_i:
            Masked share of some matrix multiplicand (tensor) `X` or a list of the like if ``batch==True``.
        F_i:
            Masked share of some matrix multiplicand (tensor) `Y` or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``E_i`` and ``F_i`` have to be
            equal-sized lists and the result will be list of pair-wise support results of the elements of ``zip(E_i, F_i)``.
        byzantine: bool
            Whether the given shares are / contain byzantine values.

        Returns
        -------
        E, F, G:
            ``(E, F, G)`` where ``E`` is the masked tensor of multiplicand `X`, ``F`` is the masked tensor of multiplicand `Y`
            and `G = E @ F` or a list of the like if ``batch==True``.
        """

        if batch:
            msg_size = 0
            for _E_i, _F_i in zip(E_i, F_i):
                msg_size += get_tensor_size(_E_i) + get_tensor_size(_F_i)
        else:
            msg_size = get_tensor_size(E_i) + get_tensor_size(F_i)

        if group_num != 0 or node_id != 0:
            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        return await self._generic_support((E_i, F_i), group_num, node_id, iteration, self._reconstruct_mat_mul if not batch else self._reconstruct_mat_mul_batch, msg_size, byzantine, send_result)

    async def sec_comp_support(self, group_num: int, node_id: int, iteration, t_alpha_i, batch=False, byzantine=False, send_result=True):
        """
        Collects all additive shares ``t_alpha_i`` from all groups i. ``t_alpha_i`` are masked shares of the difference
        `x-y` of some tensors `x` and `y` which are to be compared. Consensus confirmation is performed for all ``t_alpha_i``
        received by the nodes of group i. `t_alpha` is reconstructed from all `t_alpha_i`` which contains the information of
        `sign(x-y)`.

        Parameters
        ----------
        group_num:
            Group number of the node calling this method.
        t_alpha_i:
            Masked share of the difference `x-y` of some tensors `x` and `y` or a list of the like if ``batch==True``.
        batch:
            Toggle between single (False) and batched (True) support. If ``batch==True``, ``t_alpha_i`` has to be
            a list and the result will be list of support results of the elements of ``t_alpha_i``.
        byzantine: bool
            Whether the given share(s) are / contain byzantine values.

        Returns
        -------
        t_alpha:
            The masked tensor `t_alpha = t*(x-y)` of `x-y`, where `sign(t_alpha) = sign(x-y)`, or a list of the like if ``batch==True``.
        """

        if batch:
            msg_size = 0
            for _t_alpha_i in t_alpha_i:
                msg_size += get_tensor_size(_t_alpha_i)
        else:
            msg_size = get_tensor_size(t_alpha_i)

        if group_num != 0 or node_id != 0:
            self.comm_evaluator.msgs += 1
            self.comm_evaluator.msg_size += msg_size

        return await self._generic_support(t_alpha_i, group_num, node_id, iteration, self._reconstruct_cmp if not batch else self._reconstruct_cmp_batch, msg_size, byzantine, send_result)

    def get_comm_cost(self):
        return self.comm_evaluator.msgs, self.comm_evaluator.msg_size

    async def sec_forward_convolution_support(self, group_num: int, node_id: int, iteration, X_i, F_i, F_dim, in_dim, byzantine=False, send_result=True):
        """
        X_i: masked(!) input matrix
        """
        return await self._generic_support((X_i, F_i, F_dim, in_dim), group_num, node_id, iteration, self._reconstruct_forward_convolution, 0, byzantine, send_result)

    async def sec_backward_convolution_support(self, group_num: int, node_id: int, iteration, layer_input, partials_prev, F_dim, in_dim, byzantine=False, send_result=True):
        """
        X_i: masked(!) input matrix
        """
        return await self._generic_support((layer_input, partials_prev, F_dim, in_dim), group_num, node_id, iteration, self._reconstruct_backward_convolution, 0, byzantine, send_result)

    async def _generic_support(self, share, group_num, node_id, iteration, reconstruction_fct, msg_size, byzantine, send_result):
        """
        Generic support function to collect secret shares, reconstruct secrets and queue the reconstructed result.

        This function will block if not all shares of all groups were received to avoid callbacks.

        Parameters
        ----------
        share:
            An additive secret share of group ``group_num``. Shares of all groups will be saved in the ``sec_shares``
            dictionary inherited by :class:`actors.abstract.Collector`. If all shares of a group were collected, a consensus
            confirmation is performed for this group, whose result is placed in the ``sec_shares_cons`` dictionary.
            If the shares of `all` groups were collected (and consensus-confirmed), the secret is reconstructed via
            a callable reconstruction function (see below) and placed in the result queue ``sec_result``.
        group_num:
            The group number of the node calling a support function.
        reconstruction_fct:
            A callable function with no parameters. Reconstructs the collected secret from the consensus-confirmed shares
            stored in ``sec_shares_cons``.
        byzantine: bool
            Whether the given share(s) are / contain byzantine values.

        Returns
        -------
        result:
            The reconstructed secret.
        """

        # reset helper variables or wait until the previous result was received by all other nodes
        await self._reset_or_wait(iteration)

        await self._save_shares(share, group_num, byzantine)

        # check if all shares of all groups were received ('ready' condition)
        ready = await self._check_ready(msg_size, sender_group='auxiliary', iteration=iteration)

        # reconstruct and save results if ready
        if ready:
            res = reconstruction_fct()

            if cfg.optimize_communication and send_result:
                await self.send_to_uncollected_nodes(res, shares=False, sender_group='auxiliary', iteration=iteration, send_byzantine=self.byzantine)

            await self.sec_result.put(res)

        result = await self._await_result()

        await self._optional_reset()

        send_byzantine = self.byzantine
        if cfg.optimize_communication:
            new_collection_ids = self.apply_collection_strategy(iteration + 1 if util.constants.update_collection_strategy else 0)
            receive_next = node_id not in new_collection_ids[group_num]
        else:
            receive_next = None

        return (result, receive_next, send_byzantine) if send_result else None

    def _reconstruct_mul(self):
        """
        Reconstruct e, f, g=e*f from consensus-confirmed additive shares.
        """
        e, f = self._generic_double_reconstruct()
        return e, f, e * f

    def _reconstruct_mat_mul(self):
        """
        Reconstruct E, F, G=E@F from consensus-confirmed additive shares.
        """
        E, F = self._generic_double_reconstruct()
        return E, F, E @ F

    def _reconstruct_cmp(self):
        """
        Reconstruct t*\alpha from consensus-confirmed additive shares.
        """
        shares = []
        for k in self.sec_shares_cons.keys():
            shares.append(self.sec_shares_cons[k])
        return sum(shares)

    def _reconstruct_mul_batch(self):
        """
        Reconstruct all e, f, g=e*f from consensus-confirmed additive shares.
        """
        e, f = self._generic_double_reconstruct_batch()

        g = []
        for e_elem, f_elem in zip(e, f):
            g.append(e_elem * f_elem)

        return e, f, g

    def _reconstruct_mat_mul_batch(self):
        """
        Reconstruct E, F, G=E@F from consensus-confirmed additive shares.
        """
        E, F = self._generic_double_reconstruct_batch()

        G = []
        for E_elem, F_elem in zip(E, F):
            G.append(E_elem @ F_elem)

        return E, F, G

    def _reconstruct_cmp_batch(self):
        """
        Reconstruct t*\alpha from consensus-confirmed additive shares.
        """
        t_alpha = None
        for k in self.sec_shares_cons.keys():
            t_alpha_i = self.sec_shares_cons[k]  # consensus-confirmed list of 't_alpha_i's of group i
            if t_alpha is None:
                t_alpha = t_alpha_i
            else:
                t_alpha = [t_alpha_elem + next_t_alpha_share for t_alpha_elem, next_t_alpha_share in zip(t_alpha, t_alpha_i)]
        return t_alpha

    def _reconstruct_forward_convolution(self):
        X_shares = []
        F_shares = []
        F_dim, in_dim, = None, None
        for k in self.sec_shares_cons.keys():
            X_share, F_share, F_dim, in_dim = self.sec_shares_cons[k]
            X_shares.append(X_share)
            F_shares.append(F_share)
        X = sum(X_shares)
        F = sum(F_shares)

        results = []

        for m in range(in_dim[1] - F_dim[2] + 1):
            for n in range(in_dim[2] - F_dim[3] + 1):
                results.append(_apply_filter(m, n, X, F, F_dim))

        return results

    def _reconstruct_backward_convolution(self):
        layer_input_shares = []
        partials_prev_shares = []
        F_dim, in_dim, = None, None
        for k in self.sec_shares_cons.keys():
            layer_input_share, partials_prev_share, F_dim, in_dim = self.sec_shares_cons[k]
            layer_input_shares.append(layer_input_share)
            partials_prev_shares.append(partials_prev_share)
        layer_input = sum(layer_input_shares)
        partials_prev = sum(partials_prev_shares)

        results = []

        for m in range(self.F_dim[1]):
            for n in range(self.F_dim[2]):
                for o in range(self.F_dim[3]):
                    results.append(calc_update(m, n, o, layer_input, partials_prev, F_dim, in_dim))

        return results

    def _generic_double_reconstruct(self):
        x_shares = []
        y_shares = []
        for k in self.sec_shares_cons.keys():
            x_share, y_share = self.sec_shares_cons[k]
            x_shares.append(x_share)
            y_shares.append(y_share)
        x = sum(x_shares)
        y = sum(y_shares)
        return x, y

    def _generic_double_reconstruct_batch(self):
        x = None
        y = None
        shares_initialized = False
        for k in self.sec_shares_cons.keys():
            x_shares_i, y_shares_i = self.sec_shares_cons[k]  # consensus-confirmed list of 'x_i's and 'y_i's of group i
            if not shares_initialized:
                x = x_shares_i
                y = y_shares_i
                shares_initialized = True
            else:
                # element-wise addition for each list (batch) of secret shares
                x = [x_elem + next_x_share for x_elem, next_x_share in zip(x, x_shares_i)]
                y = [y_elem + next_y_share for y_elem, next_y_share in zip(y, y_shares_i)]
        return x, y
