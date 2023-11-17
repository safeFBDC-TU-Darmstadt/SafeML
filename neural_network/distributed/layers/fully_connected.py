from neural_network.distributed.layers import Layer


class FullyConnected(Layer):
    """
    A fully connected layer of a neural network.
    """

    _auxiliary_nodes: []
    _group_num: int

    def __init__(self, batch_size, lr, W, b, calc_partials, empty_like, diag, zeros_like, auxiliary_nodes, group_num, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The size of the batches that will be processed.
        lr: float
            The learning rate of the layer.
        W:
            The weight tensor of the layer. Must be a 2D tensor.
        b:
            The bias tensor of the layer. Must be a 1D tensor.
        empty_like: callable
            A function that returns an empty tensor with the same shape and data type as the input tensor.
        diag: callable
            A function that returns a 2D tensor with the input ndarray on the diagonal.
        zeros_like: callable
            A function that returns a tensor of zeros with the same shape and data type as the input tensor.
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine)
        self.W = W  # weight (share) matrix
        self.b = b  # bias (share) matrix
        self._gradient = None
        self._empty_like = empty_like
        self._zeros_like = zeros_like
        self._diag = diag
        self._input = None
        self._auxiliary_nodes = auxiliary_nodes
        self._group_num = group_num
        self.calc_partials = calc_partials

    async def forward(self, x):
        """
        Compute the forward pass of the fully connected layer.

        Parameters
        ----------
        x:
            A share of the input data for the forward pass. Might be a tensor or a list of tensors (batched input). The input
            tensor(s) can have any dimensions.

        Returns
        -------
        output:
            A share of the output of the forward pass. If ``x`` is a list of tensors (batched input), ``output`` is a
            list of tensors with the same length as ``x``.
        """

        batch = type(x) == list
        if batch:
            for i in range(len(x)):
                x[i] = x[i].reshape(-1)
            self._input = x
            W = [self.W for _ in range(len(x))]  # TODO: no replication
        else:
            x = x.reshape(-1)
            self._input = x
            W = self.W

        result = await self.sec_comp.sec_mat_mul(W, x, batch, self.byzantine)

        if batch:
            for i in range(len(result)):
                result[i] += self.b
        else:
            result += self.b

        return result

    async def backward(self, partials_prev):
        """
        Compute the backward pass of the fully connected layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output of the layer. Might be a tensor or a list of
            tensors (batched backward pass). The given tensor(s) can have any dimensions.


        Returns
        -------
        new_partials:
            Shares of the partial derivatives of the loss with respect to the input of the layer. If ``partials_prev``
            is a list of tensors, ``new_partials`` is a list of tensors with the same length as ``partials_prev``.
        """

        batch = type(partials_prev) == list

        if batch:
            partials_prev = [_partials_prev.reshape(-1, 1) for _partials_prev in partials_prev]

        else:
            partials_prev = partials_prev.reshape(-1, 1)

        W_update = await self.sec_comp.sec_mul(self._input, partials_prev, batch=batch, byzantine=self.byzantine)

        if batch:
            W_update = sum(W_update)
            b_update = sum([_partials_prev.reshape(-1) for _partials_prev in partials_prev])
        else:
            b_update = partials_prev.reshape(-1)

        if self.calc_partials:
            if batch:
                new_partials = await self.sec_comp.sec_mat_mul([self.W.T for _ in range(len(partials_prev))], partials_prev, batch, self.byzantine)
            else:
                new_partials = await self.sec_comp.sec_mat_mul(self.W.T, partials_prev, batch, self.byzantine)
        else:
            new_partials = None

        if batch:
            self.W -= (self._lr * W_update) / len(partials_prev)
            self.b -= (self._lr * b_update) / len(partials_prev)
        else:
            self.W -= self._lr * W_update
            self.b -= self._lr * b_update

        return new_partials

    def clone(self, group_num):
        """
        Create a clone of this layer for the given group number ``group_num``.

        Parameters
        ----------
        group_num: int
            The group number to create the copy for.

        Returns
        -------
        clone:
            A clone of this layer for the given group number ``group_num``. The returned layer has the same
            attributes (batch size, learning rate, ...) as this layer, except for the group number. Secret-shared
            parameters (weights, biases) are not copied as they will be overwritten by other secret shares.
        """

        return FullyConnected(self._batch_size, self._lr, None, None, self.calc_partials, self._empty_like, self._diag, self._zeros_like, self._auxiliary_nodes, group_num, self.byzantine)
