from neural_network.distributed.layers import Layer


class ExactReLU(Layer):
    """
    A layer that applies the exact Rectified Linear Unit (ReLU) activation function.
    """

    _auxiliary_nodes: []
    _group_num: int

    def __init__(self, batch_size, lr, clone, zeros_like, auxiliary_nodes, group_num, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate to be used for training the layer.
        clone: callable
            A function that clones the input array to prevent modifying the original data.
        zeros_like: callable
            A function that returns a tensor of zeros with the same shape and data type as the input tensor.
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine)
        self._clone = clone
        self._zeros_like = zeros_like
        self._input = None
        self._auxiliary_nodes = auxiliary_nodes
        self._group_num = group_num
        self._input_signs = None

    async def forward(self, x):
        """
        Computes the forward pass of the ExactReLU layer.

        Parameters
        ----------
        x:
            A share of the input tensor to be processed by the layer. If a list is provided, it is treated as a batch.

        Returns
        -------
        output:
            The output data after applying the ExactReLU activation function to the input data. If ``x`` is a list of
            tensors (batched input), ``output`` is a list of tensors with the same length as ``x``.
        """

        batch = type(x) == list

        if batch:  # batch inference
            self._input = []
            y = []
            for i in range(len(x)):
                x[i] = x[i].reshape(-1)
                self._input.append(self._clone(x[i]))
                y.append(self._zeros_like(x[i]))
        else:
            x = x.reshape(-1)
            self._input = self._clone(x)
            y = self._zeros_like(x)

        self._input_signs = await self.sec_comp.sec_cmp(x, y, batch, self.byzantine)

        if batch:
            for i in range(len(x)):
                x[i][self._input_signs[i] < 0] = 0
        else:
            x[self._input_signs < 0] = 0

        return x

    async def backward(self, partials_prev):
        """
        Computes the backward pass of the ExactReLU layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output of this layer in the previous iteration
            of backpropagation. If a list is provided, it is treated as a batch.

        Returns
        -------
        new_partials:
            Shares of the partial derivatives of the loss with respect to the input data, computed by backpropagating
            the partial derivatives of the loss with respect to the output of this layer.
        """

        if type(partials_prev) == list:
            for i, (_in_signs, _partials_prev) in enumerate(zip(self._input_signs, partials_prev)):
                partials_prev[i] = _partials_prev.reshape(-1)
                partials_prev[i][_in_signs < 0] = 0

        else:
            partials_prev = partials_prev.reshape(-1)
            partials_prev[self._input_signs < 0] = 0

        return partials_prev

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
            attributes (batch size, learning rate, ...) as this layer, except for the group number.
        """

        return ExactReLU(self._batch_size, self._lr, self._clone, self._zeros_like, self._auxiliary_nodes, group_num, self.byzantine)
