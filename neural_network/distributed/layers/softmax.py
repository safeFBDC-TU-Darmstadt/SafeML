from neural_network.distributed.layers import Layer


class Softmax(Layer):
    """
    A layer that applies the softmax activation function (https://en.wikipedia.org/wiki/Softmax_function).
    """

    _group_num: int

    def __init__(self, batch_size, lr, output_mask, group_num, byzantine) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate to be used for training the layer.
        output_mask:
            A additive secret share of an output mask. Must have the same dimensions as the output of the layer before this.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine)
        self.output_mask = output_mask
        self._group_num = group_num
        self._output = None

    async def forward(self, x):
        """
        Computes the forward pass of the Softmax layer (https://en.wikipedia.org/wiki/Softmax_function).

        Indirectly uses the model owner's support function by calling the mediator nodes.

        Parameters
        ----------
        x:
            A share of the input tensor to be processed by the layer. If a list is provided, it is treated as a batch.

        Returns
        -------
        output:
            A share of the output data after applying the softmax activation function
            to the input data. If ``x`` is a list of tensors (batched input), ``output`` is a list of tensors with the same length
            as ``x``. To obtain the softmax outputs, the masked input tensor is sent to every mediator node. The mediator nodes
            will reconstruct the masked secret tensor and forward it to the model owner. The model owner will unmask the
            secret tensor, perform the softmax activation and respond with shares of this output. The share corresponding
            to this group number (``group_num``) is returned to this node.
        """

        batch = type(x) == list

        self._output = await self.sec_comp.sec_softmax(x, self.output_mask, batch, self.byzantine)

        return self._output

    async def backward(self, label_share):
        """
        Computes the backward pass of the Softmax layer (https://en.wikipedia.org/wiki/Softmax_function).

        Parameters
        ----------
        label_share:
            A share of the label or target output of the layer. If a list is provided, batch training is performed.

        Returns
        -------
        new_partials:
            Shares of the partial derivatives of the loss with respect to the input data.
        """

        batch = type(label_share) == list

        if batch:
            result = [_out - _label_share for _out, _label_share in zip(self._output, label_share)]
        else:
            result = self._output - label_share

        return result

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

        return Softmax(self._batch_size, self._lr, None, group_num, self.byzantine)
