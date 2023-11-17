class Layer:
    """
    Abstract base class for neural network layers.
    """

    _batch_size: int
    _lr: float
    iteration = 0
    worker_node = None  # set by the worker node after the instantiation of this object

    def __init__(self, batch_size, lr, byzantine):
        """
        Initializes the Layer object.

        Parameters
        ----------
        batch_size : int
            The size of the batches that will be passed through the network.
        lr : float
            The learning rate to use during training.
        """

        self._batch_size = batch_size
        self._lr = lr
        self.sec_comp = None
        self.byzantine = byzantine

    async def forward(self, layer_input):
        """
        Performs a forward pass of the layer.

        Parameters
        ----------
        layer_input:
            A share of the input to the layer, with shape (batch_size, ...).

        Returns
        -------
        layer_output:
            A share of the output of the layer, with shape (batch_size, ...).
        """

        pass

    async def backward(self, partials_prev):
        """
        Performs a backward pass of the layer.

        Parameters
        ----------
        partials_prev:
            A share of the partial derivatives of the loss with respect to the output of the layer in the previous layer, with shape ``(batch_size, ...)``.

        Returns
        -------
        partials:
            A share of the partial derivatives of the loss with respect to the output of the layer, with shape ``(batch_size, ...)``.
        """

        pass

    def update_parameters(self):
        pass

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
            parameters (e.g. weights, filters, biases) are not copied as they will be overwritten by other secret shares.
        """

        pass
