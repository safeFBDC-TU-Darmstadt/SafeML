from neural_network.centralized.layers import Layer


class AvgPooling(Layer):
    """
    A layer that applies average pooling to the input.
    """

    def __init__(self, batch_size, lr, k, in_dim, output_dim, empty_like, empty) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate for the layer.
        k:
            The size of the square region to apply the average. Also used as the stride when sliding over the input.
        in_dim:
            The dimensions of the input.
        output_dim:
            The dimensions of the output.
        empty_like: callable
            A function that returns an empty tensor like the input.
        empty: callable
            A function that returns an empty tensor.
        """

        super().__init__(batch_size, lr)
        self.in_dim = in_dim
        self.output_dim = output_dim
        self._output = None
        self.k = k
        self._empty_like = empty_like
        self._empty = empty
        self._input = None

    def forward(self, X):
        """
        Applies the average pooling operation to the input ``X`` using the configured kernel size ``k`` and stride ``k``.
        Returns the result of the convolution plus the bias.

        Parameters
        ----------
        X:
            The 3D input tensor over which to apply the pooling operation. The input will not be padded.

        Returns
        -------
        conv_result:
            The result of the pooling operation as a 3D tensor obtained by calculation the average of every 2D :math:`k \times k`
            region of the input data.
        """

        batch = type(X) == list

        if batch:
            X = [_X.reshape(self.in_dim) for _X in X]
            self._output = [self._empty(self.output_dim) for _ in X]
        else:
            X = X.reshape(self.in_dim)
            self._output = self._empty(self.output_dim)

        self._input = X

        for m in range(int(self.in_dim[1] / self.k)):
            for n in range(int(self.in_dim[2] / self.k)):
                if batch:
                    for i, _X in enumerate(X):
                        s = _X[:, m * self.k:m * self.k + self.k, n * self.k:n * self.k + self.k].sum([1, 2])
                        self._output[i][:, m, n] = s / (self.k ** 2)
                else:
                    s = X[:, m * self.k:m * self.k + self.k, n * self.k:n * self.k + self.k].sum([1, 2])
                    self._output[:, m, n] = s / (self.k ** 2)

        return self._output

    def backward(self, partials_prev):
        """
        Calculates and returns the partial derivatives of the loss with respect to the input of the layer.

        Parameters
        ----------
        partials_prev:
            The partial derivatives of the loss with respect to the output.

        Returns
        -------
        new_partials:
            The partial derivatives of the loss with respect to the input.
        """

        batch = type(partials_prev) == list

        if batch:
            new_partials = [self._empty(self.in_dim) for _ in partials_prev]
            partials_prev = [_partials_prev.reshape(self.output_dim) for _partials_prev in partials_prev]
        else:
            new_partials = self._empty_like(self._input)
            partials_prev = partials_prev.reshape(self._output.shape)

        for i in range(self.in_dim[0]):
            for m in range(int(self.in_dim[1] / self.k)):
                for n in range(int(self.in_dim[2] / self.k)):
                    if batch:
                        for j, _partials_prev in enumerate(partials_prev):
                            new_partials[j][i, m * self.k:(m + 1) * self.k, n * self.k:(n + 1) * self.k] = _partials_prev[i][m][n] / (self.k ** 2)
                    else:
                        new_partials[i, m*self.k:(m+1)*self.k, n*self.k:(n+1)*self.k] = partials_prev[i][m][n] / (self.k ** 2)

        return new_partials
