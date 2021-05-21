from .types import Optimizer


class SGD(Optimizer):

    """
    Module to perform Stochastic Gradient Descent
    """

    def __init__(self, params: list, lr=0.01):
        """
        Parameters:
            params (list): List of the parameters of the network
            lr (float): The learning rate of the network
        """

        super().__init__()
        self.name = "SGD"

        self.params = params
        self.lr = lr

        if self.lr <= 0.0:
            raise ValueError("Learning rate {} should be greater than zero".format(self.lr))

    def __str__(self):
        return self.name

    def step(self):
        """
        Function to perform the single optimization step
        """

        for weight, gradient in self.params:
            if (weight is None) or (gradient is None):
                # incase of activation function modules, skip them
                continue
            else:
                weight.add_(-self.lr * gradient)

