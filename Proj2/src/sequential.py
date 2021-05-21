import torch
from .types import Module


class Sequential(Module):
    """
    Module to hold the layers and build the Network
    """

    def __init__(self, *args):
        """
        Parameters:
            *args (list[Module]): The list of modules to be constructed in the
            network.
        """
        super().__init__()
        self.name = "Sequential"

        # A list to hold all layers of the network
        self.modules = [module for module in args]

    def __str__(self):
        return self.name

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Feed Forward prediction of the network. The input is propagated through
        all the layers and the output of the final layer is returned.

        Parameters:
            input_ (Tensor): The input sample
        """
        self.inp = input_
        # incase of no layers, the input itself is returned as output
        output = input_

        for module in self.modules:
            output = module.forward(output)

        self.output = output

        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Backward propagation of the network. The error is propagated through
        all the layers iteratively.

        Parameters:
            input_ (Tensor): The input sample
        """
        # The error is propagated in the reverse (backward) direction
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)

    def param(self) -> list:
        """
        List of parameters of all modules

        Returns:
            params (list): List of tuple of weight and bias of each layer in
            the network
        """

        params = []
        for module in self.modules:
            params.extend(module.param())

        return params

    def zero_grad(self):
        """
        Sets the gradient to zero of all modules
        """

        for weight, gradient in self.param():
            if (weight is None) or (gradient is None):
                # incase of activation function modules, skip them
                continue
            else:
                gradient.zero_()
