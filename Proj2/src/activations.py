import torch
from .types import Module

# TanH activation


class TanH(Module):

    """Module to apply the Hyperbolic Tangent function"""

    def __init__(self):

        super().__init__()
        self.name = "TanH"

    def __str__(self):
        return self.name

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Returns the tensor after applying tanh to the input

        Parameters:
            input (Tensor): The tensor on which the tanh should be applied

        Returns:
            Tensor: The tensor obtained after applying the tanh on the input
        """

        self.out_ = input_.tanh()

        return self.out_

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Returns the gradient of loss with respect to the input on applying tanh

        Parameters:
            gradientwrtoutput (Tensor): gradient with respect to the output

        Returns:
            Tensor: The gradient of the loss with respect to the input
        """

        return gradwrtoutput * (1 - self.out_.pow(2))

    def param(self):

        return []


class ReLU(Module):

    """Module to apply the Rectified Linear function"""

    def __init__(self):

        super().__init__()
        self.name = "ReLU"

    def __str__(self):
        return self.name

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Returns the tensor after applying ReLU to the input.

        Parameters:
            input (Tensor): The tensor on which the ReLU should be applied

        Returns:
            Tensor: The tensor obtained after applying the ReLU on the input
        """

        self.out = input_.clamp(min=0.0)

        return self.out

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Returns the gradient of loss with respect to the input on applying ReLU

        Parameters:
            gradientwrtoutput (Tensor): gradient with respect to the output

        Returns:
            Tensor: The gradient of the loss with respect to the input
        """

        self.out[self.out <= 0] = 0
        self.out[self.out > 0] = 1

        return gradwrtoutput * self.out

    def param(self):

        return []


class Sigmoid(Module):

    """Module to apply the Sigmoid function"""

    def __init__(self):

        super().__init__()
        self.name = "Sigmoid"

    def __str__(self):
        return self.name

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Returns the tensor after applying sigmoid to the input.

        Parameters:
            input (Tensor): The tensor on which the sigmoid should be applied

        Returns:
            Tensor: The tensor obtained after applying the sigmoid on the input
        """

        self.out_ = input_.sigmoid()

        return self.out_

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Returns the gradient of loss with respect to the input on applying sigmoid

        Parameters:
            gradientwrtoutput (Tensor): gradient with respect to the output

        Returns:
            Tensor: The gradient of the loss with respect to the input
        """

        return gradwrtoutput * (self.out_ - self.out_ ** 2)

    def param(self):

        return []
