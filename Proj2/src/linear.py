import torch
import math
from .types import Module


class Linear(Module):

    """
    Module that implements a linear matrix operation layer
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, weightsinit: str = "uniform"):
        """
        Initialises the layer by creating empty weight and bias tensors
        and initialising them using uniform distribution.

        Parameters:
            in_features (int): The size of each input sample
            out_features (int): The size of each output sample
            bias – If set to False, the layer will not learn an additive bias. Default: True
            weightsinit (str): The type of weight initialization to use
        """

        super().__init__()
        self.name = "Linear"

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weightsinit = weightsinit

        self.w = torch.empty(self.in_features, self.out_features)
        self.gradw = torch.empty(self.in_features, self.out_features)

        if self.bias:
            self.b = torch.empty(self.out_features)
            self.gradb = torch.empty(self.out_features)
        else:
            self.b = None
            self.gradb = None

        self.initWeights()

    def __str__(self):
        return self.name

    def initWeights(self):
        """
        Initialises the weight and bias parameters of the layer depending on
        the weightinit parameter. Irrespective of the weightinit parameter
        the bias are always zero initialised 

        If "weightsinit" is
            1. uniform (by default), the weights are initiliased using uniform distribution.
            2. xavier, the weights are initiliased using Xavier Initialisation.
            3. kaiming, the weights are initiliased using Kaiming Initialisation when using ReLU layer
        """

        if self.weightsinit == "uniform":
            k = math.sqrt(1.0 / self.in_features)
            self.w.uniform_(-k, k)

        elif self.weightsinit == "xavier":
            self.w.normal_(0, math.sqrt(2 / (self.in_features + self.out_features)))

        elif self.weightsinit == "kaiming":
            self.w.normal_(0, math.sqrt(2 / (self.in_features)))

        self.gradw.fill_(0)

        if self.b is not None:
            self.b.fill_(0)
            self.gradb.fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the layer by multiplying the input with weights and adding the bias
        """

        self.inp = input_

        if self.b is None:
            self.output = self.inp.matmul(self.w)
        else:
            self.output = self.inp.matmul(self.w).add(self.b)

        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient for the weights and biases.
        """

        gradw = self.inp.t().matmul(gradwrtoutput)
        self.gradw.add_(gradw)

        if self.b is not None:
            gradb = gradwrtoutput.sum(0)
            self.gradb.add_(gradb)
        gradient = gradwrtoutput.matmul(self.w.t())

        return gradient

    def param(self) -> list:
        """
        Returns the parameters of the layer
        """

        params = [(self.w, self.gradw)]
        if self.b is not None:
            params.append((self.b, self.gradb))

        return params

    def zero_grad(self):
        """
        Sets the gradient to zero
        """

        self.gradw.zero_()

        if self.b is not None:
            self.gradb.zero_()
