import torch
from .types import Module

# Mean Squared Error Loss


class MSELoss(Module):

    """Module to calculate the Mean Squared Error."""

    def __init__(self):
        super().__init__()
        self.name = "MSE Loss"

    def __str__(self):
        return self.name

    def forward(self, output_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the MSE Loss between output_ and target

        Parameters:
            output_ (Tensor): First tensor to calculate the MSE.
            target (Tensor): Second tensor to calculate the MSE.

        Returns:
            Tensor: The Mean Squared Loss between input_ and target
        """

        self.error = output_ - target
        self.loss = self.error.pow(2).mean()

        return self.loss

    def backward(self) -> torch.Tensor:
        """
        gradient of loss

        Returns:
            Tensor: The gradient of Mean Squared Loss between input_ and target
        """

        return (2 * self.error) / self.error.size()[0]

