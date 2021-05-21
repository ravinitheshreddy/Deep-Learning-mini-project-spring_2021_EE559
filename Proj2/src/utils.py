import torch
import math
from .sequential import Sequential
from .types import Module
from .optimizer import SGD
from typing import Union


def generate_data(num_points: int) -> Union[torch.Tensor, torch.Tensor]:
    """
    Function to generate the dataset of 1,000 points sampled uniformly
    in [0, 1]^2, each with a label 0 if outside the disk centered at (0.5; 0.5)
    of radius 1/sqrt(2*pi), and 1 inside.

    Parameters:
        num_points (int): The number of points to be generated

    Returns:
        Tensor : A two dimensional input data with points sampled between [0,1]
        Tensor : A two dimensional output data that contains labels
        corresponding to the input data generated above as one hot encoded variable
    """

    input_ = torch.Tensor(num_points, 2).uniform_(0, 1)

    labels = input_.sub(0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()

    labels_onehot = torch.empty(num_points, 2).fill_(0)
    labels_onehot[:, 0] = labels == 0
    labels_onehot[:, 1] = labels == 1

    return input_, labels_onehot


def train_model(
    model: Sequential,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    loss_criteria: Module,
    learning_rate: float,
    mini_batch_size: int,
    nb_epochs: int,
) -> list:
    """
    Function to train a model and return the epoch wise loss as a list.

    Parameters:
        model (Sequential): The neural network model
        train_input (Tensor): The input data samples
        train_target (Tensor): The target of data samples
        loss_criteria (Module): The loss function to use to train the model
        learning_rate (float): The learning rate to be update the weights of
        the model
        mini_batch_size (int): The batch size to train the model
        nb_epochs (int): The number of eppochs to train the network

    Returns:
        losses : A list of loss collected after each epoch of training
    """

    optimizer = SGD(model.param(), lr=learning_rate)
    losses = []
    for epoch_number in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss_ = loss_criteria.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            model.backward(loss_criteria.backward())
            optimizer.step()
        if epoch_number % 100 == 0:
            print("\t\tEpoch {} Training loss {}".format(epoch_number, loss_.item()))
        losses.append(loss_.item())
    return losses


def compute_nb_errors(model: Sequential, input_: torch.Tensor, target: torch.Tensor) -> int:
    """
    Computes and returns the number of misclassifications done by the model

    Parameters:
        model (Sequential): The neural network model
        input_ (Tensor): The input data samples
        target (Tensor): The target of data samples

    Returns:
        nb_data_errors (int): The number of misclassifications.

    """

    nb_data_errors = 0

    output = model.forward(input_)

    _, predicted = torch.max(output.data, 1)

    _, actual = torch.max(target.data, 1)

    for k in range(input_.size()[0]):
        if actual.data[k] != predicted[k]:
            nb_data_errors = nb_data_errors + 1
    return nb_data_errors
