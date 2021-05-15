import torch
from torch import nn
from .models import fullyconnected, simpleCNN, skipCNN
from .error import compute_nb_errors


def train_model(
    model_type: str,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_classes: torch.Tensor,
    test_input: torch.Tensor,
    test_target: torch.Tensor,
    test_classes: torch.Tensor,
    mini_batch_size: int,
    nb_epochs: int,
    nb_iterations: int,
):
    """
    Trains the specified model using train_data and tests the model using the
    compute_nb_errors function and returns errors and losses.
    
    Parameters
    ----------
    model_type: str
        String indicating the network to use. Choose among "fullyconnected", "simpleCNN", "skipCNN".
    train_input: torch.Tensor
        The training set of 2 channel tensors with each channel containig a 14 x 14 images.
    train_target: torch.Tensor
        The training set of 2 dimensional tensors indicating the class of the two digits in the images
    train_classes: torch.Tensor
        The training set of 1 dimensional tensors indicating the class to predict for each image pair
    test_input: torch.Tensor
        The test set of 2 channel tensors with each channel containig a 14 x 14 images.
    test_target: torch.Tensor
        The test set of 2 dimensional tensors indicating the class of the two digits in the images
    test_classes: torch.Tensor
        The test set of 1 dimensional tensors indicating the class to predict for each image pair
    mini_batch_size: int
        The size of the batches to be used for training
    nb_epochs: int
        The number of epochs(times) to train the network
    nb_iterations: int
        The number of time to train the network from scratch to estimate errors.
    """

    # list to store output of the function
    train_outputs = []

    # search for a GPU for training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model parameters
    model = None
    eta = 0.1
    alpha = 0.2
    # net_type holds the 4 possible scenarios of the network in a list of tuple.
    # The first element of the tuple indicates if the network should use weight sharing
    # and the second element indicates if the network should use auxiliary loss for training.
    net_types = [(False, False), (True, False), (False, True), (True, True)]

    # Iterate for four possible cases
    for (use_weight_sharing, use_auxiliary_loss) in net_types:

        # A dictionary to store outputs for each of the possible scenario.
        freturns = {
            "weight_sharing": use_weight_sharing,
            "auxiliary_loss": use_auxiliary_loss,
            "loss": [],
            "loss1": [],
            "loss2": [],
            "loss3": [],
            "error": [],
            "error1": [],
            "error2": [],
            "mean_error": [],
            "std_error": [],
        }

        # print("\nTraining", "-" * 60)
        print("\n\t Using Weight Sharing:", use_weight_sharing)
        print("\t Using Auxiliary Loss:", use_auxiliary_loss)
        print("\n")

        all_errors = torch.empty(nb_iterations)

        # Iterate for performance estimation
        for k in range(nb_iterations):
            print("\t\t Iteration number {}".format(k + 1))
            # Define the training model
            if model_type == "fullyconnected":
                model = fullyconnected(use_weight_sharing, use_auxiliary_loss)
            elif model_type == "simpleCNN":
                model = simpleCNN(use_weight_sharing, use_auxiliary_loss)
            elif model_type == "skipCNN":
                model = skipCNN(use_weight_sharing, use_auxiliary_loss)

            model.to(device)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=eta)
            criterion = nn.CrossEntropyLoss()

            loss_list = []
            loss1_list = []
            loss2_list = []
            loss3_list = []
            error_list = []
            error1_list = []
            error2_list = []

            # Iterate over several epochs
            for _ in range(nb_epochs):
                # Iterate over mini-batches
                for b in range(0, train_input.size(0), mini_batch_size):

                    if use_auxiliary_loss:
                        # get the outputs
                        output, output2, output3 = model(train_input.narrow(0, b, mini_batch_size).to(device))
                        # compute the individual losses
                        loss1 = criterion(output, train_target.narrow(0, b, mini_batch_size).to(device))
                        loss2 = criterion(output2, train_classes[:, 0].narrow(0, b, mini_batch_size).to(device))
                        loss3 = criterion(output3, train_classes[:, 1].narrow(0, b, mini_batch_size).to(device))
                        # calculate the combined loss
                        loss = (1 - alpha) * loss1 + alpha * (loss2 + loss3)  # Take weighted average

                    else:
                        # get the outputs
                        output = model(train_input.narrow(0, b, mini_batch_size).to(device))
                        # calculate the loss
                        loss = criterion(output, train_target.narrow(0, b, mini_batch_size).to(device))

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # append the losses to return as output
                loss_list.append(loss.item())
                if use_auxiliary_loss:
                    loss1_list.append(loss1.item())
                    loss2_list.append(loss2.item())
                    loss3_list.append(loss3.item())

            freturns["loss"].append(loss_list)
            if use_auxiliary_loss:
                freturns["loss1"].append(loss1_list)
                freturns["loss2"].append(loss2_list)
                freturns["loss3"].append(loss3_list)

            # Compute number of errors
            if use_auxiliary_loss:
                (nb_errors, nb_errors2, nb_errors3) = compute_nb_errors(model, test_input, test_target, test_classes)
                error = (100 * nb_errors) / test_input.size(0)
                error2 = (100 * nb_errors2) / test_classes.size(0)
                error3 = (100 * nb_errors3) / test_classes.size(0)
                print("\t\t\t test error Net {:0.2f}% {:d}/{:d}".format(error, nb_errors, test_input.size(0)))
                print("\t\t\t test error X1 {:0.2f}% {:d}/{:d}".format(error2, nb_errors2, test_classes.size(0)))
                print("\t\t\t test error X2 {:0.2f}% {:d}/{:d}".format(error3, nb_errors3, test_classes.size(0)))

                freturns["error"].append(error.item())
                freturns["error1"].append(error2.item())
                freturns["error2"].append(error3.item())

            else:
                nb_errors = compute_nb_errors(model, test_input, test_target, test_classes)
                error = (100 * nb_errors) / test_input.size(0)
                print("\t\t\t test error Net {:0.2f}% {:d}/{:d}".format(error, nb_errors, test_input.size(0)))
                freturns["error"].append(error.item())
            #
            all_errors[k] = error

        #
        error_mean = all_errors.std().item()
        error_std = all_errors.mean().item()
        print("\n \t\t Standard Deviation: {:0.2f}%".format(error_mean))
        print("\t\t Mean Error: {:0.2f}%".format(error_std))
        freturns["mean_error"].append(error_mean)
        freturns["std_error"].append(error_std)

        train_outputs.append(freturns)

    return train_outputs
