import torch

# Compute number of errors
def compute_nb_errors(model, test_input, test_target, test_classes):
    """
    Computes and returns the number of prediction mistakes
    
    Parameters
    ----------
    model: torch.Model
        The model to use to compute the error
        test_input: torch.Tensor
        The test set of 2 channel tensors with each channel containig a 14 x 14 images.
    test_target: torch.Tensor
        The test set of 2 dimensional tensors indicating the class of the two digits in the images
    test_classes: torch.Tensor
        The test set of 1 dimensional tensors indicating the class to predict for each image pair
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    nb_errors = 0

    with torch.no_grad():

        # get the ouput
        if model.use_auxiliary_loss:
            o1, o2, o3 = model(test_input.to(device))
        else:
            o1 = model(test_input.to(device))

    # Count number of errors
    if model.use_auxiliary_loss:

        # compute the number of errors in predicting the class of lesser or equal
        output1 = torch.argmax(o1, dim=1)
        expected1 = test_target.to(device)
        nb_errors = torch.count_nonzero((expected1 != output1))

        # compute the number of errors in predicting the class first image
        output2 = torch.argmax(o2, dim=1)
        expected2 = test_classes[:, 0].to(device)
        nb_errors2 = torch.count_nonzero((expected2 != output2))

        # compute the number of errors in predicting the class second image
        output3 = torch.argmax(o3, dim=1)
        expected3 = test_classes[:, 1].to(device)
        nb_errors3 = torch.count_nonzero((expected3 != output3))

        return (nb_errors, nb_errors2, nb_errors3)

    else:
        # compute the number of errors in predicting the class of lesser or equal
        output1 = torch.argmax(o1, dim=1)
        expected1 = test_target.to(device)
        nb_errors = torch.count_nonzero((expected1 != output1))

        return nb_errors
