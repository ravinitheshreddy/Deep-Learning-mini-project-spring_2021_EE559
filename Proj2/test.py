from src.utils import generate_data, train_model, compute_nb_errors
from src.sequential import Sequential
from src.linear import Linear
from src.activations import TanH, ReLU, Sigmoid
from src.loss import MSELoss


def main():

    learning_rate = 0.01
    nb_epochs = 1000
    mini_batch_size = 100

    # generate the training and test samples
    num_of_train_pairs = 1000
    num_of_test_pairs = 1000
    train_input, train_target = generate_data(num_of_train_pairs)
    test_input, test_target = generate_data(num_of_test_pairs)

    # normalising the train and test data
    mu, std = train_input.mean(), train_input.std()
    norm_train_input = train_input.sub_(mu).div_(std)
    norm_test_input = test_input.sub_(mu).div_(std)

    # defining the loss
    mseloss = MSELoss()

    # defining models

    Model_1 = Sequential(
        Linear(2, 25),
        TanH(),
        Linear(25, 25),
        TanH(),
        Linear(25, 25),
        TanH(),
        Linear(25, 25),
        TanH(),
        Linear(25, 2),
        Sigmoid(),
    )

    Model_2 = Sequential(
        Linear(2, 25, weightsinit="xavier"),
        TanH(),
        Linear(25, 25, weightsinit="xavier"),
        TanH(),
        Linear(25, 25, weightsinit="xavier"),
        TanH(),
        Linear(25, 25, weightsinit="xavier"),
        TanH(),
        Linear(25, 2, weightsinit="xavier"),
        Sigmoid(),
    )

    Model_3 = Sequential(
        Linear(2, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 2),
        Sigmoid(),
    )

    Model_4 = Sequential(
        Linear(2, 25, weightsinit="kaiming"),
        ReLU(),
        Linear(25, 25, weightsinit="kaiming"),
        ReLU(),
        Linear(25, 25, weightsinit="kaiming"),
        ReLU(),
        Linear(25, 25, weightsinit="kaiming"),
        ReLU(),
        Linear(25, 2, weightsinit="kaiming"),
        Sigmoid(),
    )

    models = {
        1: {"model": Model_1, "activation": "TanH", "loss": mseloss, "weightint": "Uniform"},
        2: {"model": Model_2, "activation": "TanH", "loss": mseloss, "weightint": "Xavier"},
        3: {"model": Model_3, "activation": "ReLU", "loss": mseloss, "weightint": "Uniform"},
        4: {"model": Model_4, "activation": "ReLU", "loss": mseloss, "weightint": "kaiming"},
    }

    for id_num, M in models.items():
        print("=" * 80)
        print(
            "\tModel {} with {} activation, with {} weight initialisation ".format(
                id_num, M["activation"], M["weightint"], str(M["loss"])
            )
        )
        print("=" * 80)

        model = M["model"]

        _ = train_model(model, norm_train_input, train_target, M["loss"], learning_rate, mini_batch_size, nb_epochs)

        nb_train_errors = compute_nb_errors(model, norm_train_input, train_target)
        nb_test_errors = compute_nb_errors(model, norm_test_input, test_target)

        train_accuracy = (100 * (norm_train_input.size(0) - nb_train_errors)) / norm_train_input.size(0)

        test_accuracy = (100 * (norm_test_input.size(0) - nb_test_errors)) / norm_test_input.size(0)

        train_error = 100 - train_accuracy
        test_error = 100 - test_accuracy

        print("\n\t Train Error {:0.2f}% {:d}/{:d}".format(train_error, nb_train_errors, train_input.size(0)))

        print("\t Test Error {:0.2f}% {:d}/{:d}".format(test_error, nb_test_errors, test_input.size(0)))

        print("\n")
    return None


if __name__ == "__main__":
	import torch
	torch.manual_seed(420)
	main()
