# Final submission: main executable test.py to call without arguments

from src.dlc_practical_prologue import generate_pair_sets
from src.train import train_model


def main():

    num_of_train_test_pairs = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
        num_of_train_test_pairs
    )

    nb_iterations = 10
    nb_epochs = 50
    mini_batch_size = 100
    model_types = ["fullyconnected", "simpleCNN", "skipCNN"]

    # Use normalized inputs
    inp_mean = train_input.mean()
    inp_std = train_input.std()
    norm_train_input = train_input.sub_(inp_mean).div_(inp_std)
    norm_test_input = test_input.sub_(inp_mean).div_(inp_std)

    # Iterate for all model architectures
    for model_type in model_types:
        print("\n\n")
        print("=" * 30)
        print("Using model:", model_type)
        print("=" * 30)

        _ = train_model(
            model_type,
            norm_train_input,
            train_target,
            train_classes,
            norm_test_input,
            test_target,
            test_classes,
            mini_batch_size,
            nb_epochs,
            nb_iterations,
        )


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    main()

