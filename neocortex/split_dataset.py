from torch.utils.data import TensorDataset


def split_dataset(
    dataset: TensorDataset, train_ratio: float
) -> tuple[TensorDataset, TensorDataset]:

    num_samples: int = len(dataset)
    num_training_samples: int = int(train_ratio * num_samples)

    model_inputs, displacement_vectors = (
        dataset.tensors
    )  # shape: (num_samples, 4), (num_samples, 2)

    X_train, y_train = (  # pylint: disable=invalid-name
        model_inputs[:num_training_samples],
        displacement_vectors[:num_training_samples],
    )  # shape: (num_training_samples, 4), (num_training_samples, 2)

    X_test, y_test = (  # pylint: disable=invalid-name
        model_inputs[num_training_samples:],
        displacement_vectors[num_training_samples:],
    )  # shape: (num_test_samples, 4), (num_test_samples, 2)

    train_dataset: TensorDataset = TensorDataset(X_train, y_train)
    test_dataset: TensorDataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset
