from torch.utils.data import TensorDataset


def split_dataset(
    dataset: TensorDataset, train_ratio: float
) -> tuple[TensorDataset, TensorDataset]:
    num_samples: int = len(dataset)
    num_train_samples: int = int(num_samples * train_ratio)

    inputs, outputs = dataset.tensors

    X_train, y_train = (  # pylint: disable=invalid-name
        inputs[:num_train_samples],
        outputs[:num_train_samples],
    )

    X_test, y_test = (  # pylint: disable=invalid-name
        inputs[num_train_samples:],
        outputs[num_train_samples:],
    )

    training_dataset: TensorDataset = TensorDataset(X_train, y_train)
    testing_dataset: TensorDataset = TensorDataset(X_test, y_test)

    return training_dataset, testing_dataset
