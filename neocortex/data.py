import torch
from torch import Tensor

Dataset = tuple[Tensor, Tensor, Tensor, Tensor]


def generate_dataset(number_of_samples: int) -> Dataset:
    initial_coords: Tensor = torch.rand(number_of_samples, 2)
    final_coords: Tensor = torch.rand(number_of_samples, 2)

    x: Tensor = torch.cat([initial_coords, final_coords], dim=1)
    y: Tensor = final_coords - initial_coords

    split_index: int = int(number_of_samples * 0.8)
    x_train: Tensor = x[:split_index]
    y_train: Tensor = y[:split_index]
    x_test: Tensor = x[split_index:]
    y_test: Tensor = y[split_index:]

    return x_train, y_train, x_test, y_test
