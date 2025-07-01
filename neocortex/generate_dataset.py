import torch
from torch import Tensor
from torch.utils.data import TensorDataset


def generate_dataset(
    num_samples: int, max_length: float, vehicle_velocity: float
) -> TensorDataset:
    lengths: Tensor = torch.rand(num_samples) * max_length
    times: Tensor = lengths / vehicle_velocity

    dataset: TensorDataset = TensorDataset(lengths, times)
    return dataset
