import torch
from torch import Tensor
from torch.utils.data import TensorDataset


def generate_dataset(num_samples: int = 5000) -> TensorDataset:
    # shape: (num_samples, 2)
    initial_points: Tensor = torch.rand(num_samples, 2)

    # shape: (num_samples, 2)
    final_points: Tensor = torch.rand(num_samples, 2)

    # shape: (num_samples, 2)
    displacement_vectors: Tensor = final_points - initial_points

    # shape: (num_samples, 4)
    model_inputs: Tensor = torch.cat((initial_points, final_points), dim=1)

    dataset: TensorDataset = TensorDataset(model_inputs, displacement_vectors)

    return dataset
