import torch
from torch import Tensor
from torch.utils.data import TensorDataset


def generate_dataset(
    number_of_samples: int,
    box_size: float,
) -> TensorDataset:

    half_box = box_size / 2

    initial_points: Tensor = (
        torch.rand(number_of_samples, 2) * box_size
    ) - half_box  # shape: (number_of_samples, 2)

    final_points: Tensor = (
        torch.rand(number_of_samples, 2) * box_size
    ) - half_box  # shape: (number_of_samples, 2)

    # shape: (number_of_samples, 2)
    displacement_vectors: Tensor = final_points - initial_points

    # shape: (number_of_samples, 4)
    model_inputs: Tensor = torch.cat((initial_points, final_points), dim=1)

    dataset: TensorDataset = TensorDataset(model_inputs, displacement_vectors)

    return dataset
