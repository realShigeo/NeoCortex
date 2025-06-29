from torch import Tensor
from torch.utils.data import TensorDataset


def normalize_dataset(
    dataset: TensorDataset, box_size: float
) -> TensorDataset:
    model_inputs, displacement_vectors = dataset.tensors

    norm_model_inputs: Tensor = model_inputs / box_size
    norm_displacement_vectors: Tensor = displacement_vectors / box_size

    norm_dataset: TensorDataset = TensorDataset(
        norm_model_inputs, norm_displacement_vectors
    )

    return norm_dataset
