from torch import Tensor
from torch.utils.data import TensorDataset


def normalize_dataset(
    dataset: TensorDataset, max_length: float
) -> TensorDataset:
    inputs, outputs = dataset.tensors

    norm_inputs: Tensor = inputs / max_length

    norm_dataset: TensorDataset = TensorDataset(norm_inputs, outputs)
    return norm_dataset
