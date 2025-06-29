import torch
from torch import nn
from torch.utils.data import TensorDataset


def evaluate(model: nn.Sequential, test_dataset: TensorDataset) -> None:
    model.eval()
    X_test, y_test = (  # pylint: disable=invalid-name
        test_dataset.tensors
    )  # shape: (num_test_samples, 4), (num_test_samples, 2)

    with torch.no_grad():
        preds = model(X_test)

        mse = torch.mean((preds - y_test) ** 2).item()
        mae = torch.mean(torch.abs(preds - y_test)).item()
        vector_error = torch.norm(preds - y_test, dim=1).mean().item()

    print(f"Evaluation MSE: {mse:.4f}")
    print(f"Evaluation MAE: {mae:.4f}")
    print(f"Mean Vector Error: {vector_error:.4f}")
