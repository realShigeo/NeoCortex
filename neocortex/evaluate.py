import torch
from torch import nn
from torch.utils.data import TensorDataset


def evaluate(model: nn.Sequential, dataset: TensorDataset) -> None:
    model.eval()
    X_test, y_test = dataset.tensors  # pylint: disable=invalid-name
    X_test = X_test.unsqueeze(1)  # pylint: disable=invalid-name
    y_test = y_test.unsqueeze(1)

    with torch.no_grad():
        predictions = model(X_test)

        mse = torch.mean((predictions - y_test) ** 2).item()
        mae = torch.mean(torch.abs(predictions - y_test)).item()

    print(f"Evaluation MSE: {mse:.4f}")
    print(f"Evaluation MAE: {mae:.4f}")
