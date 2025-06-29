import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset


def train(
    model: nn.Sequential, train_dataset: TensorDataset, num_epochs: int = 5000
) -> None:
    X_train, y_train = (  # pylint: disable=invalid-name
        train_dataset.tensors
    )  # shape: (num_training_samples, 4), (num_training_samples, 2)

    loss_function: nn.MSELoss = nn.MSELoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        preds: Tensor = model(X_train)  # shape: (num_training_samples, 2)
        loss: Tensor = loss_function(preds, y_train)  # shape: ()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")
