import torch
import torch.nn as nn
from torch import Tensor


def train(model: nn.Sequential, x_train: Tensor, y_train: Tensor) -> None:
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(500):
        y_prediction = model(x_train)
        loss = loss_function(y_prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")
