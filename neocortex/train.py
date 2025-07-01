from torch import Tensor, nn, optim
from torch.utils.data import TensorDataset


def train(
    model: nn.Sequential, dataset: TensorDataset, num_epochs: int
) -> None:
    X_train, y_train = dataset.tensors  # pylint: disable=invalid-name
    X_train = X_train.unsqueeze(1)  # pylint: disable=invalid-name
    y_train = y_train.unsqueeze(1)

    loss_function: nn.MSELoss = nn.MSELoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        predictions: Tensor = model(X_train)
        loss: Tensor = loss_function(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
