import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch import Tensor
from torch.nn import Sequential


def evaluate(model: Sequential, x_test: Tensor, y_test: Tensor) -> None:
    loss_function = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        y_prediction = model(x_test)
        test_loss = loss_function(y_prediction, y_test)

    # Print out R^2 score
    r2 = r2_score(y_test, y_prediction)
    print(f"R^2 Score: {r2:.4f}")

    print(f"\nTest Loss: {test_loss.item():.4f}")
