from torch import nn


def create_model() -> nn.Sequential:
    model: nn.Sequential = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    return model
