import torch.nn as nn


def create_model(hidden_size: int = 64) -> nn.Sequential:
    model: nn.Sequential = nn.Sequential(
        nn.Linear(4, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
    return model
