import torch.nn as nn


def build_model() -> nn.Sequential:
    hidden_size: int = 64
    model: nn.Sequential = nn.Sequential(
        nn.Linear(4, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
    return model
