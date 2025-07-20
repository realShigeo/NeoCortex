import torch
from torch import nn
from torch.utils.data import TensorDataset

from neocortex.create_model import create_model
from neocortex.evaluate import evaluate
from neocortex.generate_dataset import generate_dataset
from neocortex.normalize_dataset import normalize_dataset
from neocortex.split_dataset import split_dataset
from neocortex.train import train

# Environment Parameters
MAX_LENGTH: float = 10.5  # meters
VEHICLE_VELOCITY: float = 1.35  # meters / second

# Training Parameters
TRAIN_RATIO: float = 0.8
NUM_EPOCHS: int = 5000
NUM_SAMPLES: int = 5000


def main() -> None:
    torch.manual_seed(42)

    dataset: TensorDataset = generate_dataset(
        NUM_SAMPLES, MAX_LENGTH, VEHICLE_VELOCITY
    )

    norm_dataset: TensorDataset = normalize_dataset(dataset, MAX_LENGTH)

    train_dataset, test_dataset = split_dataset(norm_dataset, TRAIN_RATIO)

    model: nn.Sequential = create_model()

    train(model, train_dataset, NUM_EPOCHS)

    evaluate(model, test_dataset)


if __name__ == "__main__":
    main()
