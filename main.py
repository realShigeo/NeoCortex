import torch
from torch import nn
from torch.utils.data import TensorDataset

from neocortex.create_model import create_model
from neocortex.evaluate import evaluate
from neocortex.generate_dataset import generate_dataset
from neocortex.normalize_dataset import normalize_dataset
from neocortex.split_dataset import split_dataset
from neocortex.train import train

NUMBER_OF_SAMPLES: int = 5000
BOX_SIZE: float = 0.1
TRAIN_RATIO: float = 0.8
NUM_EPOCHS: int = 5000


def main() -> None:
    torch.manual_seed(42)

    dataset: TensorDataset = generate_dataset(NUMBER_OF_SAMPLES, BOX_SIZE)

    norm_dataset: TensorDataset = normalize_dataset(dataset, BOX_SIZE)

    train_dataset, test_dataset = split_dataset(norm_dataset, TRAIN_RATIO)

    model: nn.Sequential = create_model()

    train(model, train_dataset, NUM_EPOCHS)

    evaluate(model, test_dataset)


if __name__ == "__main__":
    main()
