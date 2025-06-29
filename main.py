import torch
from torch import nn
from torch.utils.data import TensorDataset

from neocortex.create_model import create_model
from neocortex.evaluate import evaluate
from neocortex.generate_dataset import generate_dataset
from neocortex.split_dataset import split_dataset
from neocortex.train import train


def main() -> None:
    torch.manual_seed(42)

    dataset: TensorDataset = generate_dataset(box_size=1000)

    train_dataset, test_dataset = split_dataset(dataset)

    model: nn.Sequential = create_model()

    train(model, train_dataset)

    evaluate(model, test_dataset)


if __name__ == "__main__":
    main()
