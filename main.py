from torch.nn import Sequential

from neocortex.data import generate_dataset
from neocortex.evaluate import evaluate
from neocortex.model import build_model
from neocortex.train import train


def main() -> None:
    x_train, y_train, x_test, y_test = generate_dataset(number_of_samples=5000)

    model: Sequential = build_model()

    train(model, x_train, y_train)

    evaluate(model, x_test, y_test)


if __name__ == "__main__":
    main()
