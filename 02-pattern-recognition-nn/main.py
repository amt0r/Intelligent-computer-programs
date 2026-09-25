"""Pattern recognition experiment: comparing 20 neural network configurations.

Compares 5 architectures x 2 activations x 2 loss functions on a line pattern
classification task (vertical, horizontal, two vertical, two horizontal lines).
"""

from dataset import train_dataset, test_dataset
from network import NeuralNetwork
from stopwatch import Stopwatch


def prepare_data(dataset):
    """Flatten 6x6 matrices into 36-element vectors."""
    X = []
    y = []
    for data in dataset.values():
        flat_matrix = [pixel for row in data["matrix"] for pixel in row]
        X.append(flat_matrix)
        y.append(data["target"])
    return X, y


def create_models():
    """Create all 20 neural network configurations."""
    models = []

    architectures = {
        "No hidden layers": [36, 2],
        "Narrow (4)": [36, 4, 2],
        "Base (16)": [36, 16, 2],
        "Wide (64)": [36, 64, 2],
        "Two hidden layers (16, 8)": [36, 16, 8, 2]
    }

    activations = ['sigmoid', 'relu']
    losses = ['mse', 'cross_entropy']

    for arch_name, layers in architectures.items():
        for act in activations:
            for loss in losses:
                full_name = f"{arch_name} | {act} | {loss}"
                models.append(
                    NeuralNetwork(layers, activation=act, name=full_name, loss_type=loss)
                )

    return models


def evaluate_model(model, X_test, test_dataset):
    """Evaluate model accuracy on test dataset."""
    correct = 0
    total = len(test_dataset)

    for test_data in test_dataset.values():
        flat_matrix = [pixel for row in test_data["matrix"] for pixel in row]
        prediction = model.forward(flat_matrix)
        binary_prediction = [1 if p >= 0.5 else 0 for p in prediction]

        if binary_prediction == test_data["target"]:
            correct += 1

    return (correct / total) * 100


def print_leaderboard(leaderboard):
    """Print sorted results table."""
    print(f"{'Rank':<5}\t| {'Architecture':<55}\t| {'Accuracy':<8}\t| {'Time (s)':<8}\t| Epochs")
    print("-" * 110)

    sorted_board = sorted(
        leaderboard.items(),
        key=lambda x: (-x[1]["accuracy"], x[1]["time"], x[1]["epochs"])
    )

    for rank, (name, stats) in enumerate(sorted_board, start=1):
        print(f"{rank:<5}\t| {name:<55}\t| {stats['accuracy']:>5.1f}%\t| {stats['time']:>8.4f}\t| {stats['epochs']}")


def main():
    X_train, y_train = prepare_data(train_dataset)
    models = create_models()

    leaderboard = {}
    timer = Stopwatch()

    for model in models:
        print(f"\n{model.name}")

        timer.start()
        epochs = model.train(X_train, y_train, learning_rate=0.2,
                             loss_type=model.loss_type, target_error=0.001)
        timer.stop()
        train_time = timer.get_time()
        timer.reset()

        print(f"{epochs} epochs ({train_time:.4f} sec).")

        accuracy = evaluate_model(model, None, test_dataset)
        print(f"Test: {int(accuracy * len(test_dataset) / 100)}/{len(test_dataset)} ({accuracy:.1f}%)")

        leaderboard[model.name] = {"epochs": epochs, "accuracy": accuracy, "time": train_time}

    print("\n" + "=" * 110)
    print_leaderboard(leaderboard)


if __name__ == "__main__":
    main()
