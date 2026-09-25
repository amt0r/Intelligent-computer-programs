"""Neural network implementation with backpropagation for pattern recognition."""

import random
import math


class BaseNeuron:
    """Base neuron with configurable weights and bias."""

    def __init__(self, num_inputs, weights=None, bias=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]

        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)

    def calculate_sum(self, inputs):
        return sum(x * w for x, w in zip(inputs, self.weights)) + self.bias


class Neuron(BaseNeuron):
    """Neuron with configurable activation function (sigmoid or relu)."""

    def __init__(self, num_inputs, activation_type='sigmoid'):
        super().__init__(num_inputs)
        self.activation_type = activation_type
        self.last_inputs = []
        self.last_output = 0.0
        self.delta = 0.0

    def forward(self, inputs):
        self.last_inputs = inputs
        S = self.calculate_sum(inputs)

        if self.activation_type == 'sigmoid':
            self.last_output = 1.0 / (1.0 + math.exp(-S))
        elif self.activation_type == 'relu':
            self.last_output = max(0.0, S)

        return self.last_output

    def get_derivative(self):
        if self.activation_type == 'sigmoid':
            return self.last_output * (1.0 - self.last_output)
        elif self.activation_type == 'relu':
            return 1.0 if self.last_output > 0 else 0.0


class NeuralNetwork:
    """Feedforward neural network with backpropagation training.

    Supports configurable architecture, activation functions (sigmoid/relu),
    and loss functions (MSE/cross-entropy).
    """

    def __init__(self, layer_sizes, activation='sigmoid', name="Unnamed Model", loss_type='mse'):
        self.name = name
        self.loss_type = loss_type
        self.layers = []

        for i in range(1, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                layer_activation = 'sigmoid'
            else:
                layer_activation = activation

            layer = [Neuron(layer_sizes[i - 1], activation_type=layer_activation) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = [neuron.forward(current_inputs) for neuron in layer]
        return current_inputs

    def backward(self, expected, learning_rate, loss_type='mse'):
        for j, output_neuron in enumerate(self.layers[-1]):
            if loss_type == 'mse':
                error = expected[j] - output_neuron.last_output
                output_neuron.delta = error * output_neuron.get_derivative()
            elif loss_type == 'cross_entropy':
                output_neuron.delta = expected[j] - output_neuron.last_output

        for i in reversed(range(len(self.layers) - 1)):
            for j, neuron in enumerate(self.layers[i]):
                error_sum = sum(next_n.weights[j] * next_n.delta for next_n in self.layers[i + 1])
                neuron.delta = error_sum * neuron.get_derivative()

        for layer in self.layers:
            for neuron in layer:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] += learning_rate * neuron.delta * neuron.last_inputs[j]
                neuron.bias += learning_rate * neuron.delta

    def train(self, X, y, learning_rate=0.1, loss_type='mse', target_error=0.005):
        epoch = 0

        while True:
            epoch_loss = 0.0

            for i in range(len(X)):
                output = self.forward(X[i])
                self.backward(y[i], learning_rate, loss_type)

                for j in range(len(y[i])):
                    epoch_loss += (y[i][j] - output[j]) ** 2

            mean_loss = epoch_loss / len(X)
            epoch += 1

            if mean_loss <= target_error:
                break

            if epoch >= 100_000:
                print(f"Stopped at 100,000 epochs. Final error: {mean_loss:.6f}")
                break

        return epoch
