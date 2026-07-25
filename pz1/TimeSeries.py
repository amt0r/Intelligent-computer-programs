from BaseNeuron import BaseNeuron
import math


class TimeSeriesNeuron(BaseNeuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)

        self.last_inputs = []
        self.last_output = 0.0
        self.delta = 0.0

    def forward(self, inputs):
        self.last_inputs = inputs

        self.last_output = super().forward(inputs)
        return self.last_output

    def activation_function(self, S):
        return 1.0 / (1.0 + math.exp(-S))

    def get_derivative(self):
        return self.last_output * (1.0 - self.last_output)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [TimeSeriesNeuron(layer_sizes[i-1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = [neuron.forward(current_inputs) for neuron in layer]
        return current_inputs[0]

    def backward(self, expected, learning_rate):
        # помилка вихідного нейрона
        output_neuron = self.layers[-1][0]
        error = expected - output_neuron.last_output
        output_neuron.delta = error * output_neuron.get_derivative()

        # алгоритм зворотного поширення для прихованих шарів
        for i in reversed(range(len(self.layers) - 1)):
            for j, neuron in enumerate(self.layers[i]):
                error_sum = sum(next_n.weights[j] * next_n.delta for next_n in self.layers[i + 1])
                neuron.delta = error_sum * neuron.get_derivative()

        # оновлення ваг (-v * E'i)
        for layer in self.layers:
            for neuron in layer:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] += learning_rate * neuron.delta * neuron.last_inputs[j]
                neuron.bias += learning_rate * neuron.delta


def main():
    raw_data = [0.07, 3.58, 0.44, 5.33, 0.56, 
                5.24, 1.99, 4.38, 0.89, 4.53, 
                1.82, 4.13, 1.88, 5.97, 1.18]
    
    data = [x / 10.0 for x in raw_data]
    train_data_raw = data[:13]
    
    training_data = []
    for i in range(len(train_data_raw) - 3):
        inputs = train_data_raw[i : i + 3]
        expected = train_data_raw[i + 3]
        training_data.append((inputs, expected))

    nn = NeuralNetwork([3, 3, 2, 1])
    learning_rate = 0.15 # v
    
    # алгоритм зворотного поширення 
    for epoch in range(100000):
        E_total = 0.0
        
        for inputs, expected in training_data:
            pred = nn.forward(inputs)
            E_total += (pred - expected) ** 2
            
            nn.backward(expected, learning_rate)
            
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Error: {E_total:.6f}")


    print("\n--- Test ---")
    print("Expected | Predicted | Difference")
    print("-------------------------------")
    
    inputs_14 = data[10:13]
    expected_14 = raw_data[13]
    pred_14 = nn.forward(inputs_14) * 10.0
    print(f"  {expected_14:.2f}    |  {pred_14:.2f}   |  {abs(expected_14 - pred_14):.2f}")
    
    inputs_15 = data[11:14]
    expected_15 = raw_data[14]
    pred_15 = nn.forward(inputs_15) * 10.0
    print(f"  {expected_15:.2f}    |  {pred_15:.2f}   |  {abs(expected_15 - pred_15):.2f}")

if __name__ == "__main__":
    main()