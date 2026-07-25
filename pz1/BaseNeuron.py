import random

class BaseNeuron:
    def __init__(self, num_inputs, weights=None, bias=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
            
        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)

    def forward(self, inputs):
        S = self.calculate_sum(inputs)
        return self.activation_function(S)

    def calculate_sum(self, inputs):
        return sum(x * w for x, w in zip(inputs, self.weights)) + self.bias

    def activation_function(self, S):
        raise NotImplementedError("todo")