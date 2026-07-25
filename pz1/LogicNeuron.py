from BaseNeuron import BaseNeuron   

class LogicNeuron(BaseNeuron):
    def activation_function(self, S):
        return 1 if S >= 0 else 0

class AndNeuron(LogicNeuron):
    def __init__(self):
        # зміщення -1.5 (поріг 1.5)
        super().__init__(num_inputs=2, weights=[1, 1], bias=-1.5)


class NotNeuron(LogicNeuron):
    def __init__(self):
        # зміщення 0
        super().__init__(num_inputs=1, weights=[-1], bias=0)


class OrNeuron(LogicNeuron):
    def __init__(self):
        # зміщення -0.5 (поріг 0.5)
        super().__init__(num_inputs=2, weights=[1, 1], bias=-0.5)


if __name__ == "__main__":

    and_neuron = AndNeuron()
    print("--- AND ---")
    print("x1 | x2 | Y")
    print("-----------")
    for x1, x2 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f" {x1} |  {x2} | {and_neuron.forward([x1, x2])}")

    not_neuron = NotNeuron()
    print("\n--- NOT ---")
    print(" x | Y")
    print("-------")
    for x in [[0], [1]]:
        print(f" {x[0]} | {not_neuron.forward(x)}")

    or_neuron = OrNeuron()
    print("\n--- OR ---")
    print("x1 | x2 | Y")
    print("-----------")
    for x1, x2 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f" {x1} |  {x2} | {or_neuron.forward([x1, x2])}")