from LogicNeuron import LogicNeuron, OrNeuron

class HiddenNeuron1(LogicNeuron):
    def __init__(self):
        super().__init__(num_inputs=2, weights=[1, -1], bias=-0.5)

class HiddenNeuron2(LogicNeuron):
    def __init__(self):
        super().__init__(num_inputs=2, weights=[-1, 1], bias=-0.5)

class XORNetwork:
    def __init__(self):
        self.hidden1 = HiddenNeuron1()
        self.hidden2 = HiddenNeuron2()
        
        self.output_neuron = OrNeuron()

    def forward(self, inputs):
        out_h1 = self.hidden1.forward(inputs)
        out_h2 = self.hidden2.forward(inputs)
        
        final_out = self.output_neuron.forward([out_h1, out_h2])
        
        return final_out

# Блок для тестування
if __name__ == "__main__":
    xor_net = XORNetwork()
    
    print("--- XOR ---")
    print("x1 | x2 | Y")
    print("-----------")
    
    for x1, x2 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        y = xor_net.forward([x1, x2])
        print(f" {x1} |  {x2} | {y}")