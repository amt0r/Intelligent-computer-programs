# 02 — Pattern Recognition Neural Network

A feedforward neural network that classifies 6×6 binary images of line patterns. Compares **20 different configurations** to find the optimal architecture.

## Structure

```
├── dataset.py     # Training (4 patterns) and test (12 noisy patterns) datasets
├── network.py     # BaseNeuron, Neuron, and NeuralNetwork classes
├── stopwatch.py   # Timer utility for benchmarking training
└── main.py        # Experiment runner: trains all configs, prints leaderboard
```

## Patterns

The network recognizes 4 types of line patterns on a 6×6 grid:

| Pattern | Target Vector |
|---------|---------------|
| Vertical line | `[0, 0]` |
| Horizontal line | `[0, 1]` |
| Two vertical lines | `[1, 0]` |
| Two horizontal lines | `[1, 1]` |

Test images include noise (1–3 flipped pixels) to evaluate generalization.

## Experiment

20 configurations are tested — all combinations of:

- **5 Architectures:** No hidden layers, Narrow (4), Base (16), Wide (64), Two hidden layers (16+8)
- **2 Activations:** Sigmoid, ReLU
- **2 Loss Functions:** MSE, Cross-Entropy

### Key Findings

1. **ReLU + Cross-Entropy** is the best combination — converges in as few as 9 epochs
2. Models **without hidden layers fail** — the patterns are not linearly separable
3. **Base (16 neurons)** architecture provides the best speed/accuracy balance

## Usage

```bash
python main.py
```
