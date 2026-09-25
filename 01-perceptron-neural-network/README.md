# 01 — Perceptron & Neural Network

Single-layer and multi-layer neural networks built from scratch to solve logic gate problems and time series forecasting.

## Structure

```
├── BaseNeuron.py      # Abstract base neuron with weighted sum calculation
├── LogicNeuron.py     # Step-activation neuron + AND, OR, NOT gates
├── XORNetwork.py      # Two-layer network solving XOR (non-linearly separable)
└── TimeSeries.py      # Feedforward NN with backpropagation for time series prediction
```

## How It Works

### Logic Gates (Perceptron)

`BaseNeuron` computes a weighted sum \( S = \sum x_i w_i + b \). `LogicNeuron` applies a step activation function (output 1 if \( S \geq 0 \), else 0). Specific gates are created by setting weights and bias manually:

| Gate | Weights | Bias |
|------|---------|------|
| AND  | [1, 1]  | -1.5 |
| OR   | [1, 1]  | -0.5 |
| NOT  | [-1]    | 0    |

### XOR Network

XOR is not linearly separable, so it requires a hidden layer. The network uses two hidden neurons feeding into an OR gate to solve it.

### Time Series Prediction

A multi-layer network `[3, 3, 2, 1]` with sigmoid activation learns to predict the next value in a numeric sequence using backpropagation (100,000 epochs).

## Usage

```bash
python LogicNeuron.py    # Test AND, OR, NOT gates
python XORNetwork.py     # Test XOR network
python TimeSeries.py     # Train and test time series prediction
```
