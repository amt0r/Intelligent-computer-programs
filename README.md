# AI Algorithms From Scratch

A collection of fundamental AI and machine learning algorithms implemented from scratch in Python — no high-level ML frameworks, just pure math and code.

## Projects

| # | Project | Description | Key Concepts |
|---|---------|-------------|--------------|
| 1 | [Perceptron & Neural Network](./01-perceptron-neural-network/) | Logic gate neurons (AND, OR, NOT, XOR) and time series prediction | Perceptron, step activation, sigmoid, backpropagation |
| 2 | [Pattern Recognition NN](./02-pattern-recognition-nn/) | Image pattern classifier comparing 20 network configurations | Feedforward NN, Sigmoid vs ReLU, MSE vs Cross-Entropy |
| 3 | [Genetic Algorithm](./03-genetic-algorithm/) | Function optimization using island-model genetic algorithm | Selection, crossover, mutation, parallel island evolution |
| 4 | [Q-Learning Agent](./04-q-learning-agent/) | Reinforcement learning agent navigating a grid world | Q-table, epsilon-greedy, reward matrix, Pygame visualization |

## Tech Stack

- **Language:** Python 3
- **Visualization:** Matplotlib, Pygame
- **Computation:** NumPy (only for Q-table matrix operations)
- **Parallelism:** `concurrent.futures` (island model)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-algorithms-from-scratch.git
cd ai-algorithms-from-scratch

# Install dependencies
pip install numpy matplotlib pygame

# Run any project
cd 01-perceptron-neural-network
python LogicNeuron.py      # Logic gates demo
python TimeSeries.py       # Time series prediction

cd ../02-pattern-recognition-nn
python main.py             # Pattern recognition experiment

cd ../03-genetic-algorithm
python main.py             # Genetic algorithm optimization

cd ../04-q-learning-agent
python main.py             # Q-Learning grid world
```

## License

This project is open source and available under the [MIT License](LICENSE).
