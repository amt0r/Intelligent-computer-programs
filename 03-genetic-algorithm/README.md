# 03 — Genetic Algorithm (Island Model)

A parallel genetic algorithm using the **island model** to find the global minimum and maximum of a mathematical function.

## Structure

```
├── config.py       # Target function, search range, precision settings
├── individual.py   # Binary-encoded individual with crossover and mutation
├── island.py       # Island evolution engine with tournament selection
├── visualizer.py   # Matplotlib plot of function with found extrema
└── main.py         # Interactive CLI entry point
```

## Target Function

$$Y(x) = \frac{1}{x} \cdot \cos\left(x^2 + \frac{1}{x}\right), \quad x \in [1, 10]$$

## Algorithm

1. **Encoding:** Real values are encoded as 17-bit binary strings (precision ε = 0.0001)
2. **Selection:** Tournament selection preserves genetic diversity
3. **Crossover:** Single-point crossover produces offspring
4. **Mutation:** Bit-flip mutation with configurable rate
5. **Elitism:** Best individuals survive to the next generation
6. **Island Model:** Multiple populations evolve independently in parallel using `ProcessPoolExecutor`, then the global best is selected

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population size | 3 | Individuals per island |
| Mutation rate | 0.15 | Probability of bit flip |
| Elitism rate | 0.1 | Fraction of elites preserved |
| Tournament size | 3 | Candidates per tournament |
| Islands | 12 | Number of parallel populations |
| Generations | 60 | Iterations per island |

## Usage

```bash
python main.py
# Enter parameters interactively or press Enter for defaults
```
