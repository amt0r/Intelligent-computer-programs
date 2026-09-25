"""Entry point for Q-Learning grid world simulation."""

from simulation import QLearningSimulation


if __name__ == "__main__":
    simulation = QLearningSimulation()
    simulation.run()
