"""Q-Learning training simulation with visualization."""

import numpy as np
import pygame

from environment import Environment
from agent import QLearningAgent
from visualizer import Visualizer


class QLearningSimulation:
    """Runs the full Q-Learning training loop with Pygame visualization."""

    TRAINING_EPISODES = 300
    MAX_STEPS_PER_EPISODE = 200
    EXPLORATION_FPS = 30
    FINAL_PATH_FPS = 3

    def __init__(self):
        self.env = Environment()
        self.agent = QLearningAgent(self.env)
        self.vis = Visualizer(self.env)

    def run(self):
        self._run_training()
        self._show_final_path()
        self._wait_for_close()

    def _run_training(self):
        for episode in range(1, self.TRAINING_EPISODES + 1):
            state = self.env.START_STATE
            path = [state]

            for _ in range(self.MAX_STEPS_PER_EPISODE):
                self.vis.handle_events()

                action = self.agent.choose_action(state)
                next_state = self.agent.update_q_value(state, action)
                path.append(next_state)

                if episode % 10 == 0 or episode <= 5:
                    self.vis.draw_grid(next_state, episode, self.agent.epsilon, path=path)
                    self.vis.tick(self.EXPLORATION_FPS)

                state = next_state

                if self.env.is_goal(state):
                    break

            self.agent.decay_epsilon()

            if episode % 10 == 0 or episode <= 5:
                self.vis.draw_grid(state, episode, self.agent.epsilon, path=path)
                pygame.time.wait(100)

        self._print_q_matrix_summary()

    def _show_final_path(self):
        optimal_path = self.agent.get_optimal_path()
        print(f"\nOptimal path: {' -> '.join(map(str, optimal_path))}")
        print(f"Path length: {len(optimal_path) - 1} steps")

        for i, state in enumerate(optimal_path):
            self.vis.handle_events()
            visited_so_far = optimal_path[: i + 1]
            self.vis.draw_grid(state, "FINAL", self.agent.epsilon, optimal_path=visited_so_far)
            self.vis.tick(self.FINAL_PATH_FPS)

        self.vis.draw_grid(optimal_path[-1], "FINAL", self.agent.epsilon, optimal_path=optimal_path)

    def _print_q_matrix_summary(self):
        print("\n=== Q-Matrix (non-zero rows) ===")
        for state in range(self.env.NUM_STATES):
            if np.any(self.q_nonzero(state)):
                values = self.agent.q_matrix[state]
                formatted = [f"{v:8.2f}" for v in values]
                best_action = self.env.ACTION_NAMES[int(np.argmax(values))]
                print(f"  State {state:2d}: [{', '.join(formatted)}]  -> {best_action}")

    def q_nonzero(self, state):
        return self.agent.q_matrix[state] != 0

    def _wait_for_close(self):
        print("\nPress ESC or close the window to exit.")
        while True:
            self.vis.handle_events()
            self.vis.tick(10)
