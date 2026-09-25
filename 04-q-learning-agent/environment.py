"""Grid world environment for Q-Learning agent."""

import numpy as np


class Environment:
    """7x5 grid world with obstacles, start and goal states."""

    ROWS = 7
    COLS = 5
    NUM_STATES = ROWS * COLS
    GOAL_STATE = 2
    START_STATE = 34
    OBSTACLES = {6, 7, 9, 11, 14, 19, 20, 21, 22, 29, 31, 32}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    NUM_ACTIONS = 4
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self):
        self.reward_matrix = self._build_reward_matrix()

    def _build_reward_matrix(self):
        rewards = np.full((self.NUM_STATES, self.NUM_ACTIONS), -1.0)

        for state in range(self.NUM_STATES):
            if state in self.OBSTACLES:
                continue

            if state == self.GOAL_STATE:
                rewards[state, :] = 100.0
                continue

            row, col = divmod(state, self.COLS)

            neighbors = {
                self.ACTION_UP: (row - 1, col),
                self.ACTION_DOWN: (row + 1, col),
                self.ACTION_LEFT: (row, col - 1),
                self.ACTION_RIGHT: (row, col + 1),
            }

            for action, (next_row, next_col) in neighbors.items():
                if 0 <= next_row < self.ROWS and 0 <= next_col < self.COLS:
                    next_state = next_row * self.COLS + next_col
                    if next_state not in self.OBSTACLES:
                        if next_state == self.GOAL_STATE:
                            rewards[state, action] = 100.0
                        else:
                            rewards[state, action] = 0.0

        return rewards

    def get_next_state(self, state, action):
        if state == self.GOAL_STATE:
            return self.GOAL_STATE

        row, col = divmod(state, self.COLS)
        moves = {
            self.ACTION_UP: (row - 1, col),
            self.ACTION_DOWN: (row + 1, col),
            self.ACTION_LEFT: (row, col - 1),
            self.ACTION_RIGHT: (row, col + 1),
        }

        next_row, next_col = moves[action]

        if not (0 <= next_row < self.ROWS and 0 <= next_col < self.COLS):
            return state

        next_state = next_row * self.COLS + next_col

        if next_state in self.OBSTACLES:
            return state

        return next_state

    def is_goal(self, state):
        return state == self.GOAL_STATE

    def state_to_grid(self, state):
        return divmod(state, self.COLS)
