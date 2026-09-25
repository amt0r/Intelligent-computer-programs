"""Q-Learning agent with epsilon-greedy exploration."""

import random
import numpy as np


class QLearningAgent:
    """Agent that learns optimal policy using Q-Learning algorithm."""

    GAMMA = 0.8

    def __init__(self, environment, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = environment
        self.q_matrix = np.zeros((environment.NUM_STATES, environment.NUM_ACTIONS))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.NUM_ACTIONS - 1)
        return int(np.argmax(self.q_matrix[state]))

    def update_q_value(self, state, action):
        next_state = self.env.get_next_state(state, action)
        reward = self.env.reward_matrix[state, action]
        max_future_q = np.max(self.q_matrix[next_state])
        self.q_matrix[state, action] = reward + self.GAMMA * max_future_q
        return next_state

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_optimal_path(self):
        path = [self.env.START_STATE]
        state = self.env.START_STATE
        visited = set()

        while not self.env.is_goal(state) and len(path) < self.env.NUM_STATES:
            if state in visited:
                break
            visited.add(state)
            action = int(np.argmax(self.q_matrix[state]))
            state = self.env.get_next_state(state, action)
            path.append(state)

        return path
