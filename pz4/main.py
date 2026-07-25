import numpy as np
import pygame
import random
import sys


class Environment:
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
        self.reward_matrix = self._build_reward_matrix() # R

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
                if 0 <= next_row < self.ROWS and 0 <= next_col < self.COLS: #межі мапи
                    next_state = next_row * self.COLS + next_col 
                    if next_state not in self.OBSTACLES: # перепона
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


class QLearningAgent:
    GAMMA = 0.8 # майбутні винагороди

    def __init__(self, environment, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = environment
        self.q_matrix = np.zeros((environment.NUM_STATES, environment.NUM_ACTIONS)) # Q
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state): # епсілон жадібність
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


class Visualizer:
    CELL_SIZE = 100
    GRID_LINE_WIDTH = 2

    COLOR_BACKGROUND = (30, 30, 46)
    COLOR_GRID_LINE = (69, 71, 90)
    COLOR_FREE_CELL = (49, 50, 68)
    COLOR_OBSTACLE = (137, 100, 186)
    COLOR_GOAL_CELL = (249, 226, 175)
    COLOR_PATH_CELL = (166, 227, 161)
    COLOR_AGENT_CELL = (137, 180, 250)
    COLOR_TEXT = (205, 214, 244)
    COLOR_HEADER_BG = (24, 24, 37)
    COLOR_EPISODE_TEXT = (250, 179, 135)
    COLOR_EPSILON_TEXT = (148, 226, 213)

    HEADER_HEIGHT = 60

    def __init__(self, environment):
        self.env = environment
        pygame.init()
        self.screen_width = self.env.COLS * self.CELL_SIZE
        self.screen_height = self.env.ROWS * self.CELL_SIZE + self.HEADER_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Q-Learning Agent — Variant 8")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 14)
        self.header_font = pygame.font.SysFont("Segoe UI", 22, bold=True)

        self.octopus_image = self._load_and_scale_image("octopus.png")
        self.star_image = self._load_and_scale_image("star.png")

    def _load_and_scale_image(self, filename):
        image = pygame.image.load(filename)
        return pygame.transform.scale(image, (self.CELL_SIZE - 10, self.CELL_SIZE - 10))

    def draw_grid(self, agent_state, episode, epsilon, path=None, optimal_path=None):
        self.screen.fill(self.COLOR_BACKGROUND)
        self._draw_header(episode, epsilon)

        optimal_set = set(optimal_path) if optimal_path else set()
        path_set = set(path) if path else set()

        for state in range(self.env.NUM_STATES):
            row, col = self.env.state_to_grid(state)
            x = col * self.CELL_SIZE
            y = row * self.CELL_SIZE + self.HEADER_HEIGHT

            if state in self.env.OBSTACLES:
                cell_color = self.COLOR_OBSTACLE
            elif optimal_path and state in optimal_set:
                cell_color = self.COLOR_PATH_CELL
            elif path and state in path_set:
                cell_color = self.COLOR_PATH_CELL
            elif state == self.env.GOAL_STATE:
                cell_color = self.COLOR_GOAL_CELL
            else:
                cell_color = self.COLOR_FREE_CELL

            pygame.draw.rect(self.screen, cell_color, (x + 1, y + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2))

            if state == self.env.GOAL_STATE and state != agent_state:
                self.screen.blit(self.star_image, (x + 5, y + 5))

            if state == agent_state:
                self.screen.blit(self.octopus_image, (x + 5, y + 5))

            state_label = self.small_font.render(str(state), True, self.COLOR_GRID_LINE)
            self.screen.blit(state_label, (x + 4, y + 2))

        for row in range(self.env.ROWS + 1):
            y = row * self.CELL_SIZE + self.HEADER_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, y), (self.screen_width, y), self.GRID_LINE_WIDTH)

        for col in range(self.env.COLS + 1):
            x = col * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.HEADER_HEIGHT), (x, self.screen_height), self.GRID_LINE_WIDTH)

        pygame.display.flip()

    def _draw_header(self, episode, epsilon):
        pygame.draw.rect(self.screen, self.COLOR_HEADER_BG, (0, 0, self.screen_width, self.HEADER_HEIGHT))

        episode_text = self.header_font.render(f"Episode: {episode}", True, self.COLOR_EPISODE_TEXT)
        self.screen.blit(episode_text, (15, 15))

        epsilon_text = self.header_font.render(f"ε = {epsilon:.4f}", True, self.COLOR_EPSILON_TEXT)
        self.screen.blit(epsilon_text, (self.screen_width - epsilon_text.get_width() - 15, 15))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    def tick(self, fps):
        self.clock.tick(fps)

    def cleanup(self):
        pygame.quit()


class QLearningSimulation:
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


if __name__ == "__main__":
    simulation = QLearningSimulation()
    simulation.run()
