"""Pygame-based visualization for Q-Learning grid world."""

import sys
import pygame


class Visualizer:
    """Renders the grid world, agent movement, and training progress."""

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
        pygame.display.set_caption("Q-Learning Agent")
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

        epsilon_text = self.header_font.render(f"\u03b5 = {epsilon:.4f}", True, self.COLOR_EPSILON_TEXT)
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
