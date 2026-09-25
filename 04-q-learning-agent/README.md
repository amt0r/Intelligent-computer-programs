# 04 — Q-Learning Agent

A reinforcement learning agent that learns to navigate a 7×5 grid world from start to goal, avoiding obstacles — visualized in real-time with Pygame.

## Structure

```
├── environment.py   # Grid world: states, actions, rewards, obstacles
├── agent.py         # Q-Learning agent with epsilon-greedy exploration
├── visualizer.py    # Pygame renderer for grid, agent, and path
├── simulation.py    # Training loop orchestrator
├── main.py          # Entry point
├── octopus.png      # Agent sprite
└── star.png         # Goal sprite
```

## Grid World

A 7×5 grid (35 states) where the agent must navigate from **state 34** (bottom-left area) to **state 2** (top area) while avoiding 12 obstacle cells.

## Algorithm

**Q-Learning** updates the Q-table using the Bellman equation:

$$Q(s, a) = R(s, a) + \gamma \cdot \max_{a'} Q(s', a')$$

- **γ = 0.8** — discount factor for future rewards
- **ε-greedy** exploration: starts at ε=1.0, decays by 0.995 per episode (min 0.01)
- **300 training episodes**, max 200 steps each
- Actions: UP, DOWN, LEFT, RIGHT

### Rewards

| Transition | Reward |
|------------|--------|
| Reach goal | +100 |
| Valid move | 0 |
| Wall/obstacle/boundary | -1 |

## Usage

```bash
pip install numpy pygame
python main.py
```

The Pygame window shows the agent learning in real-time. After training, the optimal path is animated. Press **ESC** to close.
