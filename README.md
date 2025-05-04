# Game of Life for MCTS Parameter Search

**Author:** Nate Ly

## Motivation

- **Evolutionary Game Theory:** Inspired by Conway’s Game of Life, where each cell is an agent with unique gameplay strategies, and population dynamics are observed over time.
- **Genetic Algorithms:** Leveraging evolutionary processes to tune heuristic parameters used in game-playing algorithms

## Agents as MCTS Players

Each cell represents an agent using Monte Carlo Tree Search (MCTS) with the following tunable parameters:

- **Exploration Constant (C):** Balances exploration vs. exploitation.
- **Number of Rollouts per Move:** How many simulations each agent runs per decision.
- **Rollout Depth:** Maximum depth of each simulation.
- **Backpropagation Weights:** How simulation results are propagated through the tree.
- **Backpropagation Decay:** How simulation results are discounted through the tree.

### 1. Competition Phase

- Agents play modified Game of Life matches against their neighbors, using their MCTS parameters.
- After N games the average margin is calculated
- Agents gain or lose fitness points based on game margin

### 2. Elimination Phase

- Fitness of all cells is decayed to discourage st
- Cells with fitness below 0 are killed

### 3. Birth Phase

- Following the standard revival rule (dead cell with three live neighbours), new agents are born.
- New agents adopt parameters from neighboring survivors using an ε-greedy strategy:
  - With probability ε, copy parameters from a random neighbor.
  - With probability 1−ε, copy parameters from the best neighboring agent.
- Parameters may mutate upon birth to introduce variation.
