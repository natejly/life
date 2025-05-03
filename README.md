# Game of Life for Finding Optimal Parameters for MCTS

**Author:** Nate Ly

## Motivation

- **Evolutionary Game Theory:** Inspired by Conway’s Game of Life, where each cell is an agent with unique gameplay strategies, and population dynamics are observed over time.
- **Genetic Algorithms:** Leveraging evolutionary processes to tune heuristic parameters used in game-playing algorithms such as minimax or alpha-beta pruning.

## Conway’s Game of Life Rules

1. Any live cell with fewer than two live neighbours dies (underpopulation).
2. Any live cell with two or three live neighbours lives on to the next generation.
3. Any live cell with more than three live neighbours dies (overpopulation).
4. Any dead cell with exactly three live neighbours becomes a live cell (reproduction).

## Agents as MCTS Players

Each cell represents an agent using Monte Carlo Tree Search (MCTS) with the following tunable parameters:

- **Exploration Constant (C):** Balances exploration vs. exploitation.
- **Number of Rollouts per Move:** How many simulations each agent runs per decision.
- **Rollout Depth:** Maximum depth of each simulation.
- **Backpropagation Weights:** How simulation results are propagated through the tree.

## Simulation Phases

### 1. Competition Phase

- Agents play modified Game of Life matches against their neighbors, using their MCTS parameters.
- Competitions are conducted as best-of-`N` games.
- Agents gain or lose fitness points based on game outcomes (±1) or score differences.

### 2. Elimination Phase

- Cells with fitness below a defined threshold are removed (fastest elimination).
- Additionally, cells face a probabilistic death, where survival probability is proportional to fitness and age.

### 3. Birth Phase

- Following the standard revival rule (dead cell with three live neighbours), new agents are born.
- New agents adopt parameters from neighboring survivors using an ε-greedy strategy:
  - With probability ε, copy parameters from a random neighbor.
  - With probability 1−ε, copy parameters from the best neighboring agent.
- Parameters may mutate upon birth to introduce variation.

## Project Goal

To evolve a population of MCTS agents that discover optimal parameter settings through simulated competition and mutation, and to observe spatial clusters of locally optimal strategies.

## Potential Challenges

- Defining parameter ranges that balance strategic depth with computational feasibility.
- Choosing initial distributions for agent parameters and population density.
- Setting an appropriate ε value to balance exploration of new parameters and exploitation of known good ones.
- Managing computational costs and board size for large simulations.

## Discussion Questions

- **Approach Justification:** While not guaranteed optimal, this evolutionary framework offers insight into adaptive parameter tuning.
- **Alternative Methods:** Grid search, hill climbing, or machine learning techniques might provide more direct optimization.
- **Relation to Course Projects:** Builds upon existing MCTS implementations (e.g., Cribbage, Kalah) used in class.
- **Evaluation Strategy:** Benchmark the highest-fitness evolved agent against a baseline MCTS agent to measure performance improvements.
