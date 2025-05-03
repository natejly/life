from mcts import mcts_policy
from compare import test_game
from peg_game import PeggingGame
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(420)
C = math.sqrt(2)
min_cells = 25
def plot_grid(grid):
    arr = np.array([[1 if cell != 0 else 0 for cell in row] for row in grid])

    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.imshow(arr, cmap='Greys', interpolation='none', origin='lower')

    n = arr.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
class MCTSAGENT():
    def __init__(self, time_limit=None, constant=None, rollouts=None, max_depth=None, weight=None, decay=None):
        self.time_limit = time_limit
        self.constant = constant
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.weight = weight
        self.decay = decay
        self.x = 0
        self.y = 0
        # Create the policy function
        self.policy_generator = mcts_policy(
            limit=self.time_limit,
            constant=self.constant, 
            rollouts=self.rollouts, 
            max_depth=self.max_depth, 
            weight=self.weight, 
            decay=self.decay
        )
    
    # Define policy method that can be called without parameters
    def policy(self):
        return self.policy_generator
    
    def update_policy(self, constant, rollouts, max_depth, weight, decay):
        self.policy_generator = mcts_policy(
            limit=self.time_limit,
            constant=constant,
            rollouts=rollouts,
            max_depth=max_depth,
            weight=weight,
            decay=decay
        )
    
    
def make_agent_dict():
    return {
        "time_limit": 0.005,
        "constant": C,
        "rollouts": 100,
        "max_depth": 4,
        "weight": 1,
        "decay": 1,
        "fitness": 0.0
    }



def grid_to_agent(i,j):
    d = make_agent_dict()
    for key, value in d.items():
        if key == "time_limit":
            time_limit = value
        elif key == "constant":
            constant = value
        elif key == "rollouts":
            rollouts = value
        elif key == "max_depth":
            max_depth = value
        elif key == "weight":
            weight = value
        elif key == "decay":
            decay = value
    agent = MCTSAGENT(time_limit, constant, rollouts, max_depth, weight, decay)
    agent.x = i
    agent.y = j
    return agent

def change_grid_dict(grid, i, j, new_rollouts, new_constant, new_max_depth, new_weight, new_decay):
    grid[i][j]["rollouts"] = new_rollouts
    grid[i][j]["constant"] = new_constant
    grid[i][j]["max_depth"] = new_max_depth
    grid[i][j]["weight"] = new_weight
    grid[i][j]["decay"] = new_decay
    return grid

def make_grid(size, spawn_chance):
    grid = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() <= spawn_chance:
                # they are all the same right now
                grid[i][j] = make_agent_dict()
    return grid

def get_matchups(grid):
    # get every uniqie pair of populated cells that are touching
    num_rows = num_cols = len(grid)
    pairs = set()
    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] == 0:
                continue
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    if x == 0 and y == 0:
                        continue
                    ni, nj = i + x, j + y
                    if 0 <= ni < num_rows and 0 <= nj < num_rows and grid[ni][nj] != 0:
                        if (i, j) < (ni, nj):
                            pairs.add(((i, j), (ni, nj)))
    return list(pairs)

def run_comp():
    matchups = get_matchups(grid)
    
    for pair in matchups:
        agent1 = grid_to_agent(pair[0][0], pair[0][1])
        agent2 = grid_to_agent(pair[1][0], pair[1][1])
        game = PeggingGame(4)
        margin, wins = test_game(game, 1000, agent1.policy, agent2.policy)
        grid[pair[0][0]][pair[0][1]]["fitness"] += margin
        grid[pair[1][0]][pair[1][1]]["fitness"] -= margin


        
def run_elimination():
    # remove any cells that have fitness less than 0
    for i in range(len(grid)):
        for j in range(len(grid)):
            cell = grid[i][j]
            # only check "fitness" if this is still an agent dict
            if isinstance(cell, dict) and cell.get("fitness", 0) < 0:
                print("removing cell at", i, j)
                grid[i][j] = 0
    
def repopulate_grid(grid):
    # count number of alive cells
    num_alive = sum(1 for row in grid for cell in row if isinstance(cell, dict))
    print("num alive: ", num_alive)
    # if num_alive is less than min_cells, then repopulate the grid
    if num_alive < min_cells:
        need = min_cells - num_alive
        blanks = 100 - num_alive
        probability = need / blanks
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[i][j] == 0 and random.random() <= probability:
                    grid[i][j] = make_agent_dict()

def draw(grid, title=None):
    """Helper to update the existing figure with the current grid."""
    arr = np.array([
        [1 if isinstance(cell, dict) else 0 for cell in row]
        for row in grid
    ])
    
    # Clear the axis and redraw
    ax.clear()
    
    # Display the grid
    im = ax.imshow(arr, cmap='Greys', interpolation='none', origin='lower', vmin=0, vmax=1)
    
    # Set grid lines
    n = len(grid)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title if provided
    if title:
        ax.set_title(title)
        
    # Force redraw
    fig.canvas.draw()
    fig.canvas.flush_events()

def live_simulation(iterations=10, pause=0.5):
    plt.ion()
    global fig, ax, im
    fig, ax = plt.subplots(figsize=(6,6))
    # initialize with blank grid
    init_arr = np.zeros((len(grid), len(grid)))
    im = ax.imshow(init_arr, cmap='Greys', interpolation='none', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for gen in range(iterations):
        # 1) COMPETITION
        draw(grid, title=f"Gen {gen+1} – After Repopulation")
        
        run_comp()
        draw(grid, title=f"Gen {gen+1} – After Competition")
        
        # 2) ELIMINATION
        run_elimination()
        draw(grid, title=f"Gen {gen+1} – After Elimination")
        
        # 3) REPOPULATION
        repopulate_grid(grid)

    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # make your initial grid first
    grid = make_grid(10, min_cells/100)
    # then run the live sim
    live_simulation(iterations=20, pause=0.3)