from mcts import mcts_policy
from compare import test_game
from peg_game import PeggingGame
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from helpers import draw, get_average_parameters
from concurrent.futures import ProcessPoolExecutor, as_completed

random.seed(100)
C = math.sqrt(2)
grid_size = 8
density = 0.5
n_games = 100
decay = 0.05
epsilon = 0.1
mutation_rate = 0.05
time_limit = 0.005
rollouts = 100
max_depth = 4
child_fitness = 0.0
class MCTSAGENT():
    def __init__(self, time_limit=None, constant=None, rollouts=None, max_depth=None, weight=None, decay=None):
        self.time_limit = time_limit
        self.constant = constant
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.weight = weight
        self.decay = decay

        self.policy_generator = mcts_policy(
            limit=self.time_limit,
            constant=self.constant, 
            rollouts=self.rollouts, 
            max_depth=self.max_depth, 
            weight=self.weight, 
            decay=self.decay
        )
    
    def policy(self):
        return self.policy_generator

def make_agent_dict(jitter=0.1):
    default= {
        "time_limit": time_limit,
        "constant": C,
        "rollouts": rollouts,
        "max_depth": max_depth,
        "weight": 1,
        "decay": 1,
        "fitness": child_fitness
    }
    agent = {}
    for key, val in default.items():
        if key == "fitness":
            agent[key] = val
        else:
            # relative jitter
            jitter = 1 + random.uniform(-jitter, jitter)
            jittered = val * jitter
            agent[key] = jittered
    return agent

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
    return agent

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

def _eval_matchup(args):
    """Worker: unpack coords, rebuild agents, run game, return fitness delta."""
    (i1, j1), (i2, j2) = args
    agent1 = grid_to_agent(i1, j1)
    agent2 = grid_to_agent(i2, j2)
    game = PeggingGame(4)
    margin, wins = test_game(game, 100, agent1.policy, agent2.policy)
    return i1, j1, i2, j2, margin

def run_comp(max_workers=None):
    matchups = get_matchups(grid)
    # Dispatch all matchups in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_eval_matchup, pair): pair for pair in matchups}
        for fut in as_completed(futures):
            i1, j1, i2, j2, margin = fut.result()
            # Main process updates the shared grid
            grid[i1][j1]["fitness"] += margin
            grid[i2][j2]["fitness"] -= margin

        
def run_elimination():
    # decay 
    for i in range(len(grid)):
        for j in range(len(grid)):
            cell = grid[i][j]
            # only check "fitness" if this is still an agent dict
            if isinstance(cell, dict):
                grid[i][j]["fitness"] -= decay
    # remove any cells that have fitness less than 0
    
    for i in range(len(grid)):
        for j in range(len(grid)):
            cell = grid[i][j]
            # only check "fitness" if this is still an agent dict
            if isinstance(cell, dict) and cell.get("fitness", 0) < 0:
                print("removing cell at", i, j)
                grid[i][j] = 0
    

def repopulate_grid(grid, e=epsilon, mutation_rate=mutation_rate, best_performer=None):
    rows, cols = len(grid), len(grid[0])
    births = []

    # First pass: collect all dead cells with exactly 3 live neighbours
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                continue

            # gather live neighbours
            neighbours = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and isinstance(grid[ni][nj], dict):
                        neighbours.append(grid[ni][nj])

            if len(neighbours) == 3:
                births.append((i, j, neighbours))

    # Second pass: actually spawn
    for i, j, neighbours in births:
        if best_performer and random.random() >= e:
            # Use the best performing agent as parent when available
            parent = best_performer
        else:
            # Fall back to random or fittest neighbor
            if random.random() < e:
                parent = random.choice(neighbours)
            else:
                parent = max(neighbours, key=lambda d: d.get("fitness", 0))

        # copy parameters with mutation
        child = {
            "time_limit": parent["time_limit"],
            "constant": parent["constant"] * (1 + random.normalvariate(0, mutation_rate)),
            "rollouts": max(1, int(parent["rollouts"] * (1 + random.normalvariate(0, mutation_rate)))),
            "max_depth": max(1, int(parent["max_depth"] * (1 + random.normalvariate(0, mutation_rate)))),
            "weight": parent["weight"] * (1 + random.normalvariate(0, mutation_rate)),
            "decay": parent["decay"] * (1 + random.normalvariate(0, mutation_rate)),
            "fitness": 0.0
        }

        grid[i][j] = child

def live_simulation(iterations):
    plt.ion()
    global fig, axs
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    ax_grid = axs[0, 0]
    ax_rollouts = axs[0, 1]
    ax_constant = axs[1, 0]
    ax_depth = axs[1, 1]
    ax_weight_decay = axs[1, 2]
    ax_compare = axs[0, 2]

    generations = []
    avg_rollouts, avg_constants, avg_depths, avg_weights, avg_decays = [], [], [], [], []
    all_agents_avg = []
    fittest_agent_perf = []
    best_performing_perf = []

    default_agent = MCTSAGENT(time_limit=time_limit, constant=C, rollouts=rollouts, 
                             max_depth=max_depth, weight=1, decay=1)

    for gen in range(iterations):
        generations.append(gen + 1)
        
        # 1) Run competition and elimination
        draw(grid, fig, title=f"Gen {gen+1}", ax_grid=ax_grid)
        run_comp()
        run_elimination()
        
        # 2) Get parameter averages
        params = get_average_parameters(grid)
        avg_rollouts.append(params['rollouts'])
        avg_constants.append(params['constant'])
        avg_depths.append(params['max_depth'])
        avg_weights.append(params['weight'])
        avg_decays.append(params['decay'])

        # 3) Performance comparison testing
        current_agents = [cell for row in grid for cell in row if isinstance(cell, dict)]
        if current_agents:
            # Test ALL agents against default (with reduced games per agent)
            all_winrates = []
            best_winrate = 0
            best_agent = None
            fittest_agent = max(current_agents, key=lambda x: x.get("fitness", 0))
            
            for agent_dict in current_agents:
                agent = MCTSAGENT(**{k: agent_dict[k] for k in agent_dict if k != "fitness"})
                margin, winrate = test_game(
                    PeggingGame(4), 
                    n_games, 
                    agent.policy, 
                    default_agent.policy
                )
                current_winrate = winrate * 100
                all_winrates.append(current_winrate)
                
                # Track best performing agent
                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    best_agent = agent_dict
            
            # Test fittest agent (by fitness score) with more games
            fittest_agent_obj = MCTSAGENT(**{k: fittest_agent[k] for k in fittest_agent if k != "fitness"})
            _, fittest_winrate = test_game(
                PeggingGame(4), 
                n_games, 
                fittest_agent_obj.policy, 
                default_agent.policy
            )
            
            # Store results
            all_agents_avg.append(sum(all_winrates)/len(all_winrates))
            fittest_agent_perf.append(fittest_winrate * 100)
            best_performing_perf.append(best_winrate)
            
            # Update comparison plot
            ax_compare.clear()
            ax_compare.plot(generations, all_agents_avg, 'b-', label='Population Avg')
            ax_compare.plot(generations, fittest_agent_perf, 'r-', label='Fittest Agent')
            ax_compare.plot(generations, best_performing_perf, 'g-', label='Best Performer')
            ax_compare.axhline(50, color='k', linestyle='--', label='Baseline')
            ax_compare.set_title(
                f"Performance vs Default\n"
                f"Gen {gen+1}: "
                f"Avg={all_agents_avg[-1]:.1f}%, "
                f"Fittest={fittest_agent_perf[-1]:.1f}%, "
                f"Best={best_performing_perf[-1]:.1f}%"
            )
            ax_compare.set_ylim(20, 100)
            ax_compare.legend()
            ax_compare.grid(True)


        ax_rollouts.clear()
        ax_rollouts.plot(generations, avg_rollouts, 'b-')
        ax_rollouts.set_title('Average Rollouts per Generation')
        ax_rollouts.grid(True)
        
        ax_constant.clear()
        ax_constant.plot(generations, avg_constants, 'r-')
        ax_constant.set_title('Average UCT Constant per Generation')
        ax_constant.grid(True)
        
        ax_depth.clear()
        ax_depth.plot(generations, avg_depths, 'g-')
        ax_depth.set_title('Average Max Depth per Generation')
        ax_depth.grid(True)
        
        ax_weight_decay.clear()
        ax_weight_decay.plot(generations, avg_weights, 'm-', label='Weight')
        ax_weight_decay.plot(generations, avg_decays, 'c-', label='Decay')
        ax_weight_decay.set_title('Weight and Decay Parameters')
        ax_weight_decay.legend()
        ax_weight_decay.grid(True)

        repopulate_grid(grid, best_performer=best_agent)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    grid = make_grid(grid_size, density)
    live_simulation(iterations=100)
    max_fitness = 0
    max_agent = None
    for i in range(len(grid)):
        for j in range(len(grid)):
            cell = grid[i][j]
            if isinstance(cell, dict) and cell.get("fitness", 0) > max_fitness:
                max_fitness = cell["fitness"]
                max_agent = cell
    print("Highest fitness agent:", max_agent)