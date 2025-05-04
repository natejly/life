# add feature that stores data from graph so we can see it 
from mcts import mcts_policy
from compare import test_game
from peg_game import PeggingGame
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from helpers import draw, get_average_parameters
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers import print_results, save_data_to_csv
import csv
import os


random.seed(420)
game = PeggingGame(4)
C = math.sqrt(2)
grid_size = 8
density = 0.5
n_games = 250 # 100
decay = 0 # 0.1
epsilon = 0.1
mutation_rate = 0.15
time_limit = 0.005
rollouts = 50 # 50
max_depth = 4 # 4 
child_fitness = 0 # 0.5
def make_agent_dict():
    default= {
        "time_limit": time_limit,
        "constant": C,
        "rollouts": rollouts,
        "max_depth": max_depth,
        "weight": 1,
        "decay": 1,
        "fitness": child_fitness
    }
    return default

def make_grid(size, spawn_chance):
    grid = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() <= spawn_chance:
                # they are all the same right now
                grid[i][j] = make_agent_dict()
    return grid

grid = make_grid(grid_size, density)

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


def grid_to_agent(i,j):
    d = grid[i][j]
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
    (i1, j1), (i2, j2), agent1_dict, agent2_dict = args
    agent1 = MCTSAGENT(**{k: agent1_dict[k] for k in agent1_dict if k != "fitness"})
    agent2 = MCTSAGENT(**{k: agent2_dict[k] for k in agent2_dict if k != "fitness"})
    margin, _ = test_game(game, n_games, agent1.policy, agent2.policy)
    return i1, j1, i2, j2, margin


def run_comp(max_workers=None):
    matchups = get_matchups(grid)

    # Build argument list with agent data
    args = []
    for (i1, j1), (i2, j2) in matchups:
        if not (isinstance(grid[i1][j1], dict) and isinstance(grid[i2][j2], dict)):
            continue  
        args.append(((i1, j1), (i2, j2), grid[i1][j1], grid[i2][j2]))

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_eval_matchup, arg): arg for arg in args}
        for fut in as_completed(futures):
            i1, j1, i2, j2, margin = fut.result()
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
                # print("removing cell at", i, j)
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
            parent = max(neighbours, key=lambda d: d.get("fitness", 0))

        # copy parameters with mutation
        child = {
            "time_limit": parent["time_limit"],
            "constant": parent["constant"] * (1 + random.normalvariate(0, mutation_rate)),
            "rollouts": max(1, int(parent["rollouts"] * (1 + random.normalvariate(0, mutation_rate)))),
            "max_depth": max(1, int(parent["max_depth"] * (1 + random.normalvariate(0, mutation_rate)))),
            "weight": parent["weight"] * (1 + random.normalvariate(0, mutation_rate)),
            "decay": parent["decay"] * (1 + random.normalvariate(0, mutation_rate)),
            "fitness": child_fitness
        }

        grid[i][j] = child
def _eval_agent_performance(args):
    agent_dict, default_agent_dict = args
    # Create a temporary agent from the dict
    agent = MCTSAGENT(**{k: agent_dict[k] for k in agent_dict if k != "fitness"})
    default_agent = MCTSAGENT(**default_agent_dict)
    
    # Run the test game
    margin, winrate = test_game(game, n_games, agent.policy, default_agent.policy)
    current_winrate = winrate * 100
    
    return agent_dict, current_winrate


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


    def parallel_agent_testing(current_agents, default_agent):
        # Convert default agent to a dictionary for serialization
        default_agent_dict = {
            "time_limit": default_agent.time_limit,
            "constant": default_agent.constant,
            "rollouts": default_agent.rollouts,
            "max_depth": default_agent.max_depth,
            "weight": default_agent.weight,
            "decay": default_agent.decay
        }
        
        args = [(agent_dict, default_agent_dict) for agent_dict in current_agents]
        
        all_winrates = []
        best_winrate = 0
        best_agent = None
        
        with ProcessPoolExecutor(max_workers=None) as exe:
            futures1 = {exe.submit(_eval_agent_performance, arg): arg for arg in args}
            for fut1 in as_completed(futures1):
                agent_dict, current_winrate = fut1.result()
                all_winrates.append(current_winrate)
                
                # Track best performing agent
                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    best_agent = agent_dict
        
        return all_winrates, best_winrate, best_agent

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
            fittest_agent = max(current_agents, key=lambda x: x.get("fitness", 0))
            
            # Parallel testing of all agents
            all_winrates, best_winrate, best_agent = parallel_agent_testing(current_agents, default_agent)
            
            # Test fittest agent (by fitness score) with more games
            fittest_agent_obj = MCTSAGENT(**{k: fittest_agent[k] for k in fittest_agent if k != "fitness"})
            _, fittest_winrate = test_game(game, n_games, fittest_agent_obj.policy, default_agent.policy)
            
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
            ax_compare.set_ylim(40, 70)
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
        
        print_results(gen, avg_rollouts, avg_constants, avg_depths, avg_weights, avg_decays, all_agents_avg, fittest_agent_perf, best_performing_perf)
        
    save_data_to_csv("performance_vs_default.csv",
                     ["Generation", "Population_Avg", "Fittest_Agent", "Best_Performer"],
                     list(zip(generations, all_agents_avg, fittest_agent_perf, best_performing_perf)))

    save_data_to_csv("parameter_trends.csv",
                     ["Generation", "Rollouts", "Constant", "Max_Depth", "Weight", "Decay"],
                     list(zip(generations, avg_rollouts, avg_constants, avg_depths, avg_weights, avg_decays)))

    plt.ioff()
    plt.show()
if __name__ == "__main__":
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