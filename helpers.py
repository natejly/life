
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
def draw(grid, fig, title=None, ax_grid=None):
    presence = np.zeros((len(grid), len(grid)))
    fitness = np.zeros((len(grid), len(grid)))
    
    all_fitness = [cell.get("fitness", 0) for row in grid for cell in row if isinstance(cell, dict)]
    max_fitness = max(all_fitness) if all_fitness else 1
    min_fitness = min(all_fitness) if all_fitness else 0
    
    min_darkness = 0.2  
    
    for i in range(len(grid)):
        for j in range(len(grid)):
            cell = grid[i][j]
            if isinstance(cell, dict):
                presence[i,j] = 1
                if max_fitness > min_fitness:  
                    normalized = (cell.get("fitness", 0) - min_fitness) / (max_fitness - min_fitness)
                    fitness[i,j] = min_darkness + (1 - min_darkness) * normalized
                else:
                    fitness[i,j] = min_darkness  
    
    ax_grid.clear()
    
    cmap = plt.cm.Greys_r  
    norm = plt.Normalize(vmin=0, vmax=1)
    
    colors = cmap(norm(fitness))
    colors[..., 3] = presence  
    
    im = ax_grid.imshow(colors, interpolation='none', origin='lower')
    
    n = len(grid)
    ax_grid.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_grid.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    
    if title:
        ax_grid.set_title(title)
        
    fig.canvas.draw()
    fig.canvas.flush_events()

def get_average_parameters(grid):
    agents = [cell for row in grid for cell in row if isinstance(cell, dict)]
    return {
        'rollouts': sum(agent['rollouts'] for agent in agents) / len(agents),
        'constant': sum(agent['constant'] for agent in agents) / len(agents),
        'max_depth': sum(agent['max_depth'] for agent in agents) / len(agents),
        'weight': sum(agent['weight'] for agent in agents) / len(agents),
        'decay': sum(agent['decay'] for agent in agents) / len(agents)
    }

def print_results(gen, avg_rollouts, avg_constants, avg_depths, avg_weights, avg_decays, all_agents_avg, fittest_agent_perf, best_performing_perf):
        print("Generation %d completed" % (gen + 1))
        print("Average Rollouts:", avg_rollouts[-1])
        print("Average Constant:", avg_constants[-1])
        print("Average Max Depth:", avg_depths[-1])
        print("Average Weight:", avg_weights[-1])
        print("Average Decay:", avg_decays[-1])
        print("Population Avg Performance:", all_agents_avg[-1])
        print("Fittest Agent Performance:", fittest_agent_perf[-1])
        print("Best Performing Agent Performance:", best_performing_perf[-1])
        
def save_data_to_csv(filename, headers, data_rows):
    os.makedirs("simulation_data", exist_ok=True)
    with open(os.path.join("simulation_data", filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data_rows)
