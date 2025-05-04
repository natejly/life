
import numpy as np
import matplotlib.pyplot as plt

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
    
    if not hasattr(ax_grid, 'cbar'):
        ax_grid.cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax_grid, 
            orientation='vertical'
        )
        ax_grid.cbar.set_label('Fitness')
    else:
        ax_grid.cbar.update_normal(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    
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


