import matplotlib.pyplot as plt
import csv

def load_csv_data(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader) 
        data = list(reader)
    transposed = list(zip(*data))
    return headers, [list(map(float, col)) for col in transposed]

perf_headers, perf_data = load_csv_data('simulation_data/performance_vs_default.csv')
param_headers, param_data = load_csv_data('simulation_data/parameter_trends.csv')

generations = perf_data[0]

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 2].plot(generations, perf_data[1], label='Population Avg', color='blue')
axs[0, 2].plot(generations, perf_data[2], label='Fittest Agent', color='red')
axs[0, 2].plot(generations, perf_data[3], label='Best Performer', color='green')
axs[0, 2].axhline(50, color='black', linestyle='--', label='Baseline')
axs[0, 2].set_title('Performance vs Default')
axs[0, 2].legend()
axs[0, 2].grid(True)

axs[0, 1].plot(generations, param_data[1], color='blue')
axs[0, 1].set_title('Average Rollouts per Generation')
axs[0, 1].grid(True)

axs[1, 0].plot(generations, param_data[2], color='red')
axs[1, 0].set_title('Average UCT Constant per Generation')
axs[1, 0].grid(True)

axs[1, 1].plot(generations, param_data[3], color='green')
axs[1, 1].set_title('Average Max Depth per Generation')
axs[1, 1].grid(True)

axs[1, 2].plot(generations, param_data[4], color='magenta', label='Weight')
axs[1, 2].plot(generations, param_data[5], color='cyan', label='Decay')
axs[1, 2].set_title('Weight and Decay Parameters')
axs[1, 2].legend()
axs[1, 2].grid(True)

axs[0, 0].axis('off')
axs[0, 0].set_title("Grid not recoverable from CSV")

plt.tight_layout()
plt.show()
