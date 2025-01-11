import matplotlib.pyplot as plt

# Data for the scatter plots
algorithms = ['A*', 'IDS', 'Bidirectional BFS', 'Bidirectional A*']

# Test Case 1
execution_time_1 = [0.003479, 0.001536, 0.000704, 0.000562]
memory_usage_1 = [3865, 3848, 1512, 2217]
cost_of_travel_1 = [218.158, 241.707, 241.707, 243.275]

# Test Case 2
execution_time_2 = [0.004754, 0.003884, 0.002396, 0.000689]
memory_usage_2 = [6288, 4017, 2561, 2336]
cost_of_travel_2 = [313.284, 318.773, 313.284, 318.773]

# Test Case 3
execution_time_3 = [0.024224, None, 0.002515, 0.001295]
memory_usage_3 = [21552, None, 1969, 2833]
cost_of_travel_3 = [None, None, None, None]

# Test Case 4
execution_time_4 = [0.004532, 0.028796, 0.005884, 0.001274]
memory_usage_4 = [6433, 7817, 3776, 3545]
cost_of_travel_4 = [711.540, 754.402, 754.402, 717.029]

# Combine data from all test cases
execution_times = [execution_time_1, execution_time_2, execution_time_3, execution_time_4]
memory_usages = [memory_usage_1, memory_usage_2, memory_usage_3, memory_usage_4]
cost_of_travels = [cost_of_travel_1, cost_of_travel_2, cost_of_travel_3, cost_of_travel_4]

# Plot Execution Time vs Cost of Travel for each test case
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Execution Time vs Cost of Travel', fontsize=16)

for i, (exec_times, costs) in enumerate(zip(execution_times, cost_of_travels)):
    ax = axs[i//2, i%2]
    for j, alg in enumerate(algorithms):
        if exec_times[j] is not None and costs[j] is not None:
            ax.scatter(exec_times[j], costs[j], label=alg, s=100)
    ax.set_title(f'Test Case {i+1}')
    ax.set_xlabel('Execution Time (s)')
    ax.set_ylabel('Cost of Travel')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot Memory Usage vs Cost of Travel for each test case
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Memory Usage vs Cost of Travel', fontsize=16)

for i, (mem_usages, costs) in enumerate(zip(memory_usages, cost_of_travels)):
    ax = axs[i//2, i%2]
    for j, alg in enumerate(algorithms):
        if mem_usages[j] is not None and costs[j] is not None:
            ax.scatter(mem_usages[j], costs[j], label=alg, s=100)
    ax.set_title(f'Test Case {i+1}')
    ax.set_xlabel('Memory Usage (bytes)')
    ax.set_ylabel('Cost of Travel')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
