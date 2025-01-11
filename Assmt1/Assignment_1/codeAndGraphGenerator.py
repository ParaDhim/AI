import numpy as np
import pickle
import time
import sys
from collections import deque
from queue import Queue
import time
import tracemalloc
import matplotlib.pyplot as plt

""" COMMON FUNCTIONS """

def merge_path(parents_start, parents_goal, meeting_node):
    path = []

    node = meeting_node
    while node is not None:
        path.append(node)
        node = parents_start.get(node, None)
    path.reverse()

    node = parents_goal.get(meeting_node, None)
    while node is not None:
        path.append(node)
        node = parents_goal.get(node, None)
    
    return path

def heuristic(node1, node2, node_attributes):
    if node1 not in node_attributes or node2 not in node_attributes:
        print(f"Node attributes missing for: {node1} or {node2}")
    x1, y1 = float(node_attributes[node1]['x']), float(node_attributes[node1]['y'])
    x2, y2 = float(node_attributes[node2]['x']), float(node_attributes[node2]['y'])
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

""" COMMON FUNCTIONS END"""


# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def dfs(adj_matrix, start_node, goal_node, limit, current_depth, visited):
    if start_node == goal_node:
        return [start_node]

    if current_depth >= limit:
        return None

    visited[start_node] = True

    for neighbor, weight in enumerate(adj_matrix[start_node]):
        if weight > 0 and not visited[neighbor]:
            path = dfs(adj_matrix, neighbor, goal_node, limit, current_depth + 1, visited)
            if path:
                return [start_node] + path

    visited[start_node] = False
    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    for limit in range(len(adj_matrix)):
        # print(limit)
        visited = [False] * len(adj_matrix)
        path = dfs(adj_matrix, start_node, goal_node, limit, 0, visited)
        if path:
            return path
    return None



# def get_ids_path(adj_matrix, start_node, goal_node):

#   return []


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def bfs(queue, visitedNodes, other_visitedNodes, parents, adj_matrix):
    if not queue:
        return None

    current_node = queue.popleft()
    for neighbor, cst in enumerate(adj_matrix[current_node]):
        if cst > 0 and not visitedNodes.get(neighbor, False):
            parents[neighbor] = current_node

            if neighbor in other_visitedNodes:
                return neighbor

            visitedNodes[neighbor] = True
            queue.append(neighbor)
    
    return None



def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]

    visitedNodes_start = {start_node: True}
    visitedNodes_goal = {goal_node: True}
    parents_start = {start_node: None}
    parents_goal = {goal_node: None}

    start_frontier = deque([start_node])
    goal_frontier = deque([goal_node])

    while start_frontier and goal_frontier:
        meeting_node = bfs(start_frontier, visitedNodes_start, visitedNodes_goal, parents_start, adj_matrix)
        if meeting_node is not None:
            return merge_path(parents_start, parents_goal, meeting_node)

        meeting_node = bfs(goal_frontier, visitedNodes_goal, visitedNodes_start, parents_goal, adj_matrix)
        if meeting_node is not None:
            return merge_path(parents_start, parents_goal, meeting_node)
    
    return None


# def get_bidirectional_search_path(adj_matrix, start_node, goal_node):

#   return []


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]







def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    open_list = [(0, start_node)]
    path_node_prev = {start_node: None}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node, node_attributes)}
    
    while open_list:
        open_list.sort(key=lambda x: x[0])
        _, current = open_list.pop(0)

        if current == goal_node:
            path = []
            while current is not None:
                path.append(current)
                current = path_node_prev[current]
            return path[::-1]
        
        for neighbor, cst in enumerate(adj_matrix[current]):
            if cst > 0:
                tentative_g_score = g_score[current] + cst
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    path_node_prev[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node, node_attributes)
                    
                    found = False
                    for i, (f, node) in enumerate(open_list):
                        if node == neighbor:
                            found = True
                            if f > f_score[neighbor]:
                                open_list[i] = (f_score[neighbor], neighbor)
                            break
                    
                    if not found:
                        open_list.append((f_score[neighbor], neighbor))
    
    return None




# def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):

#   return []


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def bidirectional_heuristic_step(queue, visitedNodes, other_visitedNodes, parents, adj_matrix, node_attributes, goal_node, forward):
    if not queue:
        return None
    
    queue.sort(key=lambda x: x[0])
    current_priority, current_node = queue.pop(0)
    
    row = adj_matrix[current_node]
    
    for neighbor, cst in enumerate(row):
        if cst > 0 and not visitedNodes.get(neighbor, False):
            parents[neighbor] = current_node
            if neighbor in other_visitedNodes:
                return neighbor
            visitedNodes[neighbor] = True
            if forward:
                priority = heuristic(neighbor, goal_node, node_attributes)
            else:
                if queue:
                    priority = heuristic(neighbor, queue[0][1], node_attributes)
                else:
                    priority = heuristic(neighbor, goal_node, node_attributes)
            
            queue.append((priority, neighbor))
    
    return None

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]

    visitedNodes_start = {start_node: True}
    visitedNodes_goal = {goal_node: True}
    parents_start = {start_node: None}
    parents_goal = {goal_node: None}

    start_frontier = [(0, start_node)]
    goal_frontier = [(0, goal_node)]

    while start_frontier and goal_frontier:
        meeting_node = bidirectional_heuristic_step(start_frontier, visitedNodes_start, visitedNodes_goal, parents_start, adj_matrix, node_attributes, goal_node, True)
        if meeting_node is not None:
            return merge_path(parents_start, parents_goal, meeting_node)
        
        meeting_node = bidirectional_heuristic_step(goal_frontier, visitedNodes_goal, visitedNodes_start, parents_goal, adj_matrix, node_attributes, start_node, False)
        if meeting_node is not None:
            return merge_path(parents_start, parents_goal, meeting_node)
    
    return None


# def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):

#   return []



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].


# def bonus_problem(adj_matrix):

#   return []


def bonus_problem(adj_matrix):
    def dfs(u):
        nonlocal timer
        visitedNodes[u] = True
        disc[u] = low[u] = timer
        timer += 1

        for v in range(len(adj_matrix)):
            if adj_matrix[u][v] == 0:
                continue
            if not visitedNodes[v]:
                parent[v] = u
                dfs(v)
                
                low[u] = min(low[u], low[v])
                
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    n = len(adj_matrix)
    visitedNodes = [False] * n
    disc = [float("Inf")] * n
    low = [float("Inf")] * n
    parent = [-1] * n
    bridges = []
    timer = 0

    for i in range(n):
        if not visitedNodes[i]:
            dfs(i)
    
    return bridges


# def print_path(start, end, parent_start, parent_end):
#     path = []
    
#     # Trace path from start to meeting point
#     node = end
#     while node != start:
#         path.append(node)
#         node = parent_start[node]
#     path.append(start)
#     path.reverse()  # Reversing to get the correct order from start to meeting point
    
#     # Trace path from meeting point to end
#     node = parent_end[end]
#     while node != -1:
#         path.append(node)
#         node = parent_end[node]
    
#     print("Path:", " -> ".join(map(str, path)))
    
    
# def print_connections_with_arrows(adj_matrix):
#     total_nodes = len(adj_matrix)
#     print(f"Total Nodes: {total_nodes}")
    
#     # Loop through the adjacency matrix
#     for i in range(total_nodes):
#         connected_nodes = []
#         for j in range(total_nodes):
#             if adj_matrix[i][j] > 0:  # If there's a connection
#                 connected_nodes.append(j)
        
#         # If there are connected nodes, print the arrow format
#         if connected_nodes:
#             print(f"Node {i} -> connected to these: {{{', '.join(map(str, connected_nodes))}}}")
#         else:
#             print(f"Node {i} -> connected to none")
# """ ########################################################################### """
# def measure_performance(graph, start, goal, search_function):
#     start_time = time.time()
#     path = search_function(graph, start, goal)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     memory_usage = sys.getsizeof(path)  # Rough estimate of memory usage
#     return path, execution_time, memory_usage

# def measure_performance_with_attributes(graph, start, goal, node_attributes, search_function):
#     start_time = time.time()
#     path = search_function(graph, node_attributes, start, goal)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     memory_usage = sys.getsizeof(path)  # Rough estimate of memory usage
#     return path, execution_time, memory_usage
# """ ########################################################################### """



def measure_performance(graph, start, goal, search_function):
    tracemalloc.start()
    
    start_time = time.time()
    
    path = search_function(graph, start, goal)
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    return path, execution_time, peak 

def measure_performance_with_attributes(graph, start, goal, node_attributes, search_function):
    tracemalloc.start()
    
    start_time = time.time()
    
    path = search_function(graph, node_attributes, start, goal)
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    return path, execution_time, peak

# Example of usage:
# ids_path, ids_time, ids_memory = measure_performance(graph, start_node, goal_node, get_ids_path)
# print(f'Memory Usage: {ids_memory} bytes')





def generate_scatter_plots(results):
    algorithms = list(results.keys())
    times = [results[algo]['time'] for algo in algorithms]
    memory_usages = [results[algo]['memory'] for algo in algorithms]
    csts = [results[algo]['cst'] for algo in algorithms]
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(algorithms, times, color='skyblue', s=100, edgecolor='black')
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 3, 2)
    plt.scatter(algorithms, memory_usages, color='lightgreen', s=100, edgecolor='black')
    plt.xlabel('Algorithm')
    plt.ylabel('Memory Usage (bytes)')
    plt.title('Memory Usage Comparison')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 3, 3)
    plt.scatter(algorithms, csts, color='salmon', s=100, edgecolor='black')
    plt.xlabel('Algorithm')
    plt.ylabel('cst of Travel')
    plt.title('cst of Travel (Optimality) Comparison')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig('/Users/parasdhiman/Desktop/assmt/AI/AI Assmt/Assmt1/Assignment_1/performance_comparison3.png', format='png')
    
    plt.show()

def calculate_cst_of_travel(path, adj_matrix):
    total_cst = 0
    for i in range(len(path) - 1):
        total_cst += adj_matrix[path[i]][path[i+1]]
    return total_cst


if __name__ == "__main__":
    adj_matrix = np.load('/Users/parasdhiman/Desktop/assmt/AI/AI Assmt/Assmt1/Assignment_1/IIIT_Delhi.npy')
    with open('/Users/parasdhiman/Desktop/assmt/AI/AI Assmt/Assmt1/Assignment_1/IIIT_Delhi.pkl', 'rb') as f:
        node_attributes = pickle.load(f)
    # Print available nodes
    # print(f'Total Nodes: {len(adj_matrix)}')
    # print(f'Nodes: {list(range(len(adj_matrix)))}')'
    # for mn in [1,7,6,27,9,2,6]: # Assuming this is the key you're interested in
    #     values = node_attributes[mn]
    #     print(f'node_attributes[{mn}] values: {values}')
    start_node = int(input("Enter the start node: "))
    end_node = int(input("Enter the end node: "))
    # print(f'node_attributes keys: {list(node_attributes.keys())}')
    # print(f'node_attributes values: {list(node_attributes.values())}')

    # Example usage
    # print_connections_with_arrows(adj_matrix)

    # print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
    # print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
    # print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
    # print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
    print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
    # vulnerable_roads = bonus_problem(adj_matrix)
    # print("Vulnerable roads:", vulnerable_roads)
    # Measure and print performance for each search function
    print()
    # ids_path, ids_time, ids_memory = measure_performance(adj_matrix, start_node, end_node, get_ids_path)
    # print(f'Iterative Deepening Search Path: {ids_path}')
    # print(f'Execution Time: {ids_time:.6f} seconds')
    # print(f'Memory Usage: {ids_memory} bytes')
    # print()

    # bidirectional_path, bidirectional_time, bidirectional_memory = measure_performance(adj_matrix, start_node, end_node, get_bidirectional_search_path)
    # print(f'Bidirectional Search Path: {bidirectional_path}')
    # print(f'Execution Time: {bidirectional_time:.6f} seconds')
    # print(f'Memory Usage: {bidirectional_memory} bytes')
    # print()

    # # Use the modified function for A* search
    # astar_path, astar_time, astar_memory = measure_performance_with_attributes(adj_matrix, start_node, end_node, node_attributes, get_astar_search_path)
    # print(f'A* Path: {astar_path}')
    # print(f'Execution Time: {astar_time:.6f} seconds')
    # print(f'Memory Usage: {astar_memory} bytes')
    # print()

    # bidirectional_heuristic_path, bidirectional_heuristic_time, bidirectional_heuristic_memory = measure_performance_with_attributes(adj_matrix, start_node, end_node, node_attributes, get_bidirectional_heuristic_search_path)
    # print(f'Bidirectional Heuristic Search Path: {bidirectional_heuristic_path}')
    # print(f'Execution Time: {bidirectional_heuristic_time:.6f} seconds')
    # print(f'Memory Usage: {bidirectional_heuristic_memory} bytes')
    # print()
    results = {}

    for algo in ['bidirectional', 'ids', 'astar', 'bidirectional_astar']:
        
        if algo == 'bidirectional':
            path, exec_time, mem_usage = measure_performance(adj_matrix, start_node, end_node, get_bidirectional_search_path)
        # elif algo == 'ids' and results['bidirectional'] is not None:
        #     path, exec_time, mem_usage = measure_performance(adj_matrix, start_node, end_node, get_ids_path)
        elif algo == 'ids':
            if results['bidirectional']['path'] is not None:
               path, exec_time, mem_usage = measure_performance(adj_matrix, start_node, end_node, get_ids_path)
            else:
                results['ids'] = {'path': None, 'time': None, 'memory': None, 'cst': None}
                continue
        elif algo == 'astar':
            path, exec_time, mem_usage = measure_performance_with_attributes(adj_matrix, start_node, end_node, node_attributes, get_astar_search_path)
        elif algo == 'bidirectional_astar':
            path, exec_time, mem_usage = measure_performance_with_attributes(adj_matrix, start_node, end_node, node_attributes, get_bidirectional_heuristic_search_path)

        # Calculate the cst of travel (e.g., based on path length or edge weights)
        if path is not None:
            cst_of_travel = calculate_cst_of_travel(path, adj_matrix)  # Define this function based on your criteria
        else:
            cst_of_travel = None
        
        # Store the results, including the cst
        results[algo] = {'path': path, 'time': exec_time, 'memory': mem_usage, 'cst': cst_of_travel}

    # Generate scatter plots
    for algo, metrics in results.items():
        if metrics['path'] is not None:
            print(f'{algo.capitalize()} Path: {metrics["path"]}')
            print(f'Execution Time: {metrics["time"]:.6f} seconds')
            print(f'Memory Usage: {metrics["memory"]} bytes')
            print(f'Cost of Travel: {metrics["cst"]}')
            print()
        else:
            if algo == 'ids' and metrics['path'] is None:
                print(f'{algo.capitalize()} did not find a path.')
                print(f'Execution Time: {metrics["time"]} seconds')
                print(f'Memory Usage: {metrics["memory"]} bytes')
                print()
            else:
                print(f'{algo.capitalize()} did not find a path.')
                print(f'Execution Time: {metrics["time"]:.6f} seconds')
                print(f'Memory Usage: {metrics["memory"]} bytes')
                print()
    generate_scatter_plots(results)
        
    


# # Example usage
# if __name__ == "__main__":
#     # Example adjacency matrix
#     adj_matrix = [
#         [0, 1, 1, 0, 0],
#         [1, 0, 0, 1, 0],
#         [1, 0, 0, 0, 1],
#         [0, 1, 0, 0, 1],
#         [0, 0, 1, 1, 0]
#     ]

#     vulnerable_roads = bonus_problem1(adj_matrix)
#     print("Vulnerable roads:", vulnerable_roads)
#     vulnerable_roads = bonus_problem(adj_matrix)
#     print("Vulnerable roads:", vulnerable_roads)
    

"""" ********************************************************************************************************************************** """
# if __name__ == "__main__":
#   adj_matrix = np.load('IIIT_Delhi.npy')
#   with open('IIIT_Delhi.pkl', 'rb') as f:
#     node_attributes = pickle.load(f)

#   start_node = int(input("Enter the start node: "))
#   end_node = int(input("Enter the end node: "))

#   print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
#   print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
#   print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#   print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#   print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
"""" ********************************************************************************************************************************** """