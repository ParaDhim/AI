import numpy as np
import pickle

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

def depth_limited_search(adj_matrix, start_node, goal_node, limit, visited):
    if start_node == goal_node:
        return [start_node]
    if limit <= 0:
        return None
    visited[start_node] = True
    for neighbor, cost in enumerate(adj_matrix[start_node]):
        if cost > 0 and not visited[neighbor]:
            path = depth_limited_search(adj_matrix, neighbor, goal_node, limit-1, visited)
            if path:
                return [start_node] + path
    visited[start_node] = False
    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    for limit in range(len(adj_matrix)):
        visited = [False] * len(adj_matrix)
        path = depth_limited_search(adj_matrix, start_node, goal_node, limit, visited)
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


from collections import deque

# def bfs_step(queue, visited, other_visited, parents):
#     current_node = queue.popleft()
#     for neighbor, cost in enumerate(adj_matrix[current_node]):
#         if cost > 0 and not visited[neighbor]:
#             parents[neighbor] = current_node
#             if neighbor in other_visited:
#                 return neighbor
#             visited[neighbor] = True
#             queue.append(neighbor)
#     return None

def bfs_step(queue, visited, other_visited, parents, adj_matrix):
    if not queue:
        return None

    current_node = queue.popleft()
    for neighbor, cost in enumerate(adj_matrix[current_node]):
        if cost > 0 and not visited.get(neighbor, False):  # Use get() to avoid KeyError
            parents[neighbor] = current_node

            if neighbor in other_visited:  # If it's visited by the other BFS, return meeting node
                return neighbor

            visited[neighbor] = True
            queue.append(neighbor)
    
    return None

# Reconstruct the path from start to goal via the meeting node
def reconstruct_path(parents_start, parents_goal, meeting_node):
    path = []

    # Path from start to meeting_node
    node = meeting_node
    while node is not None:
        path.append(node)
        node = parents_start.get(node, None)
    path.reverse()

    # Path from meeting_node to goal
    node = parents_goal.get(meeting_node, None)
    while node is not None:
        path.append(node)
        node = parents_goal.get(node, None)
    
    return path

# Function to get the path using bidirectional search
def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]

    visited_start = {start_node: True}
    visited_goal = {goal_node: True}
    parents_start = {start_node: None}
    parents_goal = {goal_node: None}

    queue_start = deque([start_node])
    queue_goal = deque([goal_node])

    while queue_start and queue_goal:
        # Expand the BFS from the start side
        meeting_node = bfs_step(queue_start, visited_start, visited_goal, parents_start, adj_matrix)
        if meeting_node is not None:
            return reconstruct_path(parents_start, parents_goal, meeting_node)

        # Expand the BFS from the goal side
        meeting_node = bfs_step(queue_goal, visited_goal, visited_start, parents_goal, adj_matrix)
        if meeting_node is not None:
            return reconstruct_path(parents_start, parents_goal, meeting_node)
    
    # If no path is found
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


import heapq

def heuristic(node1, node2, node_attributes):
    # print(f"Node1: {node1}, Node2: {node2}")
    if node1 not in node_attributes or node2 not in node_attributes:
        print(f"Node attributes missing for: {node1} or {node2}")
    x1, y1 = float(node_attributes[node1]['x']), float(node_attributes[node1]['y'])
    x2, y2 = float(node_attributes[node2]['x']), float(node_attributes[node2]['y'])
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5




def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    open_list = []
    heapq.heappush(open_list, (0, start_node))
    came_from = {start_node: None}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node, node_attributes)}
    
    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal_node:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for neighbor, cost in enumerate(adj_matrix[current]):
            if cost > 0:
                tentative_g_score = g_score[current] + cost
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node, node_attributes)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
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

def bidirectional_heuristic_step(queue, visited, other_visited, parents, adj_matrix, node_attributes, goal_node, forward):
    if not queue:
        return None
    
    current_priority, current_node = heapq.heappop(queue)
    row = adj_matrix[current_node]
    
    for neighbor, cost in enumerate(row):
        if cost > 0 and not visited.get(neighbor, False):
            parents[neighbor] = current_node
            if neighbor in other_visited:
                return neighbor
            visited[neighbor] = True
            if forward:
                priority = heuristic(neighbor, goal_node, node_attributes)
            else:
                if queue:  # Ensure queue is not empty
                    priority = heuristic(neighbor, queue[0][1], node_attributes)  # Access the node part of the tuple
                else:
                    priority = heuristic(neighbor, goal_node, node_attributes)  # Fallback if queue is empty
            heapq.heappush(queue, (priority, neighbor))
    return None

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]

    visited_start = {start_node: True}
    visited_goal = {goal_node: True}
    parents_start = {start_node: None}
    parents_goal = {goal_node: None}

    queue_start = [(0, start_node)]  # Initialize as list
    queue_goal = [(0, goal_node)]    # Initialize as list

    while queue_start and queue_goal:
        meeting_node = bidirectional_heuristic_step(queue_start, visited_start, visited_goal, parents_start, adj_matrix, node_attributes, goal_node, True)
        if meeting_node is not None:
            return reconstruct_path(parents_start, parents_goal, meeting_node)
        
        meeting_node = bidirectional_heuristic_step(queue_goal, visited_goal, visited_start, parents_goal, adj_matrix, node_attributes, start_node, False)
        if meeting_node is not None:
            return reconstruct_path(parents_start, parents_goal, meeting_node)
    
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


def bonus_problem(adj_matrix):
    def count_components(graph):
        visited = set()
        components = 0
        
        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor, cost in enumerate(graph[current]):
                    if cost > 0 and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        
        for node in range(len(graph)):
            if node not in visited:
                components += 1
                visited.add(node)
                dfs(node)
        return components
    
    initial_components = count_components(adj_matrix)
    vulnerable_edges = []
    
    for u in range(len(adj_matrix)):
        for v, cost in enumerate(adj_matrix[u]):
            if cost > 0:
                # Temporarily remove edge
                adj_matrix[u][v] = 0
                adj_matrix[v][u] = 0
                new_components = count_components(adj_matrix)
                if new_components > initial_components:
                    vulnerable_edges.append((u, v))
                # Restore edge
                adj_matrix[u][v] = cost
                adj_matrix[v][u] = cost
    
    return vulnerable_edges



# def bonus_problem(adj_matrix):

#   return []





############




############


class Solution:
    def __init__(self):
        self.timer = 1

    def dfs(self, node, parent, vis, adj, tin, low, bridges):
        vis[node] = 1
        tin[node] = low[node] = self.timer
        self.timer += 1
        for neighbor in adj[node]:
            if neighbor == parent:
                continue
            if vis[neighbor] == 0:
                self.dfs(neighbor, node, vis, adj, tin, low, bridges)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > tin[node]:
                    bridges.append([node, neighbor])
            else:
                low[node] = min(low[node], tin[neighbor])

    def criticalConnections(self, n, connections):
        adj = [[] for _ in range(n)]
        for u, v in connections:
            adj[u].append(v)
        
        vis = [0] * n
        tin = [0] * n
        low = [0] * n
        bridges = []
        self.dfs(0, -1, vis, adj, tin, low, bridges)
        return bridges

def bonus_problem1(adj_matrix):
    n = len(adj_matrix)
    connections = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] > 0:
                connections.append([i, j])
    
    solution = Solution()
    vulnerable_roads = solution.criticalConnections(n, connections)
    
    # For directed graphs, we need to check if removing each edge increases the number of components
    if not all(adj_matrix[j][i] > 0 for i, j in connections):  # Check if the graph is directed
        def count_components(graph):
            visited = set()
            components = 0
            def dfs(node):
                stack = [node]
                while stack:
                    current = stack.pop()
                    for neighbor, cost in enumerate(graph[current]):
                        if cost > 0 and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            for node in range(len(graph)):
                if node not in visited:
                    components += 1
                    visited.add(node)
                    dfs(node)
            return components
        
        initial_components = count_components(adj_matrix)
        vulnerable_roads = []
        for u, v in connections:
            # Temporarily remove edge
            original_cost = adj_matrix[u][v]
            adj_matrix[u][v] = 0
            new_components = count_components(adj_matrix)
            if new_components > initial_components:
                vulnerable_roads.append([u, v])
            # Restore edge
            adj_matrix[u][v] = original_cost

    return vulnerable_roads


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



# if __name__ == "__main__":
#     adj_matrix = np.load('/Users/parasdhiman/Desktop/code/AI Assmt/Assmt1/Assignment_1/IIIT_Delhi.npy')
#     with open('/Users/parasdhiman/Desktop/code/AI Assmt/Assmt1/Assignment_1/IIIT_Delhi.pkl', 'rb') as f:
#         node_attributes = pickle.load(f)
#     # Print available nodes
#     print(f'Total Nodes: {len(adj_matrix)}')
#     print(f'Nodes: {list(range(len(adj_matrix)))}')
#     start_node = int(input("Enter the start node: "))
#     end_node = int(input("Enter the end node: "))
#     # print(f'node_attributes keys: {list(node_attributes.keys())}')
#     # print(f'node_attributes values: {list(node_attributes.values())}')

#     # Example usage
#     # print_connections_with_arrows(adj_matrix)

#     # print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
#     # print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
#     # print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#     # print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#     print(f'Bonus Problem: {bonus_problem(adj_matrix)}')


# Example usage
if __name__ == "__main__":
    # Example adjacency matrix
    adj_matrix = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ]
    vulnerable_roads = bonus_problem(adj_matrix)
    print("Vulnerable roads:", vulnerable_roads)
    
