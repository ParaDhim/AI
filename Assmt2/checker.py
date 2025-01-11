# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque
import time
import psutil
import os

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping"""
    trip_to_route = defaultdict(list)
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']].append(row['route_id'])
        
    # Map route_id to a list of stops in order of their sequence"""
    route_to_stops = defaultdict(list)
    sorted_stop_times = df_stop_times.sort_values(['trip_id', 'stop_sequence'])
    sorted_stop_times.head
    for trip_id, stop_grp in sorted_stop_times.groupby('trip_id'):
        if trip_id in trip_to_route:
            route_id = trip_to_route[trip_id][0]
            stops = stop_grp['stop_id'].to_list()
            route_to_stops[route_id].extend(stops)
            
    # Ensure each route only has unique stops"""
    for route_id in route_to_stops:
        # Use dict.fromkeys() to preserve order while removing duplicates
        route_to_stops[route_id] = list(dict.fromkeys(route_to_stops[route_id]))
    
    # Count trips per stop"""
    stop_trip_count = dict(df_stop_times['stop_id'].value_counts())

    # Create fare rules for routes
    fare_rules = {}
    for i in range(len(df_fare_rules['route_id'])):
        route_id = df_fare_rules['route_id'][i]
        fare_id = df_fare_rules['fare_id'][i]

        if route_id not in fare_rules:
            fare_rules[route_id] = []

        fare_rules[route_id].append(fare_id)

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(
        df_fare_rules,
        df_fare_attributes,
        on='fare_id',
        how='left'
    )

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_counts = defaultdict(int)
    for trip_id, routes in trip_to_route.items():
        # Since we stored routes as a list in trip_to_route, take the first route
        route_id = routes[0]
        route_trip_counts[route_id] += 1
    
    res = sorted(route_trip_counts.items(), key = lambda ele: ele[1], reverse = True)
    
    # Return top 5 routes
    return res[:5]
    pass  # Implementation here

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    res = sorted(stop_trip_count.items(), key = lambda ele: ele[1], reverse = True)
    
    # Return top 5 routes
    return res[:5]
    pass  # Implementation here

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_routes = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_routes[stop_id].add(route_id)
            
    stop_counts = []
    for stop_id, routes in stop_routes.items():
        route_count = len(routes)
        stop_counts.append((stop_id, route_count))

    res = sorted(stop_counts, key = lambda ele: ele[1], reverse = True)
    
    return res[:5]
    pass  # Implementation here

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    connections = {}
    
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            current_stop = stops[i]
            next_stop = stops[i + 1]
            
            # same order for consistency
            stop_pair = (current_stop, next_stop) if current_stop < next_stop else (next_stop, current_stop)

            
            # Add the route to this pair's list of routes
            if stop_pair not in connections:
                connections[stop_pair] = []
            connections[stop_pair].append(route_id)
    
    result = []
    for stop_pair, routes in connections.items():
        if len(routes) == 1:
            result.append((stop_pair, routes[0]))
    
    return result
    pass  # Implementation here

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Step 1: Create a network graph
    grph = nx.Graph()
    
    # Step 2: Add edges (connections between stops) from each route
    edge_colors = []
    edge_labels = []
    
    unique_routes = list(route_to_stops.keys())
    color_palette = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_routes)))
    
    rt_color_map = {}
    for i in range (len(unique_routes)):
        route_id = unique_routes[i]
        r, g, b, _ = color_palette[i]
        
        color_string = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
        rt_color_map[route_id] = color_string
    
    for route_id, stops_on_route in route_to_stops.items():
        route_name = df_routes[df_routes['route_id'] == route_id]['route_long_name'].iloc[0]
        for i in range(len(stops_on_route) - 1):
            current_stop = stops_on_route[i]
            next_stop = stops_on_route[i + 1]
            
            grph.add_edge(current_stop, next_stop)
            
            edge_colors.append(rt_color_map[route_id])
            edge_labels.append(f"Route: {route_name}")
        
    # Step 3: Calculate state_positions for the graph
    stop_positions = nx.shell_layout(grph)
    
    # Step 4: Create edge trace
    edge_trace = []
    for i in range (len(grph.edges())):
        edge = list(grph.edges())[i]
        color = edge_colors[i]
        label = edge_labels[i]
        
        x0, y0 = stop_positions[edge[0]]
        x1, y1 = stop_positions[edge[1]]
        
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color=color),
                hoverinfo='text',
                text=label,
                mode='lines'
            )
        )
        
    # Step 5: Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in grph.nodes():
        x, y = stop_positions[node]
        node_x.append(x)
        node_y.append(y)
        # Get stop name for hover text
        stop_name = df_stops[df_stops['stop_id'] == node]['stop_name'].iloc[0]
        node_text.append(f"Stop: {stop_name}<br>ID: {node}")
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    # Step 6: Create the figure
    fig = go.Figure(data=edge_trace + [node_trace],
                   layout=go.Layout(
                       title='Transit Network Map',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    # Add a legend showing route colors
    for route_id in unique_routes:
        route_name = df_routes[df_routes['route_id'] == route_id]['route_long_name'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=route_name,
            line=dict(color=rt_color_map[route_id], width=2),
            showlegend=True
        ))
    
    # Step 7: Show the plot
    fig.show()
    pass  # Implementation here

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    
    dir_routes = []
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            # Check if start_stop comes before end_stop in the sequence
            if stops.index(start_stop) < stops.index(end_stop):
                dir_routes.append(route_id)
    return dir_routes
    
    pass  # Implementation here

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    
    # Query for routes that contain both start and end stops
    query_result = pyDatalog.ask(f'RouteHasStop(R, {start}) & RouteHasStop(R, {end})')
    
    # Process the results, assuming each answer contains a single route_id
    if query_result:
        return sorted(set(route_id[0] for route_id in query_result.answers))
    return []
    
    pass  # Implementation here

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # Clear any existing facts
    pyDatalog.clear()
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

    # Add facts about which routes contain which stops
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)

    # Define valid path rules for direct routes
    OptimalRoute(X, Y) <= (
        RouteHasStop(X, start_stop_id) &
        RouteHasStop(X, stop_id_to_include) &
        RouteHasStop(X, end_stop_id)
    )

    # Rule for paths with one transfer
    OptimalRoute(X, Y) <= (
        RouteHasStop(X, start_stop_id) &
        RouteHasStop(X, stop_id_to_include) &
        RouteHasStop(Y, stop_id_to_include) &
        RouteHasStop(Y, end_stop_id) &
        (X != Y)
    )

    # Query for valid paths
    results = OptimalRoute(X, Y)

    # Convert results to list of tuples
    paths = [(x, stop_id_to_include, y) for x, y in results 
             if x is not None and y is not None]

    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024
    execution_metrics = {
        'execution_time': end_time - start_time,
        'memory_usage': final_memory - initial_memory
    }
    return sorted(list(set(paths)))
    


# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    
     # Clear any existing facts
    pyDatalog.clear()
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Add facts about which routes contain which stops
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)

    # Define rules starting from the end stop
    OptimalRoute(X, Y) <= (
        RouteHasStop(X, end_stop_id) &
        RouteHasStop(X, stop_id_to_include) &
        RouteHasStop(X, start_stop_id)
    )

    # Rule for paths with one transfer (backward direction)
    OptimalRoute(X, Y) <= (
        RouteHasStop(X, end_stop_id) &
        RouteHasStop(X, stop_id_to_include) &
        RouteHasStop(Y, stop_id_to_include) &
        RouteHasStop(Y, start_stop_id) &
        (X != Y)
    )

    # Query for valid paths
    results = OptimalRoute(X, Y)

    # Convert results to list of tuples
    paths = [(x, stop_id_to_include, y) for x, y in results 
             if x is not None and y is not None]

    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024
    execution_metrics = {
        'execution_time': end_time - start_time,
        'memory_usage': final_memory - initial_memory
    }
    return sorted(list(set(paths)))
    
    
    pass  # Implementation here


pyDatalog.create_terms('route_stop, R, X, Start, Via, End, Stop, R1, R2')
pyDatalog.create_terms('can_board, can_transfer, reachable, path')
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Optimized PDDL-based route planning implementation using PyDatalog
    Returns list of tuples (route1, transfer_stop, route2) representing valid paths
    """
    # Initialize PyDatalog
    pyDatalog.clear()
    # Initialize PyDatalog globally
    

    # # Start timing and memory tracking
    # start_time = time.time()
    # process = psutil.Process(os.getpid())
    # initial_memory = process.memory_info().rss / 1024 / 1024

    

    # Pre-process route data for faster lookups
    stop_to_routes = defaultdict(set)
    route_stops = defaultdict(set)
    transfer_routes = set()

    # Build lookup dictionaries and assert facts
    try:
        for route_id, stops in route_to_stops.items():
            route_stops[route_id] = set(stops)
            if stop_id_to_include in stops:
                transfer_routes.add(route_id)
            for stop in stops:
                stop_to_routes[stop].add(route_id)
                # Assert facts using proper PyDatalog syntax
                + route_stop(route_id, stop)

    except Exception as e:
        print(f"Error during fact assertion: {str(e)}")
        return []

    # Early termination checks
    if not stop_to_routes[start_stop_id] or not stop_to_routes[end_stop_id]:
        return []

    # Define rules (without + operator)
    can_board(R, X) <= route_stop(R, X)

    reachable(R, Start, Via, End) <= (
        can_board(R, Start) &
        route_stop(R, Via) &
        route_stop(R, End)
    )

    can_transfer(R1, R2, Stop) <= (
        route_stop(R1, Stop) &
        route_stop(R2, Stop)
    )

    path(R1, Via, R2) <= (
        reachable(R1, start_stop_id, Via, stop_id_to_include) &
        reachable(R2, stop_id_to_include, Via, end_stop_id)
    )

    result_paths = set()

    # Query for direct routes (no transfers)
    if max_transfers >= 0:
        for route in stop_to_routes[start_stop_id] & stop_to_routes[end_stop_id] & transfer_routes:
            try:
                stops = list(route_to_stops[route])
                start_idx = stops.index(start_stop_id)
                via_idx = stops.index(stop_id_to_include)
                end_idx = stops.index(end_stop_id)
                
                # Inline validity check for direct routes
                if start_idx < via_idx < end_idx:
                    result_paths.add((route, stop_id_to_include, route))
                    # print(f"State: Direct route found - {route}")
            except (ValueError, KeyError):
                continue

    # Query for routes with one transfer
    if max_transfers >= 1:
        # Query using PyDatalog syntax
        solutions = path(R1, Via, R2)
        if solutions:
            for r1, via, r2 in solutions:
                try:
                    stops1 = list(route_to_stops[r1])
                    stops2 = list(route_to_stops[r2])
                    if r1 != r2:
                        # Inline validity checks for transfer routes
                        start_to_via_valid = stops1.index(start_stop_id) < stops1.index(stop_id_to_include)
                        via_to_end_valid = True  # Allow reverse direction after transfer
                        
                        if start_to_via_valid and via_to_end_valid:
                            result_paths.add((r1, stop_id_to_include, r2))
                            # print(f"State: Transfer route found - {r1} to {r2} at {stop_id_to_include}")
                except (ValueError, KeyError):
                    continue

    # # Calculate performance metrics
    # execution_time = time.time() - start_time
    # final_memory = process.memory_info().rss / 1024 / 1024
    # memory_used = final_memory - initial_memory
    # print(f"\nPerformance Metrics:")
    # print(f"Execution Time: {execution_time:.4f} seconds")
    # print(f"Memory Usage: {memory_used:.2f} MB")
    # print(f"Number of Steps: {len(result_paths)}")

    return sorted(list(result_paths))
    
    pass  # Implementation here

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    
     # First, let's find the fare column - it might be 'price', 'fare_amount', etc.
    fare_column = None
    possible_fare_columns = ['price', 'fare', 'cost', 'fare_amount', 'amount']
    
    for col in possible_fare_columns:
        if col in merged_fare_df.columns:
            fare_column = col
            break
    
    if fare_column is None:
        # If we can't find a fare column, return the original DataFrame
        return merged_fare_df
        
    return merged_fare_df[merged_fare_df[fare_column] <= initial_fare]
    
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    
    summary = {}
    # First, find the fare column
    fare_column = None
    possible_fare_columns = ['price', 'fare', 'cost', 'fare_amount', 'amount']
    
    for col in possible_fare_columns:
        if col in pruned_df.columns:
            fare_column = col
            break
    
    if fare_column is None:
        # If we can't find a fare column, use a default value
        default_fare = 1.0
        
    for _, row in pruned_df.iterrows():
        route_id = row['route_id']
        fare = row[fare_column] if fare_column else default_fare
        
        if route_id not in summary:
            summary[route_id] = {
                'min_price': fare,
                'stops': set(route_to_stops[route_id])
            }
        else:
            summary[route_id]['min_price'] = min(summary[route_id]['min_price'], fare)

    return summary
    
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    
    queue = deque([(start_stop_id, [], 0, 0)])  # (current_stop, path, transfers_used, total_fare)
    visited = set()
    optimal_route = None

    while queue:
        current_stop, path, transfers_used, total_fare = queue.popleft()

        if current_stop == end_stop_id:
            if optimal_route is None or len(path) < len(optimal_route):
                optimal_route = path
            continue
        
        if transfers_used > max_transfers:
            continue
            
        if (current_stop, transfers_used) in visited:
            continue
        visited.add((current_stop, transfers_used))

        for route_id, info in route_summary.items():
            if current_stop in info['stops']:
                new_fare = total_fare + info['min_price']
                if new_fare <= initial_fare:
                    for next_stop in info['stops']:
                        if next_stop != current_stop:
                            new_transfers = transfers_used + 1 if path and path[-1][0] != route_id else transfers_used
                            if new_transfers <= max_transfers:
                                queue.append((
                                    next_stop, 
                                    path + [(route_id, next_stop)], 
                                    new_transfers,
                                    new_fare
                                ))

    return optimal_route or []
    
    pass  # Implementation here

