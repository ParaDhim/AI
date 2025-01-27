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
    route_to_stops = {
        route: [x for i, x in enumerate(stops) if x not in stops[:i]]
        for route, stops in route_to_stops.items()
    }
    
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
              - route_id (str): The ID of the route.
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
              - stop_id (str): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    pass  # Implementation here

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (str): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    pass  # Implementation here

# Function to find pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify stop pairs that are connected by exactly one direct route.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple containing two stop IDs (stop_1, stop_2).
              - route_id (str): The ID of the route connecting the two stops.
    """
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
    pass  # Implementation here

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (str): The ID of the starting stop.
        end_stop (str): The ID of the ending stop.

    Returns:
        list: A list of route IDs (str) that connect the two stops directly.
    """
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
    pass  # Implementation here

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (str): The ID of the starting stop.
        end (str): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    pass  # Implementation here

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
    pyDatalog.clear()
    pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')
    
    pass  # Implementation here

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
    pass  # Implementation here

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
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
                  route_id (str): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (str), stop_id (str)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here

# Example usage and testing
if __name__ == "__main__":
    # Make sure knowledge base is created first
    create_kb()
    
    # Get busiest routes
    top_routes = get_busiest_routes()
    
    print("Top 5 Busiest Routes:")
    print("Route ID | Trip Count")
    print("-" * 20)
    for route_id, trip_count in top_routes:
        print(f"{route_id:8} | {trip_count:>10}")
        
    # Additional verification
    print("\nVerification:")
    print(f"Total number of routes analyzed: {len(trip_to_route)}")
    print(f"Maximum trips for a single route: {max(dict(top_routes).values())}")