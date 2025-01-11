
# def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
#     """Perform forward chaining to find optimal routes considering transfers.
    
#     Args:
#         start_stop_id (int): The starting stop ID.
#         end_stop_id (int): The ending stop ID.
#         stop_id_to_include (int): The stop ID where a transfer occurs.
#         max_transfers (int): The maximum number of transfers allowed.
        
#     Returns:
#         list: A list of tuples (route_id, via_stop, end_stop) representing valid paths
#     """
#     result_paths = set()
    
    # # Find routes containing the required stops
    # for route_id, stops in route_to_stops.items():
    #     try:
    #         # Check if all required stops are in this route
    #         if stop_id_to_include in stops and (start_stop_id in stops or end_stop_id in stops):
    #             via_idx = stops.index(stop_id_to_include)
    #             # print(route_id)
    #             # Case 1: Direct route containing all stops
    #             if start_stop_id in stops and end_stop_id in stops:
    #                 start_idx = stops.index(start_stop_id)
    #                 end_idx = stops.index(end_stop_id)
                    
    #                 # Check if the route is valid (stops are in correct order)
    #                 if min(start_idx, end_idx) <= via_idx <= max(start_idx, end_idx):
    #                     result_paths.add((route_id, stop_id_to_include, end_stop_id))
    #                     # print("yes")
                        
                
    #             # Case 2: Route contains via stop and either start or end
    #             elif max_transfers >= 1:
    #                 # Find connecting routes
    #                 for other_route, other_stops in route_to_stops.items():
    #                     # print(other_route)
    #                     if other_route != route_id:
    #                         # print(other_route)
    #                         if start_stop_id in stops and end_stop_id in other_stops:
    #                             # print(other_route)
    #                             if stop_id_to_include in other_stops:
                                    
    #                                 result_paths.add((route_id, stop_id_to_include, other_route))
    #                                 # print((route_id, stop_id_to_include, other_route))
    #                                 # print("yes2")
    #                                 # print(other_route)
    #                                 # print(end_stop_id)
    #                         elif start_stop_id in other_stops and end_stop_id in stops:
    #                             # print((other_route, stop_id_to_include, route_id,start_stop_id, end_stop_id))
    #                             # print(other_stops)
    #                             if stop_id_to_include in other_stops:
    #                                 # print(other_route)
    #                                 # print((other_route, stop_id_to_include, route_id,start_stop_id, end_stop_id))
    #                                 result_paths.add((other_route, stop_id_to_include, route_id))
    #                                 # print((other_route, stop_id_to_include, route_id))
    #                                 # print("yes3")
    #                                 # print(other_route)
    #     except ValueError:
    #         continue
    
    # # Convert set to list for return
    # return sorted(list(result_paths))







   
    # dir_routes = []
    # for route_id, stops in route_to_stops.items():
    #     start_indexes = [i for i, stop in enumerate(stops) if stop == start_stop]
    #     end_indexes = [i for i, stop in enumerate(stops) if stop == end_stop]
    #     for start_idx in start_indexes:
    #         for end_idx in end_indexes:
    #             if end_idx > start_idx:
    #                 dir_routes.append(route_id)
    #                 break
    #         if route_id in dir_routes:
    #             break
            
    # return sorted(dir_routes)
    
    
    
    
    
    
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
    result_paths = set()
    
    # Find routes containing the required stops
    for route_id, stops in route_to_stops.items():
        try:
            # Check if all required stops are in this route
            if stop_id_to_include in stops and (start_stop_id in stops or end_stop_id in stops):
                via_idx = stops.index(stop_id_to_include)
                # print(route_id)
                # Case 1: Direct route containing all stops
                if start_stop_id in stops and end_stop_id in stops:
                    start_idx = stops.index(start_stop_id)
                    end_idx = stops.index(end_stop_id)
                    
                    # Check if the route is valid (stops are in correct order)
                    if min(start_idx, end_idx) <= via_idx <= max(start_idx, end_idx):
                        result_paths.add((route_id, stop_id_to_include, end_stop_id))
                        # print("yes")
                        
                
                # Case 2: Route contains via stop and either start or end
                elif max_transfers >= 1:
                    # Find connecting routes
                    for other_route, other_stops in route_to_stops.items():
                        # print(other_route)
                        if other_route != route_id:
                            # print(other_route)
                            if start_stop_id in stops and end_stop_id in other_stops:
                                # print(other_route)
                                if stop_id_to_include in other_stops:
                                    
                                    result_paths.add((route_id, stop_id_to_include, other_route))
                                    # print((route_id, stop_id_to_include, other_route))
                                    # print("yes2")
                                    # print(other_route)
                                    # print(end_stop_id)
                            elif start_stop_id in other_stops and end_stop_id in stops:
                                # print((other_route, stop_id_to_include, route_id,start_stop_id, end_stop_id))
                                # print(other_stops)
                                if stop_id_to_include in other_stops:
                                    # print(other_route)
                                    # print((other_route, stop_id_to_include, route_id,start_stop_id, end_stop_id))
                                    result_paths.add((other_route, stop_id_to_include, route_id))
                                    # print((other_route, stop_id_to_include, route_id))
                                    # print("yes3")
                                    # print(other_route)
        except ValueError:
            continue
    
    # Convert set to list for return
    return sorted(list(result_paths))
    
    
    pass  # Implementation here

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
    
    result_paths = []
    seen_paths = set()  # To avoid duplicates while maintaining order
    
    # Start from the end stop and work backwards
    for route_id, stops in route_to_stops.items():
        try:
            # Check if the route contains the end stop and via stop
            if end_stop_id in stops and stop_id_to_include in stops:
                end_idx = stops.index(end_stop_id)
                via_idx = stops.index(stop_id_to_include)
                
                # Case 1: Direct route containing all stops
                if start_stop_id in stops:
                    start_idx = stops.index(start_stop_id)
                    # Check if the route is valid (stops are in correct order)
                    if min(via_idx, end_idx) <= start_idx <= max(via_idx, end_idx):
                        path_tuple = (end_stop_id, stop_id_to_include, route_id)
                        path_key = str(path_tuple)  # Convert to string for hashing
                        # print("yes1")
                        # print(path_tuple)
                        if path_key not in seen_paths:
                            seen_paths.add(path_key)
                            result_paths.append(path_tuple)
                
                # Case 2: Route contains end stop and via stop, need to find connecting route
                elif max_transfers >= 1:
                    # Look for routes that can connect to our current route at the via stop
                    for connecting_route, connecting_stops in route_to_stops.items():
                        if connecting_route != route_id:
                            # Check if connecting route has start stop and via stop
                            if start_stop_id in connecting_stops and stop_id_to_include in connecting_stops:
                                conn_start_idx = connecting_stops.index(start_stop_id)
                                conn_via_idx = connecting_stops.index(stop_id_to_include)
                                
                                # Verify the order in connecting route
                                if min(conn_start_idx, conn_via_idx) <= max(conn_start_idx, conn_via_idx):
                                    path_tuple = (route_id, stop_id_to_include, connecting_route)
                                    # print("yes2")
                                    # print(path_tuple)
                                    path_key = str(path_tuple)
                                    if path_key not in seen_paths:
                                        seen_paths.add(path_key)
                                        result_paths.append(path_tuple)
                                        
            
            # Additional case: Route contains start stop and via stop
            elif start_stop_id in stops and stop_id_to_include in stops and max_transfers >= 1:
                start_idx = stops.index(start_stop_id)
                via_idx = stops.index(stop_id_to_include)
                
                # Look for routes that can connect from via stop to end stop
                for next_route, next_stops in route_to_stops.items():
                    if next_route != route_id:
                        if end_stop_id in next_stops and stop_id_to_include in next_stops:
                            next_end_idx = next_stops.index(end_stop_id)
                            next_via_idx = next_stops.index(stop_id_to_include)
                            
                            # Verify the order in next route
                            if min(next_via_idx, next_end_idx) <= max(next_via_idx, next_end_idx):
                                path_tuple = (next_route, stop_id_to_include, route_id)
                                # print("yes3")
                                # print(path_tuple)
                                path_key = str(path_tuple)
                                if path_key not in seen_paths:
                                    seen_paths.add(path_key)
                                    result_paths.append(path_tuple)
                                
        except ValueError:
            continue
    
    # Sort the results for consistent output
    return sorted(result_paths)
    
    
    pass  # Implementation here



def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Optimized PDDL-based route planning implementation using PyDatalog
    Returns list of tuples (route1, transfer_stop, route2) representing valid paths
    """
    # Initialize PyDatalog globally
    pyDatalog.create_terms('route_stop, R, X, Start, Via, End, Stop, R1, R2')
    pyDatalog.create_terms('can_board, can_transfer, reachable, path')

    # # Start timing and memory tracking
    # start_time = time.time()
    # process = psutil.Process(os.getpid())
    # initial_memory = process.memory_info().rss / 1024 / 1024

    # Initialize PyDatalog
    pyDatalog.clear()

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
                    print(f"State: Direct route found - {route}")
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
                            print(f"State: Transfer route found - {r1} to {r2} at {stop_id_to_include}")
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





# def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
#     """
#     Implement forward chaining using PyDatalog
#     """
#     # Clear any existing facts
#     pyDatalog.clear()
#     start_time = time.time()
#     process = psutil.Process(os.getpid())
#     initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

#     # Add facts about which routes contain which stops
#     for route_id, stops in route_to_stops.items():
#         for stop_id in stops:
#             +RouteHasStop(route_id, stop_id)

#     # Define valid path rules for direct routes
#     OptimalRoute(X, Y) <= (
#         RouteHasStop(X, start_stop_id) &
#         RouteHasStop(X, stop_id_to_include) &
#         RouteHasStop(X, end_stop_id)
#     )

#     # Rule for paths with one transfer
#     OptimalRoute(X, Y) <= (
#         RouteHasStop(X, start_stop_id) &
#         RouteHasStop(X, stop_id_to_include) &
#         RouteHasStop(Y, stop_id_to_include) &
#         RouteHasStop(Y, end_stop_id) &
#         (X != Y)
#     )

#     # Query for valid paths
#     results = OptimalRoute(X, Y)

#     # Convert results to list of tuples
#     paths = [(x, stop_id_to_include, y) for x, y in results 
#              if x is not None and y is not None]

#     end_time = time.time()
#     final_memory = process.memory_info().rss / 1024 / 1024
#     execution_metrics = {
#         'execution_time': end_time - start_time,
#         'memory_usage': final_memory - initial_memory
#     }
#     return sorted(list(set(paths)))

# def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
#     """
#     Implement backward chaining using PyDatalog
#     """
#     # Clear any existing facts
#     pyDatalog.clear()
#     start_time = time.time()
#     process = psutil.Process(os.getpid())
#     initial_memory = process.memory_info().rss / 1024 / 1024

#     # Add facts about which routes contain which stops
#     for route_id, stops in route_to_stops.items():
#         for stop_id in stops:
#             +RouteHasStop(route_id, stop_id)

#     # Define rules starting from the end stop
#     OptimalRoute(X, Y) <= (
#         RouteHasStop(X, end_stop_id) &
#         RouteHasStop(X, stop_id_to_include) &
#         RouteHasStop(X, start_stop_id)
#     )

#     # Rule for paths with one transfer (backward direction)
#     OptimalRoute(X, Y) <= (
#         RouteHasStop(X, end_stop_id) &
#         RouteHasStop(X, stop_id_to_include) &
#         RouteHasStop(Y, stop_id_to_include) &
#         RouteHasStop(Y, start_stop_id) &
#         (X != Y)
#     )

#     # Query for valid paths
#     results = OptimalRoute(X, Y)

#     # Convert results to list of tuples
#     paths = [(x, stop_id_to_include, y) for x, y in results 
#              if x is not None and y is not None]

#     end_time = time.time()
#     final_memory = process.memory_info().rss / 1024 / 1024
#     execution_metrics = {
#         'execution_time': end_time - start_time,
#         'memory_usage': final_memory - initial_memory
#     }
#     return sorted(list(set(paths)))



# def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
#     """
#     Implement forward chaining using PyDatalog
#     """
    
#     # Initialize PyDatalog
#     pyDatalog.create_terms('route_has_stop, connected_route, valid_path, start_stop, end_stop, via_stop, route, X, Y, Z')
    
#     # Clear any existing facts
#     pyDatalog.clear()
    
#     start_time = time.time()
#     process = psutil.Process(os.getpid())
#     initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
#     # Add facts about which routes contain which stops
#     for route_id, stops in route_to_stops.items():
#         for stop_id in stops:
#             +route_has_stop(route_id, stop_id)
    
#     # Define valid path rules for direct routes
#     valid_path(X, Y) <= (
#         route_has_stop(X, start_stop_id) &
#         route_has_stop(X, stop_id_to_include) &
#         route_has_stop(X, end_stop_id)
#     )
    
#     # Rule for paths with one transfer
#     valid_path(X, Y) <= (
#         route_has_stop(X, start_stop_id) &
#         route_has_stop(X, stop_id_to_include) &
#         route_has_stop(Y, stop_id_to_include) &
#         route_has_stop(Y, end_stop_id) &
#         (X != Y)
#     )
    
#     # Query for valid paths
#     results = valid_path(X, Y)
    
#     # Convert results to list of tuples
#     paths = [(x, stop_id_to_include, y) for x, y in results 
#              if x is not None and y is not None]
    
#     end_time = time.time()
#     final_memory = process.memory_info().rss / 1024 / 1024
    
#     execution_metrics = {
#         'execution_time': end_time - start_time,
#         'memory_usage': final_memory - initial_memory
#     }
    
#     return sorted(list(set(paths)))

# def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
#     """
#     Implement backward chaining using PyDatalog
#     """
    
#     # Initialize PyDatalog
#     pyDatalog.create_terms('route_has_stop, connected_route, valid_path, start_stop, end_stop, via_stop, route, X, Y, Z')
    
#     # Clear any existing facts
#     pyDatalog.clear()
    
#     start_time = time.time()
#     process = psutil.Process(os.getpid())
#     initial_memory = process.memory_info().rss / 1024 / 1024
    
#     # Add facts about which routes contain which stops
#     for route_id, stops in route_to_stops.items():
#         for stop_id in stops:
#             +route_has_stop(route_id, stop_id)
    
#     # Define rules starting from the end stop
#     valid_path(X, Y) <= (
#         route_has_stop(X, end_stop_id) &
#         route_has_stop(X, stop_id_to_include) &
#         route_has_stop(X, start_stop_id)
#     )
    
#     # Rule for paths with one transfer (backward direction)
#     valid_path(X, Y) <= (
#         route_has_stop(X, end_stop_id) &
#         route_has_stop(X, stop_id_to_include) &
#         route_has_stop(Y, stop_id_to_include) &
#         route_has_stop(Y, start_stop_id) &
#         (X != Y)
#     )
    
#     # Query for valid paths
#     results = valid_path(X, Y)
    
#     # Convert results to list of tuples
#     paths = [(x, stop_id_to_include, y) for x, y in results 
#              if x is not None and y is not None]
    
#     end_time = time.time()
#     final_memory = process.memory_info().rss / 1024 / 1024
    
#     execution_metrics = {
#         'execution_time': end_time - start_time,
#         'memory_usage': final_memory - initial_memory
#     }
    
#     return sorted(list(set(paths)))