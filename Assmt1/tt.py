from dataclasses import dataclass
from typing import List, Dict

# Node structure with estimated cost (heuristic) to the goal and path cost (arc values)
graph = {
    'S': {'A': (3, 2), 'B': (1, 1), 'C': (5, 8)},
    'A': {'D': (4, 4), 'G1': (10, 0)},
    'B': {'D': (4, 4), 'F': (2, 3), 'G3': (12, 0)},
    'C': {'G3': (11, 0)},
    'D': {'E': (2, 1), 'G2': (5, 0)},
    'E': {'G1': (2, 0)},
    'F': {'D': (1, 4)}
}

@dataclass
class Node:
    name: str
    g: int  # Cost from the start node
    h: int  # Estimated cost to the goal (heuristic)
    parent: 'Node' = None  # Reference to the parent node

    def get_f(self):
        return self.g + self.h

    def isgoal(self):
        return self.name == 'G1'

    def get_children(self):
        children = []
        for child, (path_cost, heuristic) in graph.get(self.name, {}).items():
            children.append(Node(child, self.g + path_cost, heuristic, self))
        return children

    def get_path(self) -> List[str]:
        path = []
        node = self
        while node:
            path.append(node.name)
            node = node.parent
        return path[::-1]  # Reverse the path to start from the initial node

class IDAStar:
    def __init__(self, init_state: Node):
        self.thresh = init_state.get_f()  # Initial threshold is the f-value of the start node
        self.next_thresh = float('inf')  # Used to store the min value that exceeds the current threshold
        self.init_state = init_state

    def update_threshold(self):
        self.thresh = self.next_thresh
        self.next_thresh = float('inf')

    def solve(self):
        while True:
            success = self._solve(self.init_state)
            if success:
                return True
            self.update_threshold()

    def _solve(self, state: Node) -> bool:
        if state.isgoal():
            print(f"Reached goal: {state.name} with total cost: {state.g}")
            print("Path:", " -> ".join(state.get_path()))
            return True

        for child in state.get_children():
            f_value = child.get_f()
            if f_value <= self.thresh:
                if self._solve(child):
                    return True
            else:
                self.next_thresh = min(self.next_thresh, f_value)

        return False

# Start node S with initial g=0 and h=8 (estimated cost to goal)
start_state = Node(name='S', g=0, h=8)

# Initialize and solve using IDA* to reach G1
ida_star_solver = IDAStar(start_state)
ida_star_solver.solve()