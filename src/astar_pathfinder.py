"""
A* Search Pathfinder for Warehouse Environment.

Implements A* search to find optimal paths using f(n) = g(n) + h(n),
where g(n) is the path cost and h(n) is the Manhattan distance heuristic.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from warehouse_env import WarehouseEnv


@dataclass
class SearchResult:
    """Container for search results and statistics."""
    path: List[str]  # Sequence of actions to reach the goal
    path_length: int  # Number of steps in the path
    nodes_expanded: int  # Number of nodes removed from frontier and explored
    computation_time: float  # Time taken in seconds
    success: bool  # Whether a path was found
    frontier_max_size: int  # Maximum size of the frontier during search
    explored_count: int  # Total nodes added to explored set


@dataclass(order=True)
class PriorityNode:
    """Node for the priority queue, ordered by f(n) = g(n) + h(n)."""
    f_cost: int  # f(n) = g(n) + h(n) - used for priority ordering
    g_cost: int = field(compare=False)  # g(n) - actual path cost
    state: Tuple = field(compare=False)  # (row, col, has_item)
    path: List[str] = field(compare=False)  # Actions taken to reach this state


class AStarPathfinder:
    """
    A* Search pathfinder for the warehouse environment.
    
    Uses a priority queue ordered by f(n) = g(n) + h(n) to find
    the optimal path from start to goal (pickup item and deliver to dropoff).
    
    The heuristic h(n) is the Manhattan distance to the current goal:
    h(n) = |x_n - x_goal| + |y_n - y_goal|
    """
    
    # Movement actions and their deltas
    MOVE_ACTIONS = ["N", "E", "S", "W"]
    MOVE_DELTAS = {
        "N": (-1, 0),
        "E": (0, 1),
        "S": (1, 0),
        "W": (0, -1),
    }
    
    def __init__(self, env: WarehouseEnv):
        """
        Initialize the pathfinder with a warehouse environment.
        
        Args:
            env: The warehouse environment to search in.
        """
        self.env = env
        self.grid = env.grid
        self.height = env.height
        self.width = env.width
        
        # Cache pickup and dropoff positions
        self.pickup_pos = self._find_tile("P")
        self.dropoff_pos = self._find_tile("D")
    
    def _is_wall(self, row: int, col: int) -> bool:
        """Check if a position is a wall or out of bounds."""
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return True
        return self.grid[row][col] == "#"
    
    def _find_tile(self, tile: str) -> Optional[Tuple[int, int]]:
        """Find the position of a specific tile in the grid."""
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == tile:
                    return (r, c)
        return None
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        h(n) = |x_n - x_goal| + |y_n - y_goal|
        
        Args:
            pos1: First position (row, col).
            pos2: Second position (row, col).
            
        Returns:
            Manhattan distance between the positions.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _heuristic(self, state: Tuple[int, int, bool]) -> int:
        """
        Calculate the heuristic value for a state.
        
        The heuristic estimates the remaining cost to reach the goal:
        - If robot doesn't have item: distance to pickup + distance from pickup to dropoff
        - If robot has item: distance to dropoff
        
        Args:
            state: Current state as (row, col, has_item).
            
        Returns:
            Estimated cost to goal (admissible heuristic).
        """
        row, col, has_item = state
        current_pos = (row, col)
        
        if has_item:
            # Need to go to dropoff
            return self._manhattan_distance(current_pos, self.dropoff_pos)
        else:
            # Need to go to pickup, then to dropoff
            dist_to_pickup = self._manhattan_distance(current_pos, self.pickup_pos)
            dist_pickup_to_dropoff = self._manhattan_distance(self.pickup_pos, self.dropoff_pos)
            return dist_to_pickup + dist_pickup_to_dropoff
    
    def _get_neighbors(self, state: Tuple[int, int, bool]) -> List[Tuple[str, Tuple[int, int, bool]]]:
        """
        Get valid neighboring states from the current state.
        
        Args:
            state: Current state as (row, col, has_item).
            
        Returns:
            List of (action, new_state) tuples for valid transitions.
        """
        row, col, has_item = state
        neighbors = []
        
        # Try movement actions
        for action in self.MOVE_ACTIONS:
            dr, dc = self.MOVE_DELTAS[action]
            new_row, new_col = row + dr, col + dc
            
            if not self._is_wall(new_row, new_col):
                neighbors.append((action, (new_row, new_col, has_item)))
        
        # Try PICK action at pickup location
        if self.pickup_pos and (row, col) == self.pickup_pos and not has_item:
            neighbors.append(("PICK", (row, col, True)))
        
        # Try DROP action at dropoff location
        if self.dropoff_pos and (row, col) == self.dropoff_pos and has_item:
            neighbors.append(("DROP", (row, col, False)))
        
        return neighbors
    
    def search(self, start_pos: Optional[Tuple[int, int]] = None) -> SearchResult:
        """
        Perform A* Search to find optimal path.
        
        The goal is to pick up an item at 'P' and deliver it to 'D'.
        Uses f(n) = g(n) + h(n) where h(n) is Manhattan distance.
        
        Args:
            start_pos: Starting position (row, col). If None, uses env's start_pos.
            
        Returns:
            SearchResult containing the path and statistics.
        """
        start_time = time.perf_counter()
        
        # Initialize start state
        if start_pos is None:
            start_pos = self.env.start_pos
        
        start_state = (start_pos[0], start_pos[1], False)  # (row, col, has_item)
        
        # Validate environment
        if self.dropoff_pos is None:
            return SearchResult(
                path=[],
                path_length=0,
                nodes_expanded=0,
                computation_time=time.perf_counter() - start_time,
                success=False,
                frontier_max_size=0,
                explored_count=0
            )
        
        # Calculate initial heuristic
        h_start = self._heuristic(start_state)
        
        # Priority queue: ordered by f(n) = g(n) + h(n)
        frontier: List[PriorityNode] = []
        heapq.heappush(frontier, PriorityNode(
            f_cost=h_start,  # f = g + h = 0 + h
            g_cost=0,
            state=start_state,
            path=[]
        ))
        
        # Track states in frontier for efficient lookup (state -> g_cost)
        frontier_states: Dict[Tuple[int, int, bool], int] = {start_state: 0}
        
        # Explored set
        explored: Set[Tuple[int, int, bool]] = set()
        
        # Statistics
        nodes_expanded = 0
        frontier_max_size = 1
        
        while frontier:
            # Track maximum frontier size
            frontier_max_size = max(frontier_max_size, len(frontier))
            
            # Get node with lowest f(n) = g(n) + h(n)
            current = heapq.heappop(frontier)
            current_g = current.g_cost
            current_state = current.state
            current_path = current.path
            
            # Remove from frontier tracking
            if current_state in frontier_states:
                del frontier_states[current_state]
            
            # Skip if already explored
            if current_state in explored:
                continue
            
            # Mark as explored
            explored.add(current_state)
            nodes_expanded += 1
            
            # Expand neighbors
            for action, neighbor_state in self._get_neighbors(current_state):
                new_g = current_g + 1  # Each action costs 1 step
                new_path = current_path + [action]
                
                # Check for goal immediately when DROP action is available
                if action == "DROP":
                    computation_time = time.perf_counter() - start_time
                    return SearchResult(
                        path=new_path,
                        path_length=len(new_path),
                        nodes_expanded=nodes_expanded,
                        computation_time=computation_time,
                        success=True,
                        frontier_max_size=frontier_max_size,
                        explored_count=len(explored)
                    )
                
                if neighbor_state in explored:
                    continue
                
                # Check if this state is already in frontier with lower or equal g cost
                if neighbor_state in frontier_states:
                    if frontier_states[neighbor_state] <= new_g:
                        continue  # Skip if existing path is better or equal
                
                # Calculate f(n) = g(n) + h(n)
                h = self._heuristic(neighbor_state)
                f = new_g + h
                
                # Add to frontier
                frontier_states[neighbor_state] = new_g
                heapq.heappush(frontier, PriorityNode(
                    f_cost=f,
                    g_cost=new_g,
                    state=neighbor_state,
                    path=new_path
                ))
        
        # No path found
        computation_time = time.perf_counter() - start_time
        return SearchResult(
            path=[],
            path_length=0,
            nodes_expanded=nodes_expanded,
            computation_time=computation_time,
            success=False,
            frontier_max_size=frontier_max_size,
            explored_count=len(explored)
        )
    
    def search_to_position(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> SearchResult:
        """
        Perform A* to find optimal path between two positions (ignoring items).
        
        Args:
            start_pos: Starting position (row, col).
            goal_pos: Goal position (row, col).
            
        Returns:
            SearchResult containing the path and statistics.
        """
        start_time = time.perf_counter()
        
        # Simplified state: just (row, col)
        start_state = start_pos
        
        # Calculate initial heuristic
        h_start = self._manhattan_distance(start_pos, goal_pos)
        
        # Priority queue: (f_cost, g_cost, state, path)
        frontier: List[Tuple[int, int, Tuple[int, int], List[str]]] = []
        heapq.heappush(frontier, (h_start, 0, start_state, []))
        
        # Track states in frontier (state -> g_cost)
        frontier_states: Dict[Tuple[int, int], int] = {start_state: 0}
        
        # Explored set
        explored: Set[Tuple[int, int]] = set()
        
        # Statistics
        nodes_expanded = 0
        frontier_max_size = 1
        
        while frontier:
            frontier_max_size = max(frontier_max_size, len(frontier))
            
            # Get node with lowest f(n)
            f_cost, g_cost, current_state, current_path = heapq.heappop(frontier)
            
            # Remove from frontier tracking
            if current_state in frontier_states:
                del frontier_states[current_state]
            
            if current_state in explored:
                continue
            
            explored.add(current_state)
            nodes_expanded += 1
            
            # Check if goal reached
            if current_state == goal_pos:
                computation_time = time.perf_counter() - start_time
                return SearchResult(
                    path=current_path,
                    path_length=len(current_path),
                    nodes_expanded=nodes_expanded,
                    computation_time=computation_time,
                    success=True,
                    frontier_max_size=frontier_max_size,
                    explored_count=len(explored)
                )
            
            # Expand neighbors
            row, col = current_state
            for action in self.MOVE_ACTIONS:
                dr, dc = self.MOVE_DELTAS[action]
                new_row, new_col = row + dr, col + dc
                
                if self._is_wall(new_row, new_col):
                    continue
                
                neighbor_state = (new_row, new_col)
                
                if neighbor_state in explored:
                    continue
                
                new_g = g_cost + 1
                
                if neighbor_state in frontier_states:
                    if frontier_states[neighbor_state] <= new_g:
                        continue
                
                # Calculate f(n) = g(n) + h(n)
                h = self._manhattan_distance(neighbor_state, goal_pos)
                new_f = new_g + h
                
                frontier_states[neighbor_state] = new_g
                new_path = current_path + [action]
                heapq.heappush(frontier, (new_f, new_g, neighbor_state, new_path))
        
        # No path found
        computation_time = time.perf_counter() - start_time
        return SearchResult(
            path=[],
            path_length=0,
            nodes_expanded=nodes_expanded,
            computation_time=computation_time,
            success=False,
            frontier_max_size=frontier_max_size,
            explored_count=len(explored)
        )


# Demo / test code
if __name__ == "__main__":
    # Create environment
    env = WarehouseEnv()
    env.reset()
    
    print("Warehouse Layout:")
    print(env.render_with_legend())
    print()
    
    # Create A* pathfinder
    pathfinder = AStarPathfinder(env)
    
    # Run search
    print("Running A* Search...")
    result = pathfinder.search()
    
    print("\n=== A* Search Results ===")
    print(f"Success: {result.success}")
    print(f"Path: {' -> '.join(result.path)}")
    print(f"Path length: {result.path_length} steps")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Max frontier size: {result.frontier_max_size}")
    print(f"States explored: {result.explored_count}")
    print(f"Computation time: {result.computation_time * 1000:.3f} ms")
    
    # Compare with simple position-to-position search
    print("\n=== Position Search Test ===")
    start = (1, 1)
    goal = (5, 6)  # Dropoff position
    result2 = pathfinder.search_to_position(start, goal)
    print(f"Path from {start} to {goal}:")
    print(f"Success: {result2.success}")
    print(f"Path: {' -> '.join(result2.path)}")
    print(f"Path length: {result2.path_length} steps")
    print(f"Nodes expanded: {result2.nodes_expanded}")
