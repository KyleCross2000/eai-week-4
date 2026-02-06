"""
Uniform Cost Search (UCS) Pathfinder for Warehouse Environment.

Implements UCS to find optimal paths based on path cost (number of steps).
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
    """Node for the priority queue, ordered by path cost."""
    cost: int  # Path cost (number of steps)
    state: Tuple = field(compare=False)  # (row, col, has_item)
    path: List[str] = field(compare=False)  # Actions taken to reach this state


class UCSPathfinder:
    """
    Uniform Cost Search pathfinder for the warehouse environment.
    
    Uses a priority queue ordered by path cost (number of steps) to find
    the optimal path from start to goal (pickup item and deliver to dropoff).
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
        pickup_pos = self._find_tile("P")
        if pickup_pos and (row, col) == pickup_pos and not has_item:
            neighbors.append(("PICK", (row, col, True)))
        
        # Try DROP action at dropoff location
        dropoff_pos = self._find_tile("D")
        if dropoff_pos and (row, col) == dropoff_pos and has_item:
            neighbors.append(("DROP", (row, col, False)))
        
        return neighbors
    
    def search(self, start_pos: Optional[Tuple[int, int]] = None) -> SearchResult:
        """
        Perform Uniform Cost Search to find optimal path.
        
        The goal is to pick up an item at 'P' and deliver it to 'D'.
        
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
        
        # Find goal position (dropoff)
        dropoff_pos = self._find_tile("D")
        if dropoff_pos is None:
            return SearchResult(
                path=[],
                path_length=0,
                nodes_expanded=0,
                computation_time=time.perf_counter() - start_time,
                success=False,
                frontier_max_size=0,
                explored_count=0
            )
        
        # Priority queue: (cost, state, path)
        # Using PriorityNode for proper ordering
        frontier: List[PriorityNode] = []
        heapq.heappush(frontier, PriorityNode(cost=0, state=start_state, path=[]))
        
        # Track states in frontier for efficient lookup
        frontier_states: Dict[Tuple[int, int, bool], int] = {start_state: 0}
        
        # Explored set
        explored: Set[Tuple[int, int, bool]] = set()
        
        # Statistics
        nodes_expanded = 0
        frontier_max_size = 1
        
        while frontier:
            # Track maximum frontier size
            frontier_max_size = max(frontier_max_size, len(frontier))
            
            # Get node with lowest cost
            current = heapq.heappop(frontier)
            current_cost = current.cost
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
                new_cost = current_cost + 1  # Each action costs 1 step
                new_path = current_path + [action]
                
                # Check for goal immediately when DROP action is available
                # This avoids the state collision where (dropoff, False) was already explored
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
                
                # Check if this state is already in frontier with higher cost
                if neighbor_state in frontier_states:
                    if frontier_states[neighbor_state] <= new_cost:
                        continue  # Skip if existing path is better or equal
                
                # Add to frontier
                frontier_states[neighbor_state] = new_cost
                heapq.heappush(frontier, PriorityNode(cost=new_cost, state=neighbor_state, path=new_path))
        
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
        Perform UCS to find optimal path between two positions (ignoring items).
        
        Args:
            start_pos: Starting position (row, col).
            goal_pos: Goal position (row, col).
            
        Returns:
            SearchResult containing the path and statistics.
        """
        start_time = time.perf_counter()
        
        # Simplified state: just (row, col)
        start_state = start_pos
        
        # Priority queue
        frontier: List[Tuple[int, Tuple[int, int], List[str]]] = []
        heapq.heappush(frontier, (0, start_state, []))
        
        # Track states in frontier
        frontier_states: Dict[Tuple[int, int], int] = {start_state: 0}
        
        # Explored set
        explored: Set[Tuple[int, int]] = set()
        
        # Statistics
        nodes_expanded = 0
        frontier_max_size = 1
        
        while frontier:
            frontier_max_size = max(frontier_max_size, len(frontier))
            
            current_cost, current_state, current_path = heapq.heappop(frontier)
            
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
            
            # Expand movement neighbors only
            row, col = current_state
            for action in self.MOVE_ACTIONS:
                dr, dc = self.MOVE_DELTAS[action]
                new_row, new_col = row + dr, col + dc
                
                if self._is_wall(new_row, new_col):
                    continue
                
                neighbor_state = (new_row, new_col)
                
                if neighbor_state in explored:
                    continue
                
                new_cost = current_cost + 1
                new_path = current_path + [action]
                
                if neighbor_state in frontier_states:
                    if frontier_states[neighbor_state] <= new_cost:
                        continue
                
                frontier_states[neighbor_state] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor_state, new_path))
        
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


def main():
    """Demonstrate UCS pathfinding in the warehouse environment."""
    # Create environment
    env = WarehouseEnv()
    print("Warehouse Grid:")
    print(env.render_with_legend())
    print()
    
    # Create pathfinder
    pathfinder = UCSPathfinder(env)
    
    # Find optimal path to pickup and deliver item
    print("Running Uniform Cost Search...")
    print(f"Start position: {env.start_pos}")
    print(f"Pickup position: {pathfinder._find_tile('P')}")
    print(f"Dropoff position: {pathfinder._find_tile('D')}")
    print()
    
    result = pathfinder.search()
    
    print("=" * 50)
    print("SEARCH RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Path length: {result.path_length} steps")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Explored nodes: {result.explored_count}")
    print(f"Max frontier size: {result.frontier_max_size}")
    print(f"Computation time: {result.computation_time:.6f} seconds")
    print()
    
    if result.success:
        print("Optimal path:")
        print(" -> ".join(result.path))
        print()
        
        # Verify path by executing in environment
        print("Verifying path execution...")
        env.reset()
        total_reward = 0
        for i, action in enumerate(result.path):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Goal reached at step {i + 1}!")
                break
        
        print(f"Total reward: {total_reward:.2f}")
        print()
        print("Final state:")
        print(env.render())


if __name__ == "__main__":
    main()
