"""
Warehouse Rack Optimization using Local Search

State representation: list of 20 unique positions on a 20×20 grid
Depot location: (10, 10)
"""

import random
from typing import List, Tuple, Set

# Constants
GRID_SIZE = 20
NUM_RACKS = 20
DEPOT = (10, 10)
CONGESTION_RADIUS = 5
CONGESTION_PENALTY_MULTIPLIER = 2.0


# --- State Representation ---

def create_initial_state() -> List[Tuple[int, int]]:
    """
    Create an initial state with 20 unique rack positions on a 20×20 grid.
    Positions are (x, y) tuples where x, y ∈ [0, 19].
    The depot at (10, 10) is excluded from possible rack positions.
    
    Returns:
        List of 20 unique (x, y) positions representing rack locations.
    """
    positions: Set[Tuple[int, int]] = set()
    
    while len(positions) < NUM_RACKS:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        pos = (x, y)
        # Ensure position is unique and not at the depot
        if pos != DEPOT and pos not in positions:
            positions.add(pos)
    
    return list(positions)


def is_valid_state(state: List[Tuple[int, int]]) -> bool:
    """
    Validate that a state meets all requirements:
    - Exactly 20 racks
    - All positions unique
    - All positions within grid bounds
    - No rack at depot position
    
    Returns:
        True if state is valid, False otherwise.
    """
    if len(state) != NUM_RACKS:
        return False
    
    positions_set = set(state)
    if len(positions_set) != NUM_RACKS:
        return False  # Duplicates exist
    
    for x, y in state:
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False  # Out of bounds
        if (x, y) == DEPOT:
            return False  # Rack at depot
    
    return True


# --- Objective Function ---

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def compute_average_manhattan_distance(state: List[Tuple[int, int]]) -> float:
    """
    Compute the average Manhattan distance from all racks to the depot.
    
    Returns:
        Average Manhattan distance as a float.
    """
    total_distance = sum(manhattan_distance(pos, DEPOT) for pos in state)
    return total_distance / len(state)


def compute_congestion_penalty(state: List[Tuple[int, int]]) -> float:
    """
    Compute congestion penalty based on racks within distance 5 of depot.
    
    Congestion penalty = (count of racks within distance 5 of depot) × 2.0
    
    Returns:
        Congestion penalty as a float.
    """
    congested_count = sum(
        1 for pos in state 
        if manhattan_distance(pos, DEPOT) <= CONGESTION_RADIUS
    )
    return congested_count * CONGESTION_PENALTY_MULTIPLIER


def objective_function(state: List[Tuple[int, int]]) -> float:
    """
    Compute the objective function value for a given state.
    
    Objective = average Manhattan distance + congestion penalty
    
    Lower values are better (minimization problem).
    
    Returns:
        Objective function value as a float.
    """
    avg_distance = compute_average_manhattan_distance(state)
    congestion = compute_congestion_penalty(state)
    return avg_distance + congestion


# --- Neighborhood Definition ---

def get_neighbors(state: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Generate all neighbors by moving one rack by ±1 in x or y.
    
    A neighbor is created by:
    - Selecting one rack
    - Moving it by +1 or -1 in either x or y direction
    - Ensuring the new position is valid (within bounds, unique, not depot)
    
    Returns:
        List of all valid neighbor states.
    """
    neighbors = []
    state_set = set(state)
    
    # Possible moves: up, down, left, right
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i, (x, y) in enumerate(state):
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # Check if new position is valid
            if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
                continue  # Out of bounds
            if new_pos == DEPOT:
                continue  # Cannot place rack at depot
            if new_pos in state_set:
                continue  # Position already occupied by another rack
            
            # Create new state with the rack moved
            new_state = state.copy()
            new_state[i] = new_pos
            neighbors.append(new_state)
    
    return neighbors


def get_random_neighbor(state: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Generate a single random neighbor by moving one rack by ±1 in x or y.
    
    Returns:
        A valid neighbor state, or the original state if no valid moves exist.
    """
    state_set = set(state)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Shuffle rack indices and moves to randomize selection
    indices = list(range(len(state)))
    random.shuffle(indices)
    
    for i in indices:
        x, y = state[i]
        random.shuffle(moves)
        
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # Check if new position is valid
            if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
                continue
            if new_pos == DEPOT:
                continue
            if new_pos in state_set:
                continue
            
            # Create and return new state
            new_state = state.copy()
            new_state[i] = new_pos
            return new_state
    
    # No valid moves found (unlikely but possible)
    return state


# --- Utility Functions ---

def print_state(state: List[Tuple[int, int]]) -> None:
    """Print the state as a grid visualization."""
    grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # Mark depot
    grid[DEPOT[1]][DEPOT[0]] = 'D'
    
    # Mark racks
    for x, y in state:
        grid[y][x] = 'R'
    
    print("  " + "".join(f"{i:2}" for i in range(GRID_SIZE)))
    for y in range(GRID_SIZE):
        print(f"{y:2} " + " ".join(grid[y]))


def print_state_info(state: List[Tuple[int, int]]) -> None:
    """Print detailed information about a state."""
    avg_dist = compute_average_manhattan_distance(state)
    congestion = compute_congestion_penalty(state)
    obj_value = objective_function(state)
    
    congested_count = int(congestion / CONGESTION_PENALTY_MULTIPLIER)
    
    print(f"Average Manhattan distance: {avg_dist:.2f}")
    print(f"Racks within distance {CONGESTION_RADIUS} of depot: {congested_count}")
    print(f"Congestion penalty: {congestion:.2f}")
    print(f"Objective function value: {obj_value:.2f}")


if __name__ == "__main__":
    # Demo: Create initial state and show info
    print("Creating initial random state...")
    state = create_initial_state()
    
    print("\nInitial State Grid:")
    print_state(state)
    
    print("\nState Information:")
    print_state_info(state)
    
    print(f"\nState valid: {is_valid_state(state)}")
    
    print(f"\nNumber of neighbors: {len(get_neighbors(state))}")
    
    # Show a random neighbor
    print("\nGenerating random neighbor...")
    neighbor = get_random_neighbor(state)
    print("\nNeighbor State Info:")
    print_state_info(neighbor)
