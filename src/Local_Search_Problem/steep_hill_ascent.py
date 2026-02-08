"""
Steepest Ascent Hill Climbing for Warehouse Rack Optimization

This algorithm finds the best neighbor at each step and moves to it
if it improves the objective function. Since we're minimizing
(lower objective = better), we look for the neighbor with the lowest value.
"""

import sys
import os

# Add parent directory to path to import warehouse_racks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warehouse_racks import (
    create_initial_state,
    objective_function,
    get_neighbors,
    is_valid_state,
    print_state,
    print_state_info,
    NUM_RACKS,
    GRID_SIZE,
    DEPOT
)
from typing import List, Tuple, Optional


def steepest_ascent_hill_climbing(
    initial_state: Optional[List[Tuple[int, int]]] = None,
    max_iterations: int = 1000,
    verbose: bool = True
) -> Tuple[List[Tuple[int, int]], float, int]:
    """
    Steepest Ascent Hill Climbing algorithm for warehouse rack optimization.
    
    At each iteration, evaluates ALL neighbors and moves to the best one
    (lowest objective value) if it improves the current state.
    Terminates when no neighbor improves the current state (local minimum).
    
    Args:
        initial_state: Starting rack positions. If None, creates random initial state.
        max_iterations: Maximum number of iterations to prevent infinite loops.
        verbose: If True, prints progress information.
    
    Returns:
        Tuple of (best_state, best_objective_value, iterations_taken)
    """
    # Initialize state
    if initial_state is None:
        current_state = create_initial_state()
    else:
        current_state = initial_state.copy()
    
    current_value = objective_function(current_state)
    
    if verbose:
        print("=" * 60)
        print("STEEPEST ASCENT HILL CLIMBING")
        print("=" * 60)
        print(f"\nInitial objective value: {current_value:.4f}")
    
    iteration = 0
    improvements = []
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get all neighbors
        neighbors = get_neighbors(current_state)
        
        if not neighbors:
            if verbose:
                print(f"\nNo valid neighbors found at iteration {iteration}")
            break
        
        # Evaluate all neighbors and find the best one
        best_neighbor = None
        best_neighbor_value = float('inf')
        
        for neighbor in neighbors:
            neighbor_value = objective_function(neighbor)
            if neighbor_value < best_neighbor_value:
                best_neighbor_value = neighbor_value
                best_neighbor = neighbor
        
        # Check if best neighbor improves current state
        if best_neighbor_value < current_value:
            improvement = current_value - best_neighbor_value
            improvements.append(improvement)
            
            if verbose:
                print(f"Iteration {iteration}: {current_value:.4f} -> {best_neighbor_value:.4f} "
                      f"(improvement: {improvement:.4f})")
            
            current_state = best_neighbor
            current_value = best_neighbor_value
        else:
            # No improvement - reached local minimum
            if verbose:
                print(f"\nLocal minimum reached at iteration {iteration}")
                print(f"Best neighbor value ({best_neighbor_value:.4f}) >= "
                      f"current value ({current_value:.4f})")
            break
    
    if iteration >= max_iterations and verbose:
        print(f"\nMax iterations ({max_iterations}) reached")
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total iterations: {iteration}")
        print(f"Total improvements: {len(improvements)}")
        if improvements:
            print(f"Total improvement: {sum(improvements):.4f}")
            print(f"Average improvement per step: {sum(improvements)/len(improvements):.4f}")
    
    return current_state, current_value, iteration


def run_multiple_restarts(
    num_restarts: int = 10,
    max_iterations_per_run: int = 1000,
    verbose: bool = False
) -> Tuple[List[Tuple[int, int]], float, dict]:
    """
    Run steepest ascent hill climbing with multiple random restarts.
    
    Since hill climbing can get stuck in local minima, running multiple
    times with different starting points can find better solutions.
    
    Args:
        num_restarts: Number of times to run the algorithm.
        max_iterations_per_run: Max iterations for each run.
        verbose: If True, prints progress for each run.
    
    Returns:
        Tuple of (best_state, best_value, statistics_dict)
    """
    best_state = None
    best_value = float('inf')
    all_values = []
    all_iterations = []
    
    print(f"\nRunning {num_restarts} random restarts...")
    print("-" * 40)
    
    for restart in range(num_restarts):
        state, value, iterations = steepest_ascent_hill_climbing(
            initial_state=None,
            max_iterations=max_iterations_per_run,
            verbose=verbose
        )
        
        all_values.append(value)
        all_iterations.append(iterations)
        
        if value < best_value:
            best_value = value
            best_state = state
            print(f"Restart {restart + 1}/{num_restarts}: {value:.4f} (NEW BEST)")
        else:
            print(f"Restart {restart + 1}/{num_restarts}: {value:.4f}")
    
    stats = {
        'num_restarts': num_restarts,
        'best_value': best_value,
        'worst_value': max(all_values),
        'mean_value': sum(all_values) / len(all_values),
        'all_values': all_values,
        'all_iterations': all_iterations,
        'mean_iterations': sum(all_iterations) / len(all_iterations)
    }
    
    return best_state, best_value, stats


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducibility
    
    print("\n" + "=" * 60)
    print("WAREHOUSE RACK OPTIMIZATION")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Racks: {NUM_RACKS}, Depot: {DEPOT}")
    print("=" * 60)
    
    # Single run with verbose output
    print("\n--- Single Run (Verbose) ---")
    final_state, final_value, iterations = steepest_ascent_hill_climbing(
        verbose=True
    )
    
    print("\nFinal State Information:")
    print_state_info(final_state)
    
    print("\nFinal Grid Layout:")
    print_state(final_state)
    
    # Multiple restarts
    print("\n\n--- Multiple Restarts ---")
    best_state, best_value, stats = run_multiple_restarts(
        num_restarts=10,
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print("STATISTICS FROM MULTIPLE RESTARTS")
    print("=" * 60)
    print(f"Best objective value: {stats['best_value']:.4f}")
    print(f"Worst objective value: {stats['worst_value']:.4f}")
    print(f"Mean objective value: {stats['mean_value']:.4f}")
    print(f"Mean iterations per run: {stats['mean_iterations']:.1f}")
    
    print("\nBest Solution Found:")
    print_state_info(best_state)
    
    print("\nBest Grid Layout:")
    print_state(best_state)
