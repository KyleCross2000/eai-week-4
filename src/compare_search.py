"""
Compare UCS and A* Search Algorithms on the Warehouse Environment.

This script runs both algorithms on 10 random configurations and compares:
- Path length (should be identical for optimal algorithms)
- Nodes expanded
- Maximum frontier size
- Computation time
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from astar_pathfinder import AStarPathfinder
from ucs_pathfinder import UCSPathfinder
from warehouse_env import WarehouseEnv


@dataclass
class TrialResult:
    """Results from a single trial comparing UCS and A*."""
    trial_num: int
    start_pos: Tuple[int, int]
    pickup_pos: Tuple[int, int]
    dropoff_pos: Tuple[int, int]
    
    # UCS metrics
    ucs_path_length: int
    ucs_nodes_expanded: int
    ucs_frontier_max_size: int
    ucs_time: float
    ucs_success: bool
    
    # A* metrics
    astar_path_length: int
    astar_nodes_expanded: int
    astar_frontier_max_size: int
    astar_time: float
    astar_success: bool
    
    # Validation
    paths_match: bool


def run_comparison(num_trials: int = 10, seed: int = 42) -> List[TrialResult]:
    """
    Run UCS and A* comparison on random warehouse configurations.
    
    Args:
        num_trials: Number of random configurations to test.
        seed: Random seed for reproducibility.
        
    Returns:
        List of TrialResult objects with comparison data.
    """
    random.seed(seed)
    results: List[TrialResult] = []
    
    for trial in range(num_trials):
        # Create fresh environment and randomize
        env = WarehouseEnv()
        obs = env.reset(randomize=True)
        
        # Get positions from observation
        start_pos = obs["robot_pos"]
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]
        
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*60}")
        print(f"Start: {start_pos}, Pickup: {pickup_pos}, Dropoff: {dropoff_pos}")
        print(env.render())
        
        # Create pathfinders with the same environment state
        ucs = UCSPathfinder(env)
        astar = AStarPathfinder(env)
        
        # Run UCS
        ucs_result = ucs.search(start_pos)
        
        # Run A* (need to recreate to reset cached positions)
        astar = AStarPathfinder(env)
        astar_result = astar.search(start_pos)
        
        # Verify paths have identical length (optimality check)
        paths_match = ucs_result.path_length == astar_result.path_length
        
        print(f"\nUCS:  Length={ucs_result.path_length}, Expanded={ucs_result.nodes_expanded}, "
              f"Frontier={ucs_result.frontier_max_size}, Time={ucs_result.computation_time*1000:.3f}ms")
        print(f"A*:   Length={astar_result.path_length}, Expanded={astar_result.nodes_expanded}, "
              f"Frontier={astar_result.frontier_max_size}, Time={astar_result.computation_time*1000:.3f}ms")
        
        if not paths_match:
            print(f"⚠️  WARNING: Path lengths differ! UCS={ucs_result.path_length}, A*={astar_result.path_length}")
        else:
            print(f"✓ Path lengths match (optimal)")
        
        results.append(TrialResult(
            trial_num=trial + 1,
            start_pos=start_pos,
            pickup_pos=pickup_pos,
            dropoff_pos=dropoff_pos,
            ucs_path_length=ucs_result.path_length,
            ucs_nodes_expanded=ucs_result.nodes_expanded,
            ucs_frontier_max_size=ucs_result.frontier_max_size,
            ucs_time=ucs_result.computation_time,
            ucs_success=ucs_result.success,
            astar_path_length=astar_result.path_length,
            astar_nodes_expanded=astar_result.nodes_expanded,
            astar_frontier_max_size=astar_result.frontier_max_size,
            astar_time=astar_result.computation_time,
            astar_success=astar_result.success,
            paths_match=paths_match,
        ))
    
    return results


def print_summary_table(results: List[TrialResult]) -> None:
    """Print a summary table of all trial results."""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    # Header
    header = (
        f"{'Trial':^6} | {'Start':^8} | {'Pickup':^8} | {'Dropoff':^8} | "
        f"{'UCS Len':^7} | {'UCS Exp':^7} | {'UCS Front':^9} | {'UCS Time':^10} | "
        f"{'A* Len':^6} | {'A* Exp':^6} | {'A* Front':^8} | {'A* Time':^10} | {'Match':^5}"
    )
    print(header)
    print("-" * len(header))
    
    # Data rows
    for r in results:
        row = (
            f"{r.trial_num:^6} | {str(r.start_pos):^8} | {str(r.pickup_pos):^8} | {str(r.dropoff_pos):^8} | "
            f"{r.ucs_path_length:^7} | {r.ucs_nodes_expanded:^7} | {r.ucs_frontier_max_size:^9} | "
            f"{r.ucs_time*1000:^10.3f} | "
            f"{r.astar_path_length:^6} | {r.astar_nodes_expanded:^6} | {r.astar_frontier_max_size:^8} | "
            f"{r.astar_time*1000:^10.3f} | {'✓' if r.paths_match else '✗':^5}"
        )
        print(row)
    
    # Statistics
    print("-" * len(header))
    
    ucs_expanded = [r.ucs_nodes_expanded for r in results]
    astar_expanded = [r.astar_nodes_expanded for r in results]
    ucs_frontier = [r.ucs_frontier_max_size for r in results]
    astar_frontier = [r.astar_frontier_max_size for r in results]
    ucs_times = [r.ucs_time * 1000 for r in results]
    astar_times = [r.astar_time * 1000 for r in results]
    
    print("\nSTATISTICS:")
    print("-" * 60)
    print(f"{'Metric':<25} | {'UCS':^15} | {'A*':^15} | {'Reduction':^12}")
    print("-" * 60)
    
    # Nodes expanded
    ucs_mean_exp = np.mean(ucs_expanded)
    astar_mean_exp = np.mean(astar_expanded)
    reduction_exp = ((ucs_mean_exp - astar_mean_exp) / ucs_mean_exp) * 100 if ucs_mean_exp > 0 else 0
    print(f"{'Mean Nodes Expanded':<25} | {ucs_mean_exp:^15.2f} | {astar_mean_exp:^15.2f} | {reduction_exp:^11.1f}%")
    
    ucs_std_exp = np.std(ucs_expanded)
    astar_std_exp = np.std(astar_expanded)
    print(f"{'Std Nodes Expanded':<25} | {ucs_std_exp:^15.2f} | {astar_std_exp:^15.2f} | {'-':^12}")
    
    # Frontier size
    ucs_mean_front = np.mean(ucs_frontier)
    astar_mean_front = np.mean(astar_frontier)
    reduction_front = ((ucs_mean_front - astar_mean_front) / ucs_mean_front) * 100 if ucs_mean_front > 0 else 0
    print(f"{'Mean Max Frontier':<25} | {ucs_mean_front:^15.2f} | {astar_mean_front:^15.2f} | {reduction_front:^11.1f}%")
    
    # Time
    ucs_mean_time = np.mean(ucs_times)
    astar_mean_time = np.mean(astar_times)
    reduction_time = ((ucs_mean_time - astar_mean_time) / ucs_mean_time) * 100 if ucs_mean_time > 0 else 0
    print(f"{'Mean Time (ms)':<25} | {ucs_mean_time:^15.4f} | {astar_mean_time:^15.4f} | {reduction_time:^11.1f}%")
    
    print("-" * 60)
    
    # Optimality check
    all_match = all(r.paths_match for r in results)
    all_success = all(r.ucs_success and r.astar_success for r in results)
    print(f"\nOptimality Check: {'PASSED ✓' if all_match else 'FAILED ✗'} - All paths have identical length")
    print(f"Success Rate: {sum(1 for r in results if r.ucs_success and r.astar_success)}/{len(results)} trials")


def create_visualizations(results: List[TrialResult], save_path: str = None) -> None:
    """
    Create visualizations comparing UCS and A* performance.
    
    Args:
        results: List of trial results.
        save_path: Optional path to save the figure.
    """
    trials = [r.trial_num for r in results]
    ucs_expanded = [r.ucs_nodes_expanded for r in results]
    astar_expanded = [r.astar_nodes_expanded for r in results]
    ucs_frontier = [r.ucs_frontier_max_size for r in results]
    astar_frontier = [r.astar_frontier_max_size for r in results]
    ucs_times = [r.ucs_time * 1000 for r in results]
    astar_times = [r.astar_time * 1000 for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('UCS vs A* Search Algorithm Comparison\n(10 Random Warehouse Configurations)', 
                 fontsize=14, fontweight='bold')
    
    # Bar chart: Nodes Expanded per Trial
    ax1 = axes[0, 0]
    x = np.arange(len(trials))
    width = 0.35
    bars1 = ax1.bar(x - width/2, ucs_expanded, width, label='UCS', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, astar_expanded, width, label='A*', color='coral', alpha=0.8)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Nodes Expanded per Trial')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trials)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar chart: Mean Nodes Expanded (with error bars)
    ax2 = axes[0, 1]
    means = [np.mean(ucs_expanded), np.mean(astar_expanded)]
    stds = [np.std(ucs_expanded), np.std(astar_expanded)]
    bars = ax2.bar(['UCS', 'A*'], means, yerr=stds, capsize=5, 
                   color=['steelblue', 'coral'], alpha=0.8)
    ax2.set_ylabel('Mean Nodes Expanded')
    ax2.set_title('Mean Nodes Expanded Across 10 Trials')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax2.annotate(f'{mean:.1f}±{std:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10)
    
    # Add reduction percentage
    reduction = ((means[0] - means[1]) / means[0]) * 100 if means[0] > 0 else 0
    ax2.annotate(f'A* reduces nodes by {reduction:.1f}%',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=10, color='green', fontweight='bold')
    
    # Bar chart: Max Frontier Size per Trial
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x - width/2, ucs_frontier, width, label='UCS', color='steelblue', alpha=0.8)
    bars4 = ax3.bar(x + width/2, astar_frontier, width, label='A*', color='coral', alpha=0.8)
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Max Frontier Size')
    ax3.set_title('Maximum Frontier Size per Trial')
    ax3.set_xticks(x)
    ax3.set_xticklabels(trials)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Bar chart: Computation Time per Trial
    ax4 = axes[1, 1]
    bars5 = ax4.bar(x - width/2, ucs_times, width, label='UCS', color='steelblue', alpha=0.8)
    bars6 = ax4.bar(x + width/2, astar_times, width, label='A*', color='coral', alpha=0.8)
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Computation Time per Trial')
    ax4.set_xticks(x)
    ax4.set_xticklabels(trials)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def main():
    """Run the comparison between UCS and A* search algorithms."""
    print("="*60)
    print("UCS vs A* Search Algorithm Comparison")
    print("="*60)
    print("\nRunning 10 trials with random warehouse configurations...")
    print("Each trial: Find path from Start → Pickup → Dropoff")
    
    # Run comparison
    results = run_comparison(num_trials=10, seed=42)
    
    # Print summary table
    print_summary_table(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, save_path="search_comparison.png")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()
