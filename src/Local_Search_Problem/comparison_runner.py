import random
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from rack_state import RackState
from search_algorithms import HillClimber, SimulatedAnnealing, GeneticAlgorithm
from rack_visualizer import RackVisualizer

def run_comparison(num_runs: int = 20, seed: int = 42) -> Dict:
    """
    Run all algorithms on multiple random initial states.
    Args:
        num_runs: Number of random initial states to test.
        seed: Random seed for reproducibility.
    Returns:
        Dictionary with results from all algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    results = {
        'hill_climbing': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
        'simulated_annealing': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
        'genetic_algorithm': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
    }
    print(f"Running {num_runs} trials with 3 algorithms...")
    print("-" * 70)
    for trial in range(num_runs):
        # Generate random initial state
        initial_state = RackState()
        # Hill Climbing
        hc = HillClimber(initial_state.copy(), max_iterations=200)
        hc_best, hc_history = hc.optimize()
        results['hill_climbing']['best_values'].append(hc_best.objective_function())
        results['hill_climbing']['convergence_histories'].append(hc_history)
        if (results['hill_climbing']['best_state'] is None or
            hc_best.objective_function() <
            results['hill_climbing']['best_state'].objective_function()):
            results['hill_climbing']['best_state'] = hc_best.copy()
        # Simulated Annealing
        sa = SimulatedAnnealing(initial_state.copy(),
                               initial_temp=250.0,
                               cooling_rate=0.995,
                               max_iterations=2200)
        sa_best, sa_history = sa.optimize()
        results['simulated_annealing']['best_values'].append(sa_best.objective_function())
        results['simulated_annealing']['convergence_histories'].append(sa_history)
        if (results['simulated_annealing']['best_state'] is None or
            sa_best.objective_function() <
            results['simulated_annealing']['best_state'].objective_function()):
            results['simulated_annealing']['best_state'] = sa_best.copy()
        # Genetic Algorithm
        ga = GeneticAlgorithm(population_size=30,
                            mutation_rate=0.2,
                            max_generations=60)
        ga_best, ga_history = ga.optimize()
        results['genetic_algorithm']['best_values'].append(ga_best.objective_function())
        results['genetic_algorithm']['convergence_histories'].append(ga_history)
        if (results['genetic_algorithm']['best_state'] is None or
            ga_best.objective_function() <
            results['genetic_algorithm']['best_state'].objective_function()):
            results['genetic_algorithm']['best_state'] = ga_best.copy()
        if (trial + 1) % 5 == 0:
            print(f"Completed {trial + 1}/{num_runs} trials")
    print("-" * 70)
    return results

def print_summary_statistics(results: Dict) -> None:
    """
    Print summary statistics for all algorithms.
    Args:
        results: Results dictionary from run_comparison.
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for algorithm_name, data in results.items():
        best_values = np.array(data['best_values'])
        print(f"\n{algorithm_name.replace('_', ' ').title()}:")
        print(f"  Mean objective value:    {best_values.mean():.4f}")
        print(f"  Std deviation:           {best_values.std():.4f}")
        print(f"  Min (best) objective:    {best_values.min():.4f}")
        print(f"  Max (worst) objective:   {best_values.max():.4f}")
        print(f"  Best solution found:     {data['best_state'].objective_function():.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Warehouse Rack Placement - Algorithm Comparison")
    print("=" * 70)
    
    # Run comparison across multiple trials
    results = run_comparison(num_runs=20, seed=42)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Determine the winner
    print("\n" + "=" * 70)
    print("ALGORITHM RANKING (by mean objective value)")
    print("=" * 70)
    rankings = []
    for algo_name, data in results.items():
        mean_val = np.mean(data['best_values'])
        best_val = data['best_state'].objective_function()
        rankings.append((algo_name.replace('_', ' ').title(), mean_val, best_val))
    
    rankings.sort(key=lambda x: x[1])  # Sort by mean value (lower is better)
    
    print(f"\n{'Rank':<6} {'Algorithm':<25} {'Mean Obj':>12} {'Best Obj':>12}")
    print("-" * 55)
    for rank, (name, mean_val, best_val) in enumerate(rankings, 1):
        print(f"{rank:<6} {name:<25} {mean_val:>12.4f} {best_val:>12.4f}")
    
    print(f"\nWinner: {rankings[0][0]}")
    
    # Plot convergence curves (average across all runs)
    print("\nGenerating convergence plot...")
    avg_histories = {}
    for algo_name, data in results.items():
        histories = data['convergence_histories']
        # Pad histories to same length for averaging
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            padded_h = h + [h[-1]] * (max_len - len(h))  # Pad with final value
            padded.append(padded_h)
        avg_history = np.mean(padded, axis=0).tolist()
        avg_histories[algo_name.replace('_', ' ').title()] = avg_history
    
    fig1 = RackVisualizer.plot_convergence(avg_histories)
    plt.savefig("comparison_convergence.png", dpi=150)
    print("Saved: comparison_convergence.png")
    
    # Plot best solutions from each algorithm
    print("Generating layout comparison plot...")
    solutions_dict = {
        algo_name.replace('_', ' ').title(): data['best_state']
        for algo_name, data in results.items()
    }
    fig2 = RackVisualizer.plot_comparison_layouts(solutions_dict)
    plt.savefig("comparison_layouts.png", dpi=150)
    print("Saved: comparison_layouts.png")
    
    # Show plots
    plt.show()
    
    print("\nComparison completed successfully!")
