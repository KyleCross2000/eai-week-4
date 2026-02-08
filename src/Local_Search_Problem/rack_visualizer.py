"""
Visualization utilities for warehouse rack placement.
Provides functions to plot convergence curves and final rack layouts.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple
from rack_state import RackState
class RackVisualizer:
    """Handles visualization of rack placement solutions."""
    @staticmethod
    def plot_convergence(history_dict: Dict[str, List[float]],
                        figsize: Tuple[int, int] = (10, 6)):
        """
        Plot convergence curves for multiple algorithms.
        Args:
            history_dict: Dictionary mapping algorithm name to list of objective values.
            figsize: Figure size tuple (width, height).
        """
        plt.figure(figsize=figsize)
        for algorithm_name, history in history_dict.items():
            plt.plot(history, linewidth=2, label=algorithm_name, alpha=0.8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Objective Function Value', fontsize=12)
        plt.title('Convergence Curves')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    @staticmethod
    def plot_layout(state: RackState, title: str = "Rack Placement Layout",
                   figsize: Tuple[int, int] = (8, 8)):
        """
        Plot the warehouse layout with racks and depot.
        Args:
            state: RackState object to visualize.
            title: Plot title.
            figsize: Figure size tuple.
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Draw grid
        ax.set_xlim(-0.5, RackState.GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, RackState.GRID_SIZE - 0.5)
        ax.set_aspect('equal')
        # Grid lines
        for i in range(RackState.GRID_SIZE + 1):
            ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)
        # Draw depot
        depot_circle = patches.Circle(RackState.DEPOT, 0.3,
                                     color='red', zorder=10, label='Depot')
        ax.add_patch(depot_circle)
        # Draw congestion zone (distance < 5 from depot)
        for x in range(RackState.GRID_SIZE):
            for y in range(RackState.GRID_SIZE):
                dist = RackState.manhattan_distance(RackState.DEPOT, (x, y))
                if dist < RackState.CONGESTION_THRESHOLD and (x, y) != RackState.DEPOT:
                    rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                           linewidth=0, facecolor='yellow', alpha=0.2)
                    ax.add_patch(rect)
        # Draw racks
        for x, y in state.positions:
            rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                    linewidth=1, edgecolor='blue',
                                    facecolor='lightblue', zorder=5)
            ax.add_patch(rect)
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
        ax.set_title(f"{title}\nObjective: {state.objective_function():.2f}",
                    fontsize=12, fontweight='bold')
        ax.legend(['Depot', 'Racks', 'Congestion Zone'], loc='upper right', fontsize=10)
        plt.tight_layout()
        return fig
    @staticmethod
    def plot_comparison_layouts(solutions_dict: Dict[str, RackState],
                               figsize: Tuple[int, int] = (15, 5)):
        """
        Plot final layouts from multiple algorithms side by side.
        Args:
            solutions_dict: Dictionary mapping algorithm name to best RackState.
            figsize: Figure size tuple.
        """
        num_algorithms = len(solutions_dict)
        fig, axes = plt.subplots(1, num_algorithms, figsize=figsize)
        if num_algorithms == 1:
            axes = [axes]
        for ax, (algorithm_name, state) in zip(axes, solutions_dict.items()):
            # Draw grid
            ax.set_xlim(-0.5, RackState.GRID_SIZE - 0.5)
            ax.set_ylim(-0.5, RackState.GRID_SIZE - 0.5)
            ax.set_aspect('equal')
            # Grid lines
            for i in range(RackState.GRID_SIZE + 1):
                ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)
            # Draw depot
            depot_circle = patches.Circle(RackState.DEPOT, 0.3,
                                         color='red', zorder=10)
            ax.add_patch(depot_circle)
            # Draw congestion zone
            for x in range(RackState.GRID_SIZE):
                for y in range(RackState.GRID_SIZE):
                    dist = RackState.manhattan_distance(RackState.DEPOT, (x, y))
                    if dist < RackState.CONGESTION_THRESHOLD and (x, y) != RackState.DEPOT:
                        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                               linewidth=0, facecolor='yellow', alpha=0.2)
                        ax.add_patch(rect)
            # Draw racks
            for x, y in state.positions:
                rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                        linewidth=1, edgecolor='blue',
                                        facecolor='lightblue', zorder=5)
                ax.add_patch(rect)
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f"{algorithm_name}\nObj: {state.objective_function():.2f}",
                        fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducibility
    
    print("Testing RackVisualizer...")
    
    # Create sample states
    state1 = RackState()
    state2 = RackState()
    state3 = RackState()
    
    print(f"State 1 objective: {state1.objective_function():.2f}")
    print(f"State 2 objective: {state2.objective_function():.2f}")
    print(f"State 3 objective: {state3.objective_function():.2f}")
    
    # Test single layout plot
    print("\nPlotting single layout...")
    fig1 = RackVisualizer.plot_layout(state1, title="Test Layout")
    plt.savefig("test_layout.png", dpi=100)
    print("Saved: test_layout.png")
    
    # Test convergence plot with simulated history
    print("\nPlotting convergence curves...")
    history_dict = {
        "Algorithm A": [10.0 - i * 0.1 + random.random() * 0.5 for i in range(50)],
        "Algorithm B": [10.0 - i * 0.08 + random.random() * 0.3 for i in range(50)],
        "Algorithm C": [10.0 - i * 0.12 + random.random() * 0.4 for i in range(50)],
    }
    fig2 = RackVisualizer.plot_convergence(history_dict)
    plt.savefig("test_convergence.png", dpi=100)
    print("Saved: test_convergence.png")
    
    # Test comparison layouts
    print("\nPlotting comparison layouts...")
    solutions_dict = {
        "Random 1": state1,
        "Random 2": state2,
        "Random 3": state3,
    }
    fig3 = RackVisualizer.plot_comparison_layouts(solutions_dict)
    plt.savefig("test_comparison.png", dpi=100)
    print("Saved: test_comparison.png")
    
    print("\nAll tests completed! Check the generated PNG files.")
    plt.show()