"""
Run and animate an A* pathfinding episode in the warehouse environment.

Uses AStarPathfinder to find the optimal path and warehouse_viz to animate it.
"""

from warehouse_env import WarehouseEnv
from astar_pathfinder import AStarPathfinder
from warehouse_viz import replay_animation, save_frames_to_svg


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int] | None) -> int:
    """Calculate Manhattan distance between two positions."""
    if pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def run_episode(env: WarehouseEnv, path: list[str]) -> tuple[list[list[list[str]]], dict]:
    """
    Execute a path in the environment and collect frames and metrics.
    
    Args:
        env: The warehouse environment.
        path: List of actions to execute.
        
    Returns:
        Tuple of (frames, metrics) for visualization.
    """
    env.reset()
    
    # Find pickup and dropoff positions
    pickup_pos = None
    dropoff_pos = None
    for r, row in enumerate(env.grid):
        for c, ch in enumerate(row):
            if ch == "P":
                pickup_pos = (r, c)
            elif ch == "D":
                dropoff_pos = (r, c)
    
    # Collect initial frame and metrics
    frames = [env.render_grid()]
    metrics = {
        "rewards": [0.0],
        "battery": [env.state.battery],
        "dist_pickup": [manhattan_distance(env.state.robot_pos, pickup_pos)],
        "dist_dropoff": [manhattan_distance(env.state.robot_pos, dropoff_pos)],
    }
    
    cumulative_reward = 0.0
    
    # Execute each action in the path
    for action in path:
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        
        # Collect frame and metrics
        frames.append(env.render_grid())
        metrics["rewards"].append(cumulative_reward)
        metrics["battery"].append(env.state.battery)
        metrics["dist_pickup"].append(manhattan_distance(env.state.robot_pos, pickup_pos))
        metrics["dist_dropoff"].append(manhattan_distance(env.state.robot_pos, dropoff_pos))
        
        if terminated or truncated:
            break
    
    return frames, metrics


def main():
    """Run A* pathfinding and animate the result."""
    # Create environment
    env = WarehouseEnv()
    
    print("=" * 60)
    print("A* PATHFINDER - WAREHOUSE EPISODE")
    print("=" * 60)
    print()
    print("Warehouse Grid:")
    print(env.render_with_legend())
    print()
    
    # Create pathfinder and search for optimal path
    pathfinder = AStarPathfinder(env)
    
    print("Running A* Search...")
    print(f"  Start position: {env.start_pos}")
    print(f"  Pickup position: {pathfinder.pickup_pos}")
    print(f"  Dropoff position: {pathfinder.dropoff_pos}")
    print()
    
    result = pathfinder.search()
    
    # Display search results
    print("SEARCH STATISTICS:")
    print(f"  Success: {result.success}")
    print(f"  Path length: {result.path_length} steps")
    print(f"  Nodes expanded: {result.nodes_expanded}")
    print(f"  Explored nodes: {result.explored_count}")
    print(f"  Max frontier size: {result.frontier_max_size}")
    print(f"  Computation time: {result.computation_time:.6f} seconds")
    print()
    print("  Algorithm: A* with f(n) = g(n) + h(n)")
    print("  Heuristic: Manhattan distance h(n) = |x_n - x_goal| + |y_n - y_goal|")
    print()
    
    if not result.success:
        print("No path found! Cannot animate.")
        return
    
    print("OPTIMAL PATH:")
    print(f"  {' -> '.join(result.path)}")
    print()
    
    # Run the episode and collect frames
    print("Executing path and collecting animation frames...")
    frames, metrics = run_episode(env, result.path)
    print(f"  Collected {len(frames)} frames")
    print(f"  Final cumulative reward: {metrics['rewards'][-1]:.2f}")
    print()
    
    # Animate the episode
    print("Launching animation...")
    print("  Controls:")
    print("    SPACE - Pause/Resume")
    print("    LEFT/RIGHT - Step through frames (when paused)")
    print()
    
    replay_animation(frames, metrics, interval_ms=300, speed=1.0)


if __name__ == "__main__":
    main()
