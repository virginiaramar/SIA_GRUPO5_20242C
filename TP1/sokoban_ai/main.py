import sys
import time

from SokobanState import SokobanState, bfs_sokoban, dfs_sokoban
from collections import deque

def main():
    # Define the grid size
    grid_size = (7, 5)  # 7 columns by 5 rows

    # Initial state setup
    player_pos = (1, 1)
    box_positions = [(3, 2), (4, 3)]
    goal_positions = [(5, 2), (5, 3)]
    walls = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
             (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
             (6, 1), (6, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),(6,3),(6,2)]

    initial_state = SokobanState(player_pos, box_positions, goal_positions, walls, grid_size)

    solution = bfs_sokoban(initial_state)

    # if solution.path:
    #     print(f"time cost: {solution.time:.2f} seconds")
    #     print(f"solution cost: {solution.cost} moves")
    #     print(f"number of nodes visited: {solution.nb_nodes}")
    #     print(f"nodes in the queue: {solution.nb_boundary}")
    #     for step, state in enumerate(solution.path):
    #         print(f"Step {step}:")
    #         print(state)
    #         print("\n")
    # else:
    #     print("No solution found")
    sys.setrecursionlimit(10000)
    visited = set()
    parent = {initial_state: None}
    solution2 = dfs_sokoban(visited, initial_state, parent, 0, time.time()  )

    if solution2.path:
        print(f"time cost: {solution2.time:.2f} seconds")
        print(f"solution cost: {solution2.cost} moves")
        print(f"number of nodes visited: {solution2.nb_nodes}")
        print(f"nodes in the queue: {solution2.nb_boundary}")
        for step, state in enumerate(solution2.path):
            print(f"Step {step}:")
            print(state)
            print("\n")
    else:
        print("No solution found")

if __name__ == "__main__":
    main()
