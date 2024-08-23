import time
from collections import deque
from STATE import SokobanState

def dfs_sokoban(visited, initial_state, parent, nb_nodes, start):
    if initial_state.is_goal_state():
        end = time.time()
        path = show_path(parent, initial_state)
        cost = len(path) - 1  # Costo es el número de movimientos
        return {
            "result": "éxito",
            "path": path,
            "cost": cost,
            "nodes_expanded": nb_nodes,
            "nb_boundary": len(visited) - nb_nodes,
            "execution_time": end - start
        }

    if initial_state and initial_state not in visited:
        nb_nodes += 1
        visited.add(initial_state)

        for direction in ["up", "down", "left", "right"]:
            new_state = initial_state.move(direction)
            if new_state and new_state not in visited:
                parent[new_state] = initial_state
                result = dfs_sokoban(visited, new_state, parent, nb_nodes, start)
                if result:
                    return result

    return {"result": "fracaso"}

def run(board):
    visited = set()
    parent = {}
    start = time.time()
    return dfs_sokoban(visited, board, parent, 0, start)

def show_path(parent, current_state):
    path = []
    while current_state in parent:
        path.append(current_state)
        current_state = parent[current_state]
    path.reverse()
    return path

