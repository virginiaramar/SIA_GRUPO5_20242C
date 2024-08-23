from collections import deque
import time
from STATE import load_board_from_file, SokobanState

class Solution:
    def __init__(self, path, time, cost, nb_nodes, nb_boundary):
        self.path = path
        self.time = time
        self.cost = cost
        self.nb_nodes = nb_nodes
        self.nb_boundary = nb_boundary

def show_path(parent, current_state):
    path = []
    while current_state is not None:
        path.append(current_state)
        current_state = parent[current_state]
    path.reverse()
    return path

def bfs_sokoban(initial_state):
    start = time.time()
    queue = deque([(initial_state, [])])  # (estado, ruta)
    visited = set()
    visited.add(initial_state)
    parent = {initial_state: None}
    nb_nodes = 0

    while queue:
        current_state, path = queue.popleft()
        nb_nodes += 1

        if current_state.is_goal_state():
            end = time.time()
            full_path = show_path(parent, current_state)
            cost = len(full_path) - 1
            return Solution(path=full_path, time=end - start, cost=cost, nb_nodes=nb_nodes, nb_boundary=len(queue))

        for direction in ['up', 'down', 'left', 'right']:
            next_state = current_state.move(direction)
            if next_state and next_state not in visited:
                visited.add(next_state)
                parent[next_state] = current_state
                queue.append((next_state, path + [current_state]))

    end = time.time()
    return Solution(path=None, time=end - start, cost=None, nb_nodes=nb_nodes, nb_boundary=0)

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/medium.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)
    solution = bfs_sokoban(initial_state)
    if solution.path:
        print("Solución encontrada")
        # Puedes imprimir la solución aquí si lo deseas, por ejemplo:
        print("Número de pasos en la solución:", len(solution.path))
        #for state in solution.path:
            #print(state)
    else:
        print("No se encontró solución")

def debug_state(state):
    print("Estado Actual:")
    print(state)


