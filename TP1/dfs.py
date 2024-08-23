from collections import deque
from STATE import load_board_from_file, SokobanState

def dfs(initial_state):
    stack = [initial_state]
    visited = set()
    visited.add(initial_state)
    parent = {initial_state: None}

    while stack:
        state = stack.pop()
        if state.is_goal_state():
            return reconstruct_path(parent, state)

        for direction in ['up', 'down', 'left', 'right']:
            new_state = state.move(direction)
            if new_state and new_state not in visited:
                visited.add(new_state)
                parent[new_state] = state
                stack.append(new_state)

    return None

def reconstruct_path(parent, state):
    path = []
    while state:
        path.append(state)
        state = parent[state]
    return path[::-1]  # Reversed path

def print_solution(path):
    if path:
        #for state in path:
            #print(state)
            #print()
        print("Final state:")
        print(path[-1])
    else:
        print("No solution found.")

if __name__ == "__main__":
    # Cargar el tablero desde un archivo
    file_path = 'BOARDS\LEVELS\easy.txt'
    initial_state = load_board_from_file(file_path)
    
    # Ejecutar DFS
    path = dfs(initial_state)
    
    # Imprimir la solución
    print_solution(path)


if __name__ == "__main__":
    file_path = 'BOARDS\LEVELS\prueba.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)
    solution = dfs(initial_state)
    if solution:
        print("Solución encontrada")
        #print(solution)
    else:
        print("No se encontró solución")
