from collections import deque
from game import State

def bfs(initial_state):
    frontier = deque([initial_state])
    came_from = {initial_state: None}
    visited = set()

    while frontier:
        current_state = frontier.popleft()

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in frontier:
                frontier.append(next_state)
                came_from[next_state] = current_state

    return None  # No se encontró solución

def dfs(initial_state):
    frontier = [initial_state]
    came_from = {initial_state: None}
    visited = set()

    while frontier:
        current_state = frontier.pop()

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in frontier:
                frontier.append(next_state)
                came_from[next_state] = current_state

    return None  # No se encontró solución

def reconstruct_path(came_from, final_state):
    path = []
    current = final_state
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # Añade el estado inicial
    return path[::-1]  # Invierte el camino para que vaya del inicio al final