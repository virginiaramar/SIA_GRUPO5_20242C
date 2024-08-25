import heapq
import time
from .heuristics import simple_heuristic, advanced_heuristic
from collections import deque
from .game import State

def bfs(initial_state):
    frontier = deque([initial_state])
    came_from = {initial_state: None}
    visited = set()
    nodes_expanded = 0

    while frontier:
        current_state = frontier.popleft()
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state), nodes_expanded, len(frontier)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in frontier:
                frontier.append(next_state)
                came_from[next_state] = current_state

    return None, nodes_expanded, 0  # No se encontró solución

def dfs(initial_state):
    stack = [initial_state]  # Usamos una pila para realizar el DFS
    came_from = {initial_state: None}
    visited = set()
    nodes_expanded = 0

    while stack:
        current_state = stack.pop()  # Tomamos el último estado añadido (LIFO)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state), nodes_expanded, len(stack)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in stack:
                stack.append(next_state)
                came_from[next_state] = current_state

    return None, nodes_expanded, 0  # No se encontró solución

def astar(initial_state, heuristic=simple_heuristic, time_limit=300, node_limit=1000000):
    start_time = time.time()
    frontier = [(0, initial_state)]
    came_from = {initial_state: None}
    cost_so_far = {initial_state: 0}
    nodes_expanded = 0

    while frontier and nodes_expanded < node_limit:
        if nodes_expanded % 10000 == 0:  # Imprimir progreso cada 10000 nodos
            print(f"Nodos explorados: {nodes_expanded}, Tamaño de la frontera: {len(frontier)}")

        if time.time() - start_time > time_limit:
            print("Tiempo límite excedido")
            return None, nodes_expanded, len(frontier)

        current_cost, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state), nodes_expanded, len(frontier)

        for next_state in current_state.get_successors():
            new_cost = cost_so_far[current_state] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state

    print("Límite de nodos alcanzado" if nodes_expanded >= node_limit else "No se encontró solución")
    return None, nodes_expanded, len(frontier)
    
def reconstruct_path(came_from, final_state):
    path = []
    current = final_state
    while current in came_from:
        if current is None:
            print("Warning: None state in path reconstruction")
            break
        path.append(current)
        current = came_from[current]
    if current is not None:
        path.append(current)  # Añade el estado inicial
    return path[::-1]  # Invierte el camino para que vaya del inicio al final

def astar(initial_state, heuristic=simple_heuristic):
    frontier = [(0, initial_state)]
    came_from = {initial_state: None}
    cost_so_far = {initial_state: 0}
    nodes_expanded = 0

    while frontier:
        current_cost, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state), nodes_expanded, len(frontier)

        for next_state in current_state.get_successors():
            new_cost = cost_so_far[current_state] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state

    return None, nodes_expanded, 0

def greedy(initial_state, heuristic=simple_heuristic):
    frontier = [(heuristic(initial_state), initial_state)]
    came_from = {initial_state: None}
    visited = set()
    nodes_expanded = 0

    while frontier:
        _, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(came_from, current_state), nodes_expanded, len(frontier)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in (state for _, state in frontier):
                priority = heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state

    return None, nodes_expanded, 0

def iddfs(initial_state):
    def dfs_limited(state, depth_limit, path, visited):
        nonlocal nodes_expanded
        nodes_expanded += 1
        
        if depth_limit == 0:
            return None
        if state.is_goal():
            return path

        visited.add(state)

        for next_state in state.get_successors():
            if next_state not in visited:
                result = dfs_limited(next_state, depth_limit - 1, path + [next_state], visited)
                if result is not None:
                    return result

        visited.remove(state)
        return None

    max_depth = 0
    nodes_expanded = 0
    while True:
        visited = set()
        result = dfs_limited(initial_state, max_depth, [initial_state], visited)
        if result is not None:
            return result, nodes_expanded, len(visited)
        max_depth += 1
        if max_depth > 1000:  # Límite de seguridad para evitar bucles infinitos
            return None, nodes_expanded, 0