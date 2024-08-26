import heapq
import time
from .heuristics import simple_heuristic, advanced_heuristic, better_heuristic
from collections import deque
from .game import State

def reconstruct_path(state):
    path = []
    while state is not None:
        path.append(state)
        state = state.parent
    return path[::-1]

def bfs(initial_state):
    frontier = deque([initial_state])
    visited = set()
    nodes_expanded = 0

    while frontier:
        current_state = frontier.popleft()
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(current_state), nodes_expanded, len(frontier)

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited and next_state not in frontier:
                frontier.append(next_state)

    return None, nodes_expanded, 0

def dfs(initial_state):
    stack = [initial_state]
    visited = set()
    nodes_expanded = 0

    while stack:
        current_state = stack.pop()
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(current_state), nodes_expanded, len(stack)

        visited.add(current_state)

        for next_state in current_state.get_successors()[::-1]:  # Invierte el orden de los sucesores para DFS
            if next_state not in visited and next_state not in stack:
                stack.append(next_state)

    return None, nodes_expanded, 0

def astar(initial_state, heuristic=better_heuristic):
    frontier = [(0, initial_state)]  # heapq, initialized with the starting state
    cost_so_far = {initial_state: 0}  # Costs to reach each state
    visited = set()  # States that have already been visited
    nodes_expanded = 0

    while frontier:
        _, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(current_state), nodes_expanded, len(frontier)

        if current_state in visited:
            continue  # Skip processing if state is already visited
        
        visited.add(current_state)

        for next_state in current_state.get_successors():
            new_cost = cost_so_far[current_state] + 1  # Assuming uniform cost for moves
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))

        # Optional: Log progress periodically (adjust frequency as needed)
        if nodes_expanded % 10000 == 0:
            print(f"Nodos explorados: {nodes_expanded}, Tamaño de la frontera: {len(frontier)}")

    return None, nodes_expanded, 0

def greedy(initial_state, heuristic=simple_heuristic):
    frontier = [(heuristic(initial_state), initial_state)]  # heapq, based on heuristic
    visited = set()
    nodes_expanded = 0

    while frontier:
        _, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_state.is_goal():
            return reconstruct_path(current_state), nodes_expanded, len(frontier)

        if current_state in visited:
            continue  # Skip processing if state is already visited

        visited.add(current_state)

        for next_state in current_state.get_successors():
            if next_state not in visited:
                priority = heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))

        # Proporcionar feedback cada 10,000 nodos explorados
        if nodes_expanded % 10000 == 0:
            print(f"Greedy: Nodos explorados: {nodes_expanded}, Tamaño de la frontera: {len(frontier)}")

    return None, nodes_expanded, 0

def iddfs(initial_state, max_depth=1000, time_limit=300, depth_step=10):
    start_time = time.time()
    total_nodes_expanded = 0
    visited = set()
    limit_nodes = []
    cur_max_depth = depth_step

    while cur_max_depth <= max_depth:
        if time.time() - start_time > time_limit:
            print(f"Tiempo límite excedido en IDDFS después de explorar hasta profundidad {cur_max_depth}")
            return None, total_nodes_expanded, len(limit_nodes)

        print(f"IDDFS: Explorando profundidad {cur_max_depth}")
        result, nodes_expanded = _dfs(initial_state, 0, cur_max_depth, visited, limit_nodes)
        total_nodes_expanded += nodes_expanded

        if result is not None:
            return result, total_nodes_expanded, len(limit_nodes)
        
        if not limit_nodes:
            print(f"IDDFS: No hay más nodos para explorar. Profundidad máxima alcanzada: {cur_max_depth}")
            return None, total_nodes_expanded, 0

        cur_max_depth += depth_step
        visited.clear()
        limit_nodes.clear()

    print(f"IDDFS: Profundidad máxima {max_depth} alcanzada sin encontrar solución")
    return None, total_nodes_expanded, len(limit_nodes)

def _dfs(state, depth, max_depth, visited, limit_nodes):
    if state.is_goal():
        return [state], 1

    if depth >= max_depth:
        limit_nodes.append(state)
        return None, 1

    state_hash = hash(state)
    if state_hash in visited:
        return None, 1
    visited.add(state_hash)

    nodes_expanded = 1
    for next_state in state.get_successors():
        result, sub_nodes = _dfs(next_state, depth + 1, max_depth, visited, limit_nodes)
        nodes_expanded += sub_nodes
        if result is not None:
            return [state] + result, nodes_expanded

    return None, nodes_expanded