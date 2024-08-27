import heapq
import time
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

def astar(initial_state, heuristic):
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

def greedy(initial_state, heuristic):
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
import time

def iddfs(initial_state, max_depth=100, time_limit=300):
    start_time = time.time()
    total_nodes_expanded = 0
    
    for depth_limit in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            print(f"Tiempo límite excedido en IDDFS después de explorar hasta profundidad {depth_limit - 1}")
            return None, total_nodes_expanded, 0
        
        result, nodes_expanded = depth_limited_search(initial_state, depth_limit)
        total_nodes_expanded += nodes_expanded
        
        if result is not None:
            print(f"Solución encontrada a profundidad {depth_limit}")
            return reconstruct_path(result), total_nodes_expanded, 0
        
        if time.time() - start_time > time_limit:
            print(f"Tiempo límite excedido en IDDFS después de explorar hasta profundidad {depth_limit}")
            return None, total_nodes_expanded, 0
    
    print(f"No se encontró solución dentro de la profundidad máxima {max_depth}")
    return None, total_nodes_expanded, 0

def depth_limited_search(initial_state, depth_limit):
    stack = [(initial_state, 0)]  # (estado, profundidad)
    visited = set()
    nodes_expanded = 0
    
    while stack:
        state, depth = stack.pop()
        
        if depth > depth_limit:
            continue
        
        if state.is_goal():
            return state, nodes_expanded
        
        state_hash = hash(state)
        if state_hash in visited:
            continue
        
        visited.add(state_hash)
        nodes_expanded += 1
        
        if depth < depth_limit:
            for successor in state.get_successors():
                if hash(successor) not in visited:
                    stack.append((successor, depth + 1))
    
    return None, nodes_expanded