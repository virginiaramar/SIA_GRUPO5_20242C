def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def h1_heuristic(state):
    """Distancia mínima del jugador a la caja más cercana."""
    return min(manhattan_distance(state.player, box) for box in state.boxes) if state.boxes else 0

def h2_heuristic(state):
    """Suma de las distancias mínimas de cada caja a cualquier objetivo."""
    return sum(min(manhattan_distance(box, target) for target in state.targets) for box in state.boxes)

def h3_heuristic(state):
    """Combinación de H1 y H2."""
    player_to_box = min(manhattan_distance(state.player, box) for box in state.boxes) if state.boxes else 0
    box_to_goal = sum(min(manhattan_distance(box, target) for target in state.targets) for box in state.boxes)
    return player_to_box + box_to_goal

def h4_heuristic(state):
    player_to_box = min(manhattan_distance(state.player, box) for box in state.boxes)
    box_to_goal = sum(min(manhattan_distance(box, target) for target in state.targets) for box in state.boxes)
    
    # Penalización por cajas en esquinas que no son objetivos
    corner_penalty = sum(20 for box in state.boxes if is_corner(state, box) and box not in state.targets)
    
    # Penalización por cajas bloqueadas contra paredes
    wall_penalty = sum(10 for box in state.boxes if is_blocked_by_wall(state, box))
    
    return player_to_box + box_to_goal + corner_penalty + wall_penalty

def is_corner(state, pos):
    x, y = pos
    return ((x+1, y) in state.walls or (x-1, y) in state.walls) and \
           ((x, y+1) in state.walls or (x, y-1) in state.walls)

def is_blocked_by_wall(state, pos):
    x, y = pos
    return ((x+1, y) in state.walls and (x, y+1) in state.walls) or \
           ((x-1, y) in state.walls and (x, y+1) in state.walls) or \
           ((x+1, y) in state.walls and (x, y-1) in state.walls) or \
           ((x-1, y) in state.walls and (x, y-1) in state.walls)


# Las heurísticas originales
def simple_heuristic(state):
    return sum(min(manhattan_distance(box, target) for target in state.targets) for box in state.boxes)

def advanced_heuristic(state):
    box_cost = sum(min(manhattan_distance(box, target) for target in state.targets) for box in state.boxes)
    player_cost = min(manhattan_distance(state.player, box) for box in state.boxes)
    return box_cost + player_cost

def better_heuristic(state):
    total_distance = 0
    for box in state.boxes:
        min_distance = min(manhattan_distance(box, target) for target in state.targets)
        total_distance += min_distance
    return total_distance