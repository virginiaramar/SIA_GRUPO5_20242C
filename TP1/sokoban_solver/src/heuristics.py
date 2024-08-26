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
    return (
        distance_to_nearest_box(state) +
        distance_from_box_to_goals(state) +
        calculate_wall_penalty(state)
    )

def distance_to_nearest_box(state):
    min_distance = float('inf')
    for box in state.boxes:
        distance = manhattan_distance(state.player, box)
        min_distance = min(min_distance, distance)
    return min_distance if min_distance != float('inf') else 0

def distance_from_box_to_goals(state):
    total_distance = 0
    for box in state.boxes:
        min_distance = float('inf')
        for target in state.targets:
            distance = manhattan_distance(box, target)
            min_distance = min(min_distance, distance)
        total_distance += min_distance if min_distance != float('inf') else 0
    return total_distance

def calculate_wall_penalty(state):
    penalty = 0
    for box in state.boxes:
        adjacent_walls = 0
        for direction in ['up', 'down', 'left', 'right']:
            adjacent_pos = move_box_in_direction(state, box, direction)
            if adjacent_pos and adjacent_pos in state.walls:
                adjacent_walls += 1
        if adjacent_walls > 0:
            penalty += 20 * adjacent_walls  # Penaliza más por estar cerca de varias paredes
    return penalty

def move_box_in_direction(state, box, direction):
    directions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }
    delta_x, delta_y = directions[direction]
    new_box_pos = (box[0] + delta_x, box[1] + delta_y)
    
    if not (0 <= new_box_pos[0] < state.width and 0 <= new_box_pos[1] < state.height):
        return None
    
    return new_box_pos
