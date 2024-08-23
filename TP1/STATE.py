class SokobanState:
    def __init__(self, player_pos, box_positions, goal_positions, walls, grid_size):
        self.player_pos = player_pos
        self.box_positions = frozenset(box_positions)
        self.goal_positions = frozenset(goal_positions)
        self.walls = frozenset(walls)
        self.grid_size = grid_size

    def move(self, direction):
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        if direction not in directions:
            raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

        delta_x, delta_y = directions[direction]
        new_player_pos = (self.player_pos[0] + delta_x, self.player_pos[1] + delta_y)

        # Verifica si el nuevo movimiento está dentro del rango del tablero
        if not (0 <= new_player_pos[0] < self.grid_size[0] and 0 <= new_player_pos[1] < self.grid_size[1]):
            return None

        if new_player_pos in self.walls:
            return None  # El jugador no puede moverse a través de paredes

        if new_player_pos in self.box_positions:
            new_box_pos = (new_player_pos[0] + delta_x, new_player_pos[1] + delta_y)
            if new_box_pos in self.walls or new_box_pos in self.box_positions:
                return None  # No se puede mover la caja a través de paredes o otras cajas

            # Mueve la caja
            new_box_positions = set(self.box_positions)
            new_box_positions.remove(new_player_pos)
            new_box_positions.add(new_box_pos)
        else:
            new_box_positions = self.box_positions

        # Crea un nuevo estado con la nueva posición del jugador y las cajas
        return SokobanState(new_player_pos, new_box_positions, self.goal_positions, self.walls, self.grid_size)

    def is_goal_state(self):
        return self.box_positions == self.goal_positions

    def __eq__(self, other):
        return (self.player_pos == other.player_pos and
                self.box_positions == other.box_positions)

    def __hash__(self):
        return hash((self.player_pos, self.box_positions))

    def __str__(self):
        width, height = self.grid_size
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        for (x, y) in self.walls:
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '#'

        for (x, y) in self.goal_positions:
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '.'

        for (x, y) in self.box_positions:
            if 0 <= x < width and 0 <= y < height:
                if (x, y) in self.goal_positions:
                    grid[y][x] = '*'
                else:
                    grid[y][x] = '$'

        px, py = self.player_pos
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = '@'

        return '\n'.join(''.join(row) for row in grid)

def read_file_to_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determinar el ancho máximo del tablero
    max_width = max(len(line.rstrip()) for line in lines)
    
    # Convertir cada línea del archivo en una fila de la matriz, rellenando con espacios si es necesario
    matrix = [list(line.rstrip().ljust(max_width)) for line in lines]

    return matrix

def load_board_from_file(file_path):
    matrix = read_file_to_matrix(file_path)

    # Determinar el tamaño del tablero
    grid_size = (len(matrix[0]), len(matrix))

    player_pos = None
    box_positions = set()
    goal_positions = set()
    walls = set()

    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell == '#':
                walls.add((x, y))
            elif cell == '@':
                player_pos = (x, y)
            elif cell == '$':
                box_positions.add((x, y))
            elif cell == '.':
                goal_positions.add((x, y))
            elif cell == '*':
                box_positions.add((x, y))
                goal_positions.add((x, y))

    return SokobanState(player_pos, box_positions, goal_positions, walls, grid_size)

if __name__ == "__main__":
    file_path = 'BOARDS\LEVELS\difficult.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)
    print(initial_state)
