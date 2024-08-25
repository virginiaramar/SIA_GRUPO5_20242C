class State:
    def __init__(self, player, boxes, targets, walls, width, height):
        self.player = player  # tupla (x, y)
        self.boxes = set(boxes)  # conjunto de tuplas (x, y)
        self.targets = set(targets)  # conjunto de tuplas (x, y)
        self.walls = set(walls)  # conjunto de tuplas (x, y)
        self.width = width
        self.height = height

    def is_goal(self):
        return self.boxes == self.targets

    def __eq__(self, other):
        return (self.player == other.player and 
                self.boxes == other.boxes)

    def __hash__(self):
        return hash((self.player, frozenset(self.boxes)))
    
    def __lt__(self, other):
        # Este método es necesario para la comparación en el heap
        # Comparamos los estados basándonos en su representación de cadena
        return str(self) < str(other)

    def move(self, dx, dy):
        new_player = (self.player[0] + dx, self.player[1] + dy)
        if (new_player[0] < 0 or new_player[0] >= self.width or
            new_player[1] < 0 or new_player[1] >= self.height):
            return None  # Movimiento fuera de los límites
        new_boxes = self.boxes.copy()
        if new_player in self.walls:
            return None
        if new_player in self.boxes:
            new_box_position = (new_player[0] + dx, new_player[1] + dy)
            if (new_box_position[0] < 0 or new_box_position[0] >= self.width or
                new_box_position[1] < 0 or new_box_position[1] >= self.height or
                new_box_position in self.walls or new_box_position in self.boxes):
                return None
            new_boxes.remove(new_player)
            new_boxes.add(new_box_position)
        return State(new_player, new_boxes, self.targets, self.walls, self.width, self.height)

    def get_successors(self):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Abajo, Derecha, Arriba, Izquierda
        return [self.move(dx, dy) for dx, dy in moves if self.move(dx, dy) is not None]

    def __str__(self):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for wall in self.walls:
            if 0 <= wall[1] < self.height and 0 <= wall[0] < self.width:
                grid[wall[1]][wall[0]] = '#'
        for target in self.targets:
            if 0 <= target[1] < self.height and 0 <= target[0] < self.width:
                grid[target[1]][target[0]] = '.'
        for box in self.boxes:
            if 0 <= box[1] < self.height and 0 <= box[0] < self.width:
                if box in self.targets:
                    grid[box[1]][box[0]] = '*'
                else:
                    grid[box[1]][box[0]] = '$'
        if 0 <= self.player[1] < self.height and 0 <= self.player[0] < self.width:
            if self.player in self.targets:
                grid[self.player[1]][self.player[0]] = '+'
            else:
                grid[self.player[1]][self.player[0]] = '@'
        return '\n'.join(''.join(row) for row in grid)

def load_level(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    player = None
    boxes = set()
    targets = set()
    walls = set()
    height = len(lines)
    width = max(len(line.strip()) for line in lines)

    for y, line in enumerate(lines):
        for x, char in enumerate(line.strip()):
            if char == '@':
                player = (x, y)
            elif char == '$':
                boxes.add((x, y))
            elif char == '.':
                targets.add((x, y))
            elif char == '#':
                walls.add((x, y))
            elif char == '*':
                boxes.add((x, y))
                targets.add((x, y))
            elif char == '+':
                player = (x, y)
                targets.add((x, y))

    # Verifica que el jugador esté dentro de los límites
    if player is None or player[0] >= width or player[1] >= height:
        raise ValueError("Invalid player position or no player found")

    return State(player, boxes, targets, walls, width, height)