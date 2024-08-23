class State:
    def __init__(self, player, boxes, targets, walls, width, height):
        self.player = player  # tupla (x, y)
        self.boxes = set(boxes)  # conjunto de tuplas (x, y)
        self.targets = set(targets)  # conjunto de tuplas (x, y)
        self.walls = set(walls)  # conjunto de tuplas (x, y)
        self.width = width
        self.height = height

    def is_goal(self):  # útil si queremos comparar si las cajas están en los goals.
        return self.boxes == self.targets

    def __eq__(self, other):    # Compara estados diferentes
        return (self.player == other.player and 
                self.boxes == other.boxes)

    def __hash__(self):     # devuelve el hash del State propio, Se utiliza principalmente
                            # cuando quieres usar tu objeto en estructuras de datos como conjuntos (sets)
                            #  o como claves en diccionarios.Usamos frozenset(self.boxes) porque los conjuntos 
                            # normales no son "hashables", pero los frozensets sí.
        return hash((self.player, frozenset(self.boxes)))

def load_level(filename): # Recorre cada carácter del archivo, usando sus coordenadas.
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

def move(self, dx, dy):
        new_player = (self.player[0] + dx, self.player[1] + dy)
        new_boxes = self.boxes.copy()

        if new_player in self.walls:
            return None

        if new_player in self.boxes:
            new_box_position = (new_player[0] + dx, new_player[1] + dy)
            if new_box_position in self.walls or new_box_position in self.boxes:
                return None
            new_boxes.remove(new_player)
            new_boxes.add(new_box_position)

        return State(new_player, new_boxes, self.targets, self.walls, self.width, self.height)

def get_successors(self):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Abajo, Derecha, Arriba, Izquierda
        return [self.move(dx, dy) for dx, dy in moves if self.move(dx, dy) is not None]
