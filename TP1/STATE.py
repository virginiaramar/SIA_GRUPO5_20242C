# Run python BOARDS\game.py y cambiar el nombre de los levels

from enum import Enum
from utils.direction import Direction


class SokobanGameState:
    def __init__(self, num_moves=0):
        self.board_layout = []
        self.goal_positions = set()
        self.box_positions = set()
        self.player_position = None
        self.num_moves = num_moves
        self.move_history = []

    def __str__(self):
        return ''.join(str(move) for move in self.move_history)

    def __eq__(self, other):
        return self.box_positions == other.box_positions and self.player_position == other.player_position

    def __hash__(self):
        return hash((tuple(self.box_positions), self.player_position))

    def remaining_boxes(self):
        return len(self.box_positions - self.goal_positions)

    def clone(self):
        cloned_state = SokobanGameState()
        cloned_state.board_layout = self.board_layout
        cloned_state.box_positions = set(self.box_positions)
        cloned_state.goal_positions = set(self.goal_positions)
        cloned_state.player_position = self.player_position
        cloned_state.num_moves = self.num_moves
        cloned_state.move_history = self.move_history[:]
        return cloned_state


class CellType(Enum):
    EMPTY = ' '
    OBSTACLE = '#' #Wall
    TARGET = '.'
    BOX = '$'
    BOX_ON_TARGET = '*'
    PLAYER = '@'

    def __new__(cls, symbol: str):
        obj = object.__new__(cls)
        obj._value_ = symbol
        return obj

    def __init__(self, symbol: str):
        self.symbol = symbol

    def __str__(self):
        return self.symbol

    def __eq__(self, other):
        return self.symbol == other.symbol


class SokobanGame:
    GOALS_BOXES_MISMATCH = -1
    NO_GOALS_OR_BOXES = -2

    def __init__(self):
        self.state = SokobanGameState()

    def load_board(self, file_path='TP1/boards/board.txt'):
        with open(file_path, 'r') as file:
            x = 0
            y = 0
            max_length = 0

            for line in file:
                self.state.board_layout.append([])
                for char in line.strip():
                    if char == CellType.EMPTY.symbol:
                        self.state.board_layout[y].append(CellType.EMPTY)
                    elif char == CellType.OBSTACLE.symbol:
                        self.state.board_layout[y].append(CellType.OBSTACLE)
                    elif char == CellType.TARGET.symbol:
                        self.state.goal_positions.add((x, y))
                        self.state.board_layout[y].append(CellType.EMPTY)
                    elif char == CellType.PACKAGE.symbol:
                        self.state.box_positions.add((x, y))
                        self.state.board_layout[y].append(CellType.EMPTY)
                    elif char == CellType.PACKAGE_ON_TARGET.symbol:
                        self.state.box_positions.add((x, y))
                        self.state.goal_positions.add((x, y))
                        self.state.board_layout[y].append(CellType.EMPTY)
                    elif char == CellType.PLAYER.symbol:
                        self.state.player_position = (x, y)
                        self.state.board_layout[y].append(CellType.EMPTY)
                    x += 1
                x = 0
                y += 1

            if len(self.state.goal_positions) != len(self.state.box_positions):
                return self.GOALS_BOXES_MISMATCH

            if not self.state.goal_positions:
                return self.NO_GOALS_OR_BOXES

            for line in self.state.board_layout:
                max_length = max(max_length, len(line))

            for line in self.state.board_layout:
                while len(line) < max_length:
                    line.append(CellType.EMPTY)

    def display_board(self):
        for y, row in enumerate(self.state.board_layout):
            for x, cell in enumerate(row):
                if (x, y) == self.state.player_position:
                    print(CellType.PLAYER.symbol, end='')
                elif (x, y) in self.state.box_positions:
                    if (x, y) in self.state.goal_positions:
                        print(CellType.PACKAGE_ON_TARGET.symbol, end='')
                    else:
                        print(CellType.PACKAGE.symbol, end='')
                elif (x, y) in self.state.goal_positions:
                    print(CellType.TARGET.symbol, end='')
                else:
                    print(cell.symbol, end='')
            print()
        print()
        print('Goals:', ' '.join(str(goal) for goal in self.state.goal_positions))
        print('Player position:', self.state.player_position)
        print('Boxes remaining:', self.state.remaining_boxes())
        print('Total moves:', self.state.num_moves)

    def move_player(self, direction: Direction):
        move_x, move_y = direction.direction.x, direction.direction.y
        new_player_pos = (self.state.player_position[0] + move_x, self.state.player_position[1] + move_y)
        new_box_pos = (self.state.player_position[0] + 2 * move_x, self.state.player_position[1] + 2 * move_y)

        if self.is_valid_move(new_player_pos) and self.is_empty(new_player_pos):
            self.state.player_position = new_player_pos
        elif new_player_pos in self.state.box_positions and self.is_valid_move(new_box_pos) and self.is_empty(new_box_pos):
            self.state.box_positions.remove(new_player_pos)
            self.state.box_positions.add(new_box_pos)
            self.state.player_position = new_player_pos
        else:
            return

        self.state.num_moves += 1
        self.state.move_history.append(direction)

    def remaining_boxes(self):
        return self.state.remaining_boxes()

    def check_win(self):
        return self.state.remaining_boxes() == 0

    def is_empty(self, position):
        return (self.state.board_layout[position[1]][position[0]] == CellType.EMPTY) and \
               (position not in self.state.box_positions)

    def is_valid_move(self, position):
        return 0 <= position[0] < len(self.state.board_layout[0]) and \
               0 <= position[1] < len(self.state.board_layout)

    def update_state(self, state: SokobanGameState):
        self.state = state.clone()

    def get_state(self):
        return self.state

    def check_deadlock(self):
        for box in self.state.box_positions:
            if box in self.state.goal_positions:
                continue
            vertical_blocked = (self.state.board_layout[box[1] - 1][box[0]] == CellType.OBSTACLE) or \
                               (self.state.board_layout[box[1] + 1][box[0]] == CellType.OBSTACLE)
            horizontal_blocked = (self.state.board_layout[box[1]][box[0] - 1] == CellType.OBSTACLE) or \
                                 (self.state.board_layout[box[1]][box[0] + 1] == CellType.OBSTACLE)
            if vertical_blocked and horizontal_blocked:
                return True
        return False
