from collections import deque
import time

from Solution import Solution


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

        if direction in directions:
            delta_x, delta_y = directions[direction]
            new_player_pos = (self.player_pos[0] + delta_x, self.player_pos[1] + delta_y)
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

        if new_player_pos in self.walls:
            return None

        if new_player_pos in self.box_positions:
            new_box_pos = (new_player_pos[0] + delta_x, new_player_pos[1] + delta_y)
            if new_box_pos in self.walls or new_box_pos in self.box_positions:
                return None
            else:
                new_box_positions = set(self.box_positions)
                new_box_positions.remove(new_player_pos)
                new_box_positions.add(new_box_pos)
                return SokobanState(new_player_pos, new_box_positions, self.goal_positions, self.walls, self.grid_size)

        return SokobanState(new_player_pos, self.box_positions, self.goal_positions, self.walls, self.grid_size)

    def is_goal_state(self):
        return self.box_positions == self.goal_positions

    def __eq__(self, other):
        return (self.player_pos == other.player_pos and
                self.box_positions == other.box_positions)

    def __hash__(self):
        return hash((self.player_pos, self.box_positions))

    # "#" is a wall, "G" is where a wall should be put, "B" is a box and "*" is a box in the correct place
    def __str__(self):
        width, height = self.grid_size
        grid = [['.' for _ in range(width)] for _ in range(height)]

        for (x, y) in self.walls:
            grid[y][x] = '#'

        for (x, y) in self.goal_positions:
            grid[y][x] = 'G'

        for (x, y) in self.box_positions:
            if (x, y) in self.goal_positions:
                grid[y][x] = '*'
            else:
                grid[y][x] = 'B'

        px, py = self.player_pos
        grid[py][px] = 'P'

        return '\n'.join(''.join(row) for row in grid)


def bfs_sokoban(initial_state):
    start = time.time()
    queue = deque([initial_state])
    visited = set()
    parent = {initial_state: None}
    nb_nodes = 0
    nb_boundary = 1

    while queue:
        current_state = queue.popleft()
        nb_nodes += 1

        if current_state.is_goal_state():
            end = time.time()
            path = show_path(parent, current_state)
            cost = len(path) - 1
            return Solution(path=path, time=end - start, cost=cost, nb_nodes=nb_nodes, nb_boundary=len(queue))

        visited.add(current_state)

        for direction in ["up", "down", "left", "right"]:
            new_state = current_state.move(direction)
            if new_state and new_state not in visited:
                queue.append(new_state)
                parent[new_state] = current_state

        nb_boundary = len(queue)

    end = time.time()
    return Solution(path=None, time=end - start, cost=None, nb_nodes=nb_nodes, nb_boundary=0)


def show_path(parent, current_state):
    path = []
    parent_path = parent[current_state]
    path.append(current_state)
    while parent_path:
        path.append(parent_path)
        parent_path = parent[parent_path]
    path.reverse()
    return path
