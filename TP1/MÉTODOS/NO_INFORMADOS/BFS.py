import time
from collections import deque
from STATE import SokobanState

class BFS:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def search(self):
        start = time.time()
        queue = deque([self.initial_state])
        visited = set()
        parent = {}
        nb_nodes = 0

        while queue:
            current_state = queue.popleft()
            if current_state.is_goal_state():
                end = time.time()
                path = self.show_path(parent, current_state)
                cost = len(path) - 1
                return {
                    "result": "éxito",
                    "path": path,
                    "cost": cost,
                    "nodes_expanded": nb_nodes,
                    "nb_boundary": len(queue),
                    "execution_time": end - start
                }

            visited.add(current_state)
            nb_nodes += 1

            for direction in ["up", "down", "left", "right"]:
                new_state = current_state.move(direction)
                if new_state and new_state not in visited and new_state not in queue:
                    queue.append(new_state)
                    visited.add(new_state)
                    parent[new_state] = current_state

        return {"result": "fracaso"}

    def show_path(self, parent, current_state):
        path = []
        while current_state in parent:
            path.append(current_state)
            current_state = parent[current_state]
        path.reverse()
        return path

