from collections import deque
from STATE import load_board_from_file

class BFS:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.queue = deque([initial_state])
        self.visited = set()
        self.expanded_nodes = 0

    def search(self):
        self.expanded_nodes = 0
        self.visited.add(self.initial_state)
        
        while self.queue:
            current_state = self.queue.popleft()
            if current_state.is_goal_state():
                return self._reconstruct_path(current_state), self.expanded_nodes
            
            self.expanded_nodes += 1
            
            for direction in ['up', 'down', 'left', 'right']:
                neighbor = current_state.move(direction)
                if neighbor and neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.queue.append(neighbor)
        
        return None, self.expanded_nodes
    
    def _reconstruct_path(self, state):
        # Este método debe ser implementado para reconstruir el camino desde el estado objetivo
        # En este caso, simplemente devuelve el estado objetivo
        return state

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/medium.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)
    bfs = BFS(initial_state)
    solution, num_steps = bfs.search()
    
    print(f"Solution found with {num_steps} steps.")
    if solution:
        print("Path to solution:")
        print(solution)
    else:
        print("No solution found.")
