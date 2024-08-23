import heapq
from STATE import load_board_from_file
from heuristics import H5, H1, H2, H3, H4  # Asegúrate de importar la heurística que estás usando

class AStar:
    def __init__(self, initial_state, heuristic):
        self.initial_state = initial_state
        self.heuristic = heuristic
        self.priority_queue = []
        self.expanded_nodes = 0  # Contador de nodos expandidos
        self.visited = set()
    
    def search(self):
        self.expanded_nodes = 0
        start_node = (self.heuristic(self.initial_state), 0, self.initial_state)
        heapq.heappush(self.priority_queue, start_node)
        self.visited.add(self.initial_state)
        
        while self.priority_queue:
            _, cost, current_state = heapq.heappop(self.priority_queue)
            if current_state.is_goal_state():
                return self._reconstruct_path(current_state), self.expanded_nodes
            
            self.expanded_nodes += 1
            
            for direction in ['up', 'down', 'left', 'right']:
                neighbor = current_state.move(direction)
                if neighbor and neighbor not in self.visited:
                    self.visited.add(neighbor)
                    f_cost = cost + 1 + self.heuristic(neighbor)
                    heapq.heappush(self.priority_queue, (f_cost, cost + 1, neighbor))
        
        return None, self.expanded_nodes
    
    def _reconstruct_path(self, state):
        # Este método debe ser implementado para reconstruir el camino desde el estado objetivo
        # En este caso, simplemente devuelve el estado objetivo
        return state

def h5_heuristic(state):
    return H3(state).heuristic()

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/verydifficult.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)
    astar = AStar(initial_state, h5_heuristic)
    solution, num_steps = astar.search()
    
    print(f"Solution found with {num_steps} steps.")
    if solution:
        print("Path to solution:")
        #print(solution)
    else:
        print("No solution found.")