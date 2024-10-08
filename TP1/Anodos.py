import heapq
from STATE import load_board_from_file
from heuristics import H4  # Asegúrate de importar la heurística que estás usando

class AStar:
    def __init__(self, initial_state, heuristic):
        self.initial_state = initial_state
        self.heuristic = heuristic
        self.priority_queue = []
        self.expanded_nodes = 0  # Contador de nodos expandidos
        self.visited = set()
    
    def search(self):
        self.expanded_nodes = 0
        start_node = (self.heuristic(self.initial_state), 0, self.initial_state, [])  # El cuarto elemento es el camino
        heapq.heappush(self.priority_queue, start_node)
        self.visited.add(self.initial_state)
        
        while self.priority_queue:
            _, cost, current_state, path = heapq.heappop(self.priority_queue)
            if current_state.is_goal_state():
                return path, self.expanded_nodes
            
            self.expanded_nodes += 1
            
            for direction in ['up', 'down', 'left', 'right']:
                neighbor = current_state.move(direction)
                if neighbor and neighbor not in self.visited:
                    self.visited.add(neighbor)
                    f_cost = cost + 1 + self.heuristic(neighbor)
                    new_path = path + [direction]  # Actualiza el camino con el nuevo movimiento
                    heapq.heappush(self.priority_queue, (f_cost, cost + 1, neighbor, new_path))
        
        return None, self.expanded_nodes

def h5_heuristic(state):
    return H4(state).heuristic()

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/medium.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)

    # Utiliza la heurística H4
    heuristic = h5_heuristic

    astar = AStar(initial_state, heuristic)
    solution, num_steps = astar.search()

    if solution:
        print("¡Solución encontrada!")
        print(f"Total de pasos: {num_steps}")
        print("Camino a la solución:")
        for step in solution:
            print(step)
    else:
        print("No se encontró solución.")
