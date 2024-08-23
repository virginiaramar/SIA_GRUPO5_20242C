import heapq
from heuristics import H5  # Asegúrate de que esta importación esté correcta
from STATE import load_board_from_file


class AStar:
    def __init__(self, initial_state, heuristics_class):
        self.initial_state = initial_state
        self.heuristics = heuristics_class(initial_state)  # Inicializa heurística con el estado inicial
        self.visited = set()
        self.priority_queue = []  # Cola de prioridad para A*
        self.g_cost = {}  # Coste acumulado desde el inicio
        self.parent = {}  # Para reconstruir la solución

    def search(self):
        """Realiza la búsqueda A*."""
        # Inicia el algoritmo con el estado inicial
        heapq.heappush(self.priority_queue, (self.heuristics.heuristic(), self.initial_state))
        self.g_cost[self.initial_state] = 0

        while self.priority_queue:
            current_cost, current_state = heapq.heappop(self.priority_queue)
            
            if current_state.is_goal_state():
                # Encuentra la solución y reconstruye el camino
                path = self.reconstruct_path(current_state)
                return path, len(path) - 1  # Número de pasos es la longitud del camino menos 1

            self.visited.add(current_state)

            for direction in ['up', 'down', 'left', 'right']:
                neighbor = current_state.move(direction)
                if neighbor and neighbor not in self.visited:
                    new_g_cost = self.g_cost[current_state] + 1  # Asume coste uniforme
                    if neighbor not in self.g_cost or new_g_cost < self.g_cost[neighbor]:
                        self.g_cost[neighbor] = new_g_cost
                        f_cost = new_g_cost + self.heuristics.heuristic()  # Calcula el coste total
                        heapq.heappush(self.priority_queue, (f_cost, neighbor))
                        self.parent[neighbor] = current_state

        return None, 0  # No se encontró solución

    def reconstruct_path(self, state):
        """Reconstruye el camino desde el estado objetivo hasta el inicial."""
        path = []
        while state in self.parent:
            path.append(state)
            state = self.parent[state]
        path.append(self.initial_state)
        path.reverse()
        return path

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/medium.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)

    # Utiliza la heurística H1
    heuristics_class = H5

    astar = AStar(initial_state=initial_state, heuristics_class=heuristics_class)
    solution, num_steps = astar.search()

    if solution:
        print("¡Solución encontrada!")
        #for state in solution:
            #print(state)
        print(f"Total de pasos: {num_steps}")
    else:
        print("No se encontró solución.")