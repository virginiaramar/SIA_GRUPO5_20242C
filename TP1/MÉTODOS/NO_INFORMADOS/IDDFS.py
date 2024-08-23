#Evita explorar estados repetidos y no reinicia desde cero al incrementar la profundidad.
#Csmbiar depth_step csegun sea necesario
from STATE import load_board_from_file


class IDDFS:
    def __init__(self, initial_state, depth_step=10):
        self.initial_state = initial_state
        self.depth_step = depth_step
        self.visited = set()  # Para evitar estados repetidos
        self.limit_nodes = []  # Para almacenar nodos cuando se alcanza el límite de profundidad
        self.cur_max_depth = depth_step

    def search(self):
        """Realiza la búsqueda IDDFS."""
        while True:
            print(f"Buscando con límite de profundidad: {self.cur_max_depth}")
            result = self._dfs(self.initial_state, 0)
            if result is not None:
                return result
            if not self.limit_nodes:
                return None  # No hay más nodos que explorar, no se encontró solución
            self.cur_max_depth += self.depth_step  # Incrementa el límite de profundidad
            self.visited.clear()  # Limpia los nodos visitados
            self.limit_nodes = []  # Limpia los nodos límite

    def _dfs(self, state, depth):
        """Realiza una búsqueda en profundidad hasta el límite actual."""
        if state.is_goal_state():
            return state

        if depth >= self.cur_max_depth:
            self.limit_nodes.append(state)
            return None

        state_hash = hash(state)
        if state_hash in self.visited:
            return None  # Evita estados repetidos
        self.visited.add(state_hash)

        for direction in ['up', 'down', 'left', 'right']:
            neighbor = state.move(direction)
            if neighbor:
                result = self._dfs(neighbor, depth + 1)
                if result is not None:
                    return result

        return None

if __name__ == "__main__":
    file_path = 'BOARDS/LEVELS/medium.txt'  # Cambia esto a la ruta correcta de tu archivo
    initial_state = load_board_from_file(file_path)

    iddfs = IDDFS(initial_state=initial_state)
    solution = iddfs.search()

    if solution:
        print("¡Solución encontrada!")
        print(solution)
    else:
        print("No se encontró solución.")



