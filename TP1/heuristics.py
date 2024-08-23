class Heuristics:
    def __init__(self, state):
        self.state = state

    def is_path_clear(self, start, end):
        """Verifica si hay un camino libre entre start y end."""
        x1, y1 = start
        x2, y2 = end
        
        if x1 == x2:  # Movimiento vertical
            if y1 > y2:
                y1, y2 = y2, y1
            for y in range(y1 + 1, y2):
                if (x1, y) in self.state.walls or (x1, y) in self.state.box_positions:
                    return False
        elif y1 == y2:  # Movimiento horizontal
            if x1 > x2:
                x1, x2 = x2, x1
            for x in range(x1 + 1, x2):
                if (x, y1) in self.state.walls or (x, y1) in self.state.box_positions:
                    return False
        else:
            return False  # La distancia de Manhattan solo funciona en líneas rectas

        return True

#Calcula la distancia mínima del jugador a la caja, utilizando el método is_path_clear para verificar caminos libres.
class H1(Heuristics):
    def heuristic(self):
        min_distance = float('inf')
        for box in self.state.box_positions:
            if self.is_path_clear(self.state.player_pos, box):
                distance = abs(self.state.player_pos[0] - box[0]) + abs(self.state.player_pos[1] - box[1])
                min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 0

#Calcula la distancia mínima de cada caja a cualquier objetivo, utilizando el método is_path_clear.
class H2(Heuristics):
    def heuristic(self):
        total_distance = 0
        for box in self.state.box_positions:
            min_distance = float('inf')
            for goal in self.state.goal_positions:
                if self.is_path_clear(box, goal):
                    distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                    min_distance = min(min_distance, distance)
            total_distance += min_distance if min_distance != float('inf') else 0
        return total_distance

#Combina las heurísticas de H1 y H2 para proporcionar una heurística compuesta.
class H3(Heuristics):
    def heuristic(self):
        player_to_box = H1(self.state).heuristic()
        box_to_goal = H2(self.state).heuristic()
        return player_to_box + box_to_goal

#Igual que H3 pero se puede mover por obstáculos también
class H4(Heuristics):
    def heuristic(self):
        player_to_box = self._distance_to_nearest_box()
        box_to_goal = self._distance_from_box_to_goals()
        return player_to_box + box_to_goal

    def _distance_to_nearest_box(self):
        min_distance = float('inf')
        for box in self.state.box_positions:
            distance = abs(self.state.player_pos[0] - box[0]) + abs(self.state.player_pos[1] - box[1])
            min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 0

    def _distance_from_box_to_goals(self):
        total_distance = 0
        for box in self.state.box_positions:
            min_distance = float('inf')
            for goal in self.state.goal_positions:
                distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                min_distance = min(min_distance, distance)
            total_distance += min_distance if min_distance != float('inf') else 0
        return total_distance
