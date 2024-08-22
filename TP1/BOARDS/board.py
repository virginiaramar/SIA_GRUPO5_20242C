# board.py

class Board:
    def __init__(self, level):
        self.level = level
        self.board = [list(row) for row in level]  # Convertir a una lista de listas para mutabilidad

    def display(self):
        for row in self.board:
            print("".join(row))
