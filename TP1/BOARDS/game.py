# Run python BOARDS\game.py y cambiar el nombre de los levels

import pygame
from pygame.locals import *

class SokobanGame:
    def __init__(self, board):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('Sokoban')
        self.clock = pygame.time.Clock()
        self.board = board
        self.cell_size = 40  # Tamaño de cada celda en píxeles
        self.rows = len(board)
        self.cols = len(board[0])

        # Cargar imágenes
        self.images = {
            '#': pygame.image.load('BOARDS/IMAGES/WALL.png'),
            '@': pygame.image.load('BOARDS/IMAGES/PLAYER.png'),
            '$': pygame.image.load('BOARDS/IMAGES/BOX.png'),
            '.': pygame.image.load('BOARDS/IMAGES/GOAL.png'),
            '*': pygame.image.load('BOARDS/IMAGES/BOX_ON_GOAL.png'),
            ' ': None  # Espacio vacío, no requiere imagen
        }

        # Redimensionar imágenes para ajustarlas al tamaño de la celda
        self.images = {
            key: pygame.transform.scale(image, (self.cell_size, self.cell_size))
            for key, image in self.images.items() if image
        }

    def draw_board(self):
        # Rellena la pantalla con color marrón claro
        self.screen.fill((227, 212, 195))  # Color de fondo del tablero

        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                image = self.images.get(char)
                if image:
                    self.screen.blit(image, (x * self.cell_size, y * self.cell_size))
        
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

    def run(self):
        while True:
            self.handle_events()
            self.draw_board()
            self.clock.tick(30)  # Limita el juego a 30 FPS

def load_board_from_file(filename):
    with open(filename, 'r') as file:
        return [line.rstrip() for line in file.readlines()]

if __name__ == "__main__":
    # Cargar el tablero desde el archivo
    board = load_board_from_file('BOARDS/LEVELS/medium.txt')
    game = SokobanGame(board)
    game.run()
