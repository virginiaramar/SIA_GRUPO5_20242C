import pygame
from pygame.locals import *

class SokobanGame:
    def __init__(self, board):
        pygame.init()
        
        # Dimensiones de la ventana basadas en el tamaño del tablero
        self.cell_size = 40  # Tamaño de cada celda en píxeles
        self.rows = len(board)
        self.cols = len(board[0])
        window_width = self.cols * self.cell_size
        window_height = self.rows * self.cell_size

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Sokoban')
        self.clock = pygame.time.Clock()
        self.board = board

        # Cargar imágenes
        self.images = {
            '#': pygame.image.load('JUEGO_V/IMAGES/pared2.png'),
            '@': pygame.image.load('JUEGO_V/IMAGES/mario.png'),
            '$': pygame.image.load('JUEGO_V/IMAGES/cajama.png'),
            '.': pygame.image.load('JUEGO_V/IMAGES/bandera2.png'),
            '*': pygame.image.load('JUEGO_V/IMAGES/BOX_ON_GOAL.png'),
            ' ': None  # Espacio vacío, no requiere imagen
        }

        # Cargar imagen de fondo
        self.background_image = pygame.image.load('JUEGO_V/IMAGES/hierba2.jpg')
        self.background_image = pygame.transform.scale(self.background_image, (window_width, window_height))

        # Redimensionar imágenes para ajustarlas al tamaño de la celda
        self.images = {
            key: pygame.transform.scale(image, (self.cell_size, self.cell_size))
            for key, image in self.images.items() if image
        }

    def draw_board(self):
        # Dibujar la imagen de fondo
        self.screen.blit(self.background_image, (0, 0))

        # Dibujar el tablero
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
    board = load_board_from_file('BOARDS/LEVELS/impossible.txt')
    game = SokobanGame(board)
    game.run()
