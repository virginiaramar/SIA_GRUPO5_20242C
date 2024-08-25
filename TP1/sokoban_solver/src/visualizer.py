import pygame
import time
import os

# Tamaño de cada celda
CELL_SIZE = 50

class Visualizer:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
        pygame.display.set_caption("Super Sokoban Bros.")
        self.delay = 0.5  # Delay predeterminado de 0.5 segundos

        # Obtener el directorio del script actual
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        images_dir = os.path.join(script_dir, 'images')

        # Cargar imágenes
        self.mario_img = self.load_image(os.path.join(images_dir, 'mario.png'))
        self.wall_img = self.load_image(os.path.join(images_dir, 'pared2.png'))
        self.question_img = self.load_image(os.path.join(images_dir, 'cajama.png'))
        self.star_img = self.load_image(os.path.join(images_dir, 'Estrella.png'))
        self.flag_img = self.load_image(os.path.join(images_dir, 'bandera2.png'))
        
        # Cargar y escalar la imagen de fondo
        background_path = os.path.join(images_dir, 'hierba2.jpg')
        self.background_img = pygame.image.load(background_path)
        self.background_img = pygame.transform.scale(self.background_img, (width * CELL_SIZE, height * CELL_SIZE))

    def load_image(self, filepath):
        try:
            img = pygame.image.load(filepath)
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except pygame.error as e:
            print(f"No se pudo cargar la imagen: {filepath}")
            print(f"Error: {e}")
            # Retorna una superficie en blanco si no se puede cargar la imagen
            return pygame.Surface((CELL_SIZE, CELL_SIZE))

    def draw_state(self, state):
        # Dibujar el fondo
        self.screen.blit(self.background_img, (0, 0))
        
        for y in range(state.height):
            for x in range(state.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (x, y) in state.walls:
                    self.screen.blit(self.wall_img, rect)
                elif (x, y) in state.targets:
                    self.screen.blit(self.question_img, rect)
                if (x, y) in state.boxes:
                    if (x, y) in state.targets:
                        self.screen.blit(self.flag_img, rect)
                    else:
                        self.screen.blit(self.star_img, rect)
                if (x, y) == state.player:
                    self.screen.blit(self.mario_img, rect)
        pygame.display.flip()

    def visualize_solution(self, solution):
        for state in solution:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            if state is None:
                print("Warning: Encountered None state in solution")
                continue
            self.draw_state(state)
            time.sleep(self.delay)
        pygame.quit()

    def set_delay(self, seconds):
        """Permite ajustar el retraso entre pasos."""
        self.delay = seconds