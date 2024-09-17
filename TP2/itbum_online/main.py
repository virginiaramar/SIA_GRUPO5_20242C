import argparse
import time
import matplotlib.pyplot as plt
from src.configuration import Configuration
from src.genetic_algorithm import GeneticAlgorithm
import pygame
import sys
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg

def visualize_progress(history, character_class):
    if not history:
        print("No hay datos de generaciones para visualizar.")
        return

    generations = [g['generation'] for g in history]
    best_fitnesses = [g['best_fitness'] for g in history]
    avg_fitnesses = [g['average_fitness'] for g in history]

    plt.figure(figsize=(12, 6))
    if len(generations) == 1:
        plt.scatter(generations, best_fitnesses, label='Mejor Fitness', marker='o')
        plt.scatter(generations, avg_fitnesses, label='Fitness Promedio', marker='s')
    else:
        plt.plot(generations, best_fitnesses, label='Mejor Fitness', marker='o')
        plt.plot(generations, avg_fitnesses, label='Fitness Promedio', marker='s')
    
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    class_title = f" - Clase: {character_class}" if character_class is not None else ""
    plt.title(f'Progreso del Algoritmo Genético{class_title}')
    plt.legend()
    plt.grid(True)
    
    if len(generations) > 1:
        plt.xlim(left=min(generations), right=max(generations))
    plt.ylim(bottom=min(min(best_fitnesses), min(avg_fitnesses)) * 0.9, 
             top=max(max(best_fitnesses), max(avg_fitnesses)) * 1.1)
    
    plt.tight_layout()
    plt.show()

def print_generation_history(history):
    for gen in history:
        print(f"Generación {gen['generation']}: Mejor Fitness = {gen['best_fitness']:.4f}, Promedio = {gen['average_fitness']:.4f}")

def enhanced_pygame_visualize(ga):
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visualización del Algoritmo Genético - ITBUM ONLINE")
    clock = pygame.time.Clock()

    font_large = pygame.font.Font(None, 32)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)

    def character_to_color(character):
        r = max(0, min(255, int(character.items['strength'] * 2.55)))
        g = max(0, min(255, int(character.items['agility'] * 2.55)))
        b = max(0, min(255, int(character.items['expertise'] * 2.55)))
        return (r, g, b)

    class CharacterSprite:
        def __init__(self, character, x, y, index):
            self.character = character
            self.x = x
            self.y = y
            self.color = character_to_color(character)
            self.size = max(3, min(10, int(3 + character.get_performance())))
            self.index = index
            self.selected = False

        def draw(self, screen):
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
            if self.selected:
                pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.size + 1, 1)
            text = font_small.render(str(self.index), True, (255, 255, 255))
            screen.blit(text, (self.x - text.get_width() // 2, self.y - text.get_height() // 2))

    def create_sprite_grid(characters, start_x, start_y, cols):
        sprites = []
        for i, character in enumerate(characters):
            x = start_x + (i % cols) * 20
            y = start_y + (i // cols) * 20
            sprites.append(CharacterSprite(character, x, y, i))
        return sprites

    def draw_text(text, x, y, font=font_medium, color=(255, 255, 255)):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            rendered_text = font.render(line, True, color)
            screen.blit(rendered_text, (x, y + i * font.get_linesize()))

    def draw_fitness_graph(history):
        plt.figure(figsize=(6, 4))
        generations = [g['generation'] for g in history]
        best_fitnesses = [g['best_fitness'] for g in history]
        avg_fitnesses = [g['average_fitness'] for g in history]
        plt.plot(generations, best_fitnesses, label='Mejor')
        plt.plot(generations, avg_fitnesses, label='Promedio')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.tight_layout()
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, (WIDTH - size[0] - 20, 60))
        plt.close()

    population_sprites = create_sprite_grid(ga.initialize_population(), 20, 100, 20)
    generation = 0
    history = []

    running = True
    while running and generation < ga.stop_criteria['max_generations']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        screen.fill((0, 0, 0))

        draw_text(f"Generación {generation}", 20, 20, font_large)
        draw_text("Población:", 20, 70, font_medium)
        
        for sprite in population_sprites:
            sprite.draw(screen)

        parents = ga.parent_selection([sprite.character for sprite in population_sprites], ga.offspring_count)
        selected_indices = []
        for parent in parents:
            for i, sprite in enumerate(population_sprites):
                if sprite.character == parent:
                    sprite.selected = True
                    selected_indices.append(i)
                    break

        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = ga.crossover(parents[i], parents[i+1])
                offspring.extend([ga.mutate([child1])[0], ga.mutate([child2])[0]])
        
        if ga.config['replacement_method'] == 'young_bias':
            if len(offspring) > ga.population_size:
                new_population = ga.replacement_selection(offspring, ga.population_size)
                description = "Reemplazo: Sesgo Joven (mejores de descendencia)"
            else:
                remaining = ga.population_size - len(offspring)
                new_population = offspring + ga.replacement_selection([sprite.character for sprite in population_sprites], remaining)
                description = f"Reemplazo: Sesgo Joven ({len(offspring)} nuevos + {remaining} anteriores)"
        else:  # traditional
            combined = [sprite.character for sprite in population_sprites] + offspring
            new_population = ga.replacement_selection(combined, ga.population_size)
            description = "Reemplazo: Tradicional (mejores de padres e hijos)"

        population_sprites = create_sprite_grid(new_population, 20, 100, 20)

        best_fitness = max(sprite.character.get_performance() for sprite in population_sprites)
        avg_fitness = sum(sprite.character.get_performance() for sprite in population_sprites) / len(population_sprites)
        history.append({'generation': generation, 'best_fitness': best_fitness, 'average_fitness': avg_fitness})
        
        draw_text(f"Selección: {len(parents)} padres", 20, HEIGHT - 100, font_medium)
        draw_text(f"Índices: {', '.join(map(str, selected_indices[:10]))}{'...' if len(selected_indices) > 10 else ''}", 20, HEIGHT - 70, font_small)
        draw_text(description, 20, HEIGHT - 40, font_small)
        draw_text(f"Mejor Fitness: {best_fitness:.4f}", WIDTH - 250, HEIGHT - 60, font_medium)
        draw_text(f"Promedio: {avg_fitness:.4f}", WIDTH - 250, HEIGHT - 30, font_medium)

        draw_fitness_graph(history)

        pygame.display.flip()
        clock.tick(2)

        generation += 1

    pygame.quit()
    return max([sprite.character for sprite in population_sprites], key=lambda x: x.get_performance())

def main():
    parser = argparse.ArgumentParser(description='ITBUM ONLINE Character Optimizer')
    parser.add_argument('--history', action='store_true', help='Show generation history')
    parser.add_argument('--visualize', action='store_true', help='Visualize with Pygame')
    args = parser.parse_args()

    config = Configuration('config/config.json')
    ga_params = config.get_genetic_algorithm_params()
    ga = GeneticAlgorithm(ga_params)

    start_time = time.time()
    
    if args.visualize:
        best_character = enhanced_pygame_visualize(ga)
    else:
        best_character = ga.evolve()
    
    end_time = time.time()

    print(f"\nBest character found:")
    print(best_character)
    print(f"Performance: {best_character.get_performance():.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if args.history and not args.visualize:
        history = ga.get_generation_history()
        print(f"\nTotal generations in history: {len(history)}")
        print("\nGeneration History:")
        print_generation_history(history)
        visualize_progress(history, best_character.get_class_name())
        
if __name__ == "__main__":
    main()