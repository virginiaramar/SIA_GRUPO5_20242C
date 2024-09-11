import argparse
import time
import matplotlib.pyplot as plt
from src.configuration import Configuration
from src.genetic_algorithm import GeneticAlgorithm


def visualize_progress(history, character_class):
    if not history:
        print("No hay datos de generaciones para visualizar.")
        return

    generations = [g['generation'] for g in history]
    best_fitnesses = [g['best_fitness'] for g in history]
    avg_fitnesses = [g['average_fitness'] for g in history]

    print("Generations:", generations)
    print("Best Fitnesses:", best_fitnesses)
    print("Avg Fitnesses:", avg_fitnesses)

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

def main():
    parser = argparse.ArgumentParser(description='ITBUM ONLINE Character Optimizer')
    parser.add_argument('--history', action='store_true', help='Show generation history')
    args = parser.parse_args()

    config = Configuration('config/config.json')
    ga_params = config.get_genetic_algorithm_params()
    ga = GeneticAlgorithm(ga_params)

    start_time = time.time()
    best_character = ga.evolve()
    end_time = time.time()

    print(f"\nBest character found:")
    print(best_character)
    print(f"Performance: {best_character.get_performance():.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if args.history:
        history = ga.get_generation_history()
        print(f"\nTotal generations in history: {len(history)}")
        print("\nGeneration History:")
        print_generation_history(history)
        visualize_progress(history, best_character.get_class_name())
        
if __name__ == "__main__":
    main()