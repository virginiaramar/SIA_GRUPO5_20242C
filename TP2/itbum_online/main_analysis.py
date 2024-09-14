import argparse
import time
import matplotlib.pyplot as plt
import csv
from src.configuration import Configuration
from src.genetic_algorithm import GeneticAlgorithm

def visualize_progress(history, character_class, sim_num):
    if not history:
        print(f"No hay datos de generaciones para visualizar para la simulación {sim_num}.")
        return

    generations = [g['generation'] for g in history]
    best_fitnesses = [g['best_fitness'] for g in history]
    avg_fitnesses = [g['average_fitness'] for g in history]

    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitnesses, label=f'Mejor Fitness Sim {sim_num}', marker='o')
    plt.plot(generations, avg_fitnesses, label=f'Fitness Promedio Sim {sim_num}', marker='s')
    
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Progreso del Algoritmo Genético - Simulación {sim_num}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()

def save_all_simulations_to_csv(history, best_character, filename, sim_num, append=False):
    try:
        mode = 'a' if append else 'w'  # Append if it's not the first simulation
        with open(filename, mode, newline='') as file:
            writer = csv.writer(file, delimiter=',')
            
            if not append:  # Write the header for the first simulation
                writer.writerow(['Simulation', 'Generation', 'Best Fitness', 'Average Fitness', 'Class', 'Strength', 'Agility', 'Expertise', 'Endurance', 'Health', 'Height', 'Total Points', 'Performance'])

            for gen in history:
                # Write the generation history and best character attributes
                character_items = best_character.items
                writer.writerow([
                    sim_num, 
                    gen['generation'], 
                    f"{gen['best_fitness']:.4f}", 
                    f"{gen['average_fitness']:.4f}",
                    best_character.get_class_name(),
                    f"{character_items['strength']:.4f}",
                    f"{character_items['agility']:.4f}",
                    f"{character_items['expertise']:.4f}",
                    f"{character_items['endurance']:.4f}",
                    f"{character_items['health']:.4f}",
                    f"{best_character.height:.4f}",
                    best_character.total_points,
                    f"{best_character.get_performance():.4f}"
                ])
            
        print(f"Simulation {sim_num} data saved to {filename}")
    except Exception as e:
        print(f"Error saving data for simulation {sim_num}: {e}")

def main():
    parser = argparse.ArgumentParser(description='ITBUM ONLINE Character Optimizer - Multiple Simulations')
    parser.add_argument('--history', action='store_true', help='Show generation history for each simulation')
    parser.add_argument('--save_csv', type=str, help='Filename to save all simulation data as CSV')
    parser.add_argument('--simulations', type=int, default=1, help='Number of simulations to run')
    args = parser.parse_args()

    config = Configuration('config/config.json')
    ga_params = config.get_genetic_algorithm_params()

    for sim_num in range(1, args.simulations + 1):
        print(f"\nRunning simulation {sim_num}...\n")
        ga = GeneticAlgorithm(ga_params)

        start_time = time.time()
        best_character = ga.evolve()
        end_time = time.time()

        print(f"\nBest character found in simulation {sim_num}:")
        print(best_character)
        print(f"Performance: {best_character.get_performance():.4f}")
        print(f"Time taken for simulation {sim_num}: {end_time - start_time:.2f} seconds")

        if args.history:
            history = ga.get_generation_history()
            print(f"\nTotal generations in history for simulation {sim_num}: {len(history)}")
            print("\nGeneration History:")
            if args.save_csv:
                save_all_simulations_to_csv(history, best_character, args.save_csv, sim_num, append=(sim_num > 1))
            visualize_progress(history, best_character.get_class_name(), sim_num)

if __name__ == "__main__":
    main()