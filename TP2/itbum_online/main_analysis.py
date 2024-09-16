import argparse
import time
import matplotlib.pyplot as plt
import csv
import os
import json
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

def save_all_simulations_to_csv(history, best_character, filename, sim_num, time_taken, stop_reason, append=False):
    try:
        mode = 'a' if append else 'w'  # Append if it's not the first simulation
        with open(filename, mode, newline='') as file:
            writer = csv.writer(file, delimiter=',')
            
            if not append:  # Write the header for the first simulation
                writer.writerow(['Simulation', 'Generation', 'Best Fitness', 'Average Fitness', 'Class', 'Strength', 
                                 'Agility', 'Expertise', 'Endurance', 'Health', 'Height', 'Total Points', 
                                 'Performance', 'v_strength', 'v_agility', 'v_expertise', 'v_endurance', 
                                 'v_health', 'v_height', 'Time Taken', 'Stop Reason'])

            for gen in history:
                # Write the generation history and best character attributes
                character_items = gen['best_character'].items
                variance_history = gen['variance_attrib']
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
                    f"{gen['best_character'].height:.4f}",
                    best_character.total_points,
                    f"{best_character.get_performance():.4f}",
                    f"{variance_history['strength']:.4f}", 
                    f"{variance_history['agility']:.4f}", 
                    f"{variance_history['expertise']:.4f}", 
                    f"{variance_history['endurance']:.4f}", 
                    f"{variance_history['health']:.4f}", 
                    f"{variance_history['height']:.4f}",
                    f"{time_taken:.4f}",  # Add time taken for this simulation
                    stop_reason  # Add stop reason
                ])
            
        print(f"Simulation {sim_num} data saved to {filename}")
    except Exception as e:
        print(f"Error saving data for simulation {sim_num}: {e}")

def main():
    parser = argparse.ArgumentParser(description='ITBUM ONLINE Character Optimizer - Multiple Simulations')
    parser.add_argument('--history', action='store_true', help='Show generation history for each simulation')
    parser.add_argument('--simulations', type=int, default=1, help='Number of simulations to run')
    args = parser.parse_args()

    # YOU MUST CHANGE THE DIRECTORY WITH THE PATH FOR THE RUN
    folder_to_analize = 'hibrido/selection'

    # Define the config directory
    config_folder = 'config/' + folder_to_analize

    # Define the output directory
    output_dir = 'output/' + folder_to_analize
    
    # Traverse the config folder including subfolders
    for root, dirs, files in os.walk(config_folder):
        for file in files:
            if file.endswith('.json'):
                config_path = os.path.join(root, file)
                config = Configuration(config_path)
                ga_params = config.get_genetic_algorithm_params()

                # Read the outputName from the config file
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    output_name = config_data.get('outputName', 'simulation_output')

                # Determine the relative path to the config file and replicate it in the output directory
                relative_path = os.path.relpath(root, config_folder)
                output_subdir = os.path.join(output_dir, relative_path)
                
                # Create the output subdirectory if it doesn't exist
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Prepare the output CSV filename using the outputName from the config
                csv_filename = os.path.join(output_subdir, f"{output_name}.csv")

                for sim_num in range(1, args.simulations + 1):
                    ga = GeneticAlgorithm(ga_params)

                    start_time = time.time()
                    best_character, stop_reason = ga.evolve()
                    end_time = time.time()

                    time_taken = end_time - start_time

                    if args.history:
                        history = ga.get_generation_history()
                        # Save the history and best character to the CSV file
                        save_all_simulations_to_csv(history, best_character, csv_filename, sim_num, time_taken, stop_reason, append=(sim_num > 1))
                        # visualize_progress(history, best_character.get_class_name(), sim_num)

                    print(f"Simulation {sim_num} finished: {stop_reason}")

if __name__ == "__main__":
    main()
