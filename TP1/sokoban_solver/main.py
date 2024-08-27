import json
import time
import csv
from src.game import load_level 
from src.search import bfs, dfs, astar, greedy, iddfs
from src.heuristics import *  # Importa todas las funciones del archivo heuristics.py
from src.visualizer import Visualizer

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def run_algorithm(algorithm, initial_state, heuristic_func=None):
    start_time = time.time()
    
    if algorithm in ['bfs', 'dfs', 'iddfs']:
        solution, nodes_expanded, frontier_size = globals()[algorithm](initial_state)
        heuristic_used = "N/A"
    else:  # astar or greedy
        solution, nodes_expanded, frontier_size = globals()[algorithm](initial_state, heuristic_func)
        heuristic_used = heuristic_func.__name__ if heuristic_func else "None"
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return solution, execution_time, nodes_expanded, frontier_size, heuristic_used

def main():
    config = load_config('config.json')
    level_file = config['level_file']
    algorithm = config['algorithm']
    heuristic_name = config.get('heuristic', 'simple_heuristic')
    print_results = config.get('print_results', True)
    print_time = config.get('print_time', 0)

    initial_state = load_level(level_file)
    
    # Asignar la función heurística basándose en el nombre
    heuristic_func = globals().get(heuristic_name)

    if heuristic_func is None and algorithm in ['astar', 'greedy']:
        raise ValueError(f"Heurística '{heuristic_name}' no encontrada. Verifica el archivo heuristics.py y la configuración.")
    
    solution, execution_time, nodes_expanded, frontier_size, heuristic_used = run_algorithm(algorithm, initial_state, heuristic_func)

    result = {
        "algorithm": algorithm,
        "level": level_file,
        "heuristic": heuristic_used,
        "execution_time": execution_time,
        "moves": len(solution) - 1 if solution else 0,
        "nodes_expanded": nodes_expanded,
        "frontier_size": frontier_size,
        "depth": len(solution) - 1 if solution else 0
    }

    if print_results and execution_time >= print_time:
        print(f"Results for {level_file}, {algorithm}:")
        for key, value in result.items():
            if key != "heuristic" or (algorithm in ['astar', 'greedy']):
                print(f"{key}: {value}")

    # Guardar resultados en CSV
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ["algorithm", "level", "heuristic", "execution_time", "moves", "nodes_expanded", "frontier_size", "depth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(result)

    if solution:
        print(f"Solución encontrada en {len(solution)} pasos.")
        for i, state in enumerate(solution):
            if state is None:
                print(f"Estado nulo encontrado en el paso {i}")
            else:
                print(f"Paso {i}:")
                print(state)
        visualizer = Visualizer(initial_state.width, initial_state.height)
        visualizer.set_delay(2.0)  # Establece un retraso de 2 segundos entre cada paso
        visualizer.visualize_solution(solution)
    else:
        print("No se encontró solución.")

if __name__ == "__main__":
    main()