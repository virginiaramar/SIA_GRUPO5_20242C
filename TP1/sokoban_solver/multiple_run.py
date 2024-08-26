import json
import subprocess
import os
from datetime import datetime

# Obtener el directorio del script actual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def update_config(level, algorithm, heuristic=None):
    config_path = os.path.join(SCRIPT_DIR, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"El archivo config.json no se encuentra en {SCRIPT_DIR}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['level_file'] = os.path.join('levels', f'{level}.txt')
    config['algorithm'] = algorithm
    if heuristic:
        config['heuristic'] = heuristic
    elif 'heuristic' in config:
        del config['heuristic']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def run_main(level, algorithm, heuristic=None):
    update_config(level, algorithm, heuristic)
    main_py_path = os.path.join(SCRIPT_DIR, 'main.py')
    if not os.path.exists(main_py_path):
        raise FileNotFoundError(f"El archivo main.py no se encuentra en {SCRIPT_DIR}")
    
    print(f"Ejecutando: python {main_py_path}")
    subprocess.run(['python', main_py_path], cwd=SCRIPT_DIR, check=True)

def main():
    levels = ['level1', 'level2', 'level3','impossible']
    algorithms = ['bfs', 'dfs', 'iddfs', 'greedy', 'astar']
    heuristics = ['h1_heuristic', 'h2_heuristic', 'h3_heuristic', 'h4_heuristic']
    iterations = 10

    for level in levels:
        for algorithm in algorithms:
            if algorithm in ['bfs', 'dfs', 'iddfs']:
                for i in range(iterations):
                    run_main(level, algorithm)
                    print(f"Completado: {level}, {algorithm}, iteración {i+1}")
            elif algorithm in ['greedy', 'astar']:
                for heuristic in heuristics:
                    for i in range(iterations):
                        run_main(level, algorithm, heuristic)
                        print(f"Completado: {level}, {algorithm}, {heuristic}, iteración {i+1}")

    print(f"Resultados guardados en results.csv")

if __name__ == "__main__":
    main()




