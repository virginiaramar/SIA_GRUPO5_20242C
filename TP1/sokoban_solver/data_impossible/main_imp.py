import json
import subprocess
import os
import csv

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
    main_py_path = os.path.join(SCRIPT_DIR, 'main_impb.py')
    if not os.path.exists(main_py_path):
        raise FileNotFoundError(f"El archivo main.py no se encuentra en {SCRIPT_DIR}")
    
    print(f"Ejecutando: python {main_py_path}")
    result = subprocess.run(['python', main_py_path], cwd=SCRIPT_DIR, check=True, capture_output=True, text=True)
    return result.stdout

def save_to_csv(data, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['level', 'algorithm', 'heuristic', 'execution_time'])  # Cabecera del CSV
        for row in data:
            writer.writerow(row)

def main():
    levels = ['impossible']
    algorithms = ['bfs', 'dfs', 'iddfs', 'greedy', 'astar']
    heuristics = ['h1_heuristic', 'h2_heuristic', 'h3_heuristic', 'h4_heuristic']
    iterations = 10
    results = []  # Lista para almacenar los resultados

    for level in levels:
        for algorithm in algorithms:
            if algorithm in ['bfs', 'dfs', 'iddfs']:
                for i in range(iterations):
                    output = run_main(level, algorithm)
                    # Supongamos que el tiempo de ejecución se encuentra en la salida de `main.py`
                    execution_time = parse_execution_time(output)
                    results.append([level, algorithm, 'N/A', execution_time])
                    print(f"Completado: {level}, {algorithm}, iteración {i+1}")
            elif algorithm in ['greedy', 'astar']:
                for heuristic in heuristics:
                    for i in range(iterations):
                        output = run_main(level, algorithm, heuristic)
                        execution_time = parse_execution_time(output)
                        results.append([level, algorithm, heuristic, execution_time])
                        print(f"Completado: {level}, {algorithm}, {heuristic}, iteración {i+1}")

    # Guardar resultados en 'IMPOSSIBLE_results.csv'
    output_file = 'IMPOSSIBLE_results.csv'
    save_to_csv(results, output_file)
    print(f"Resultados guardados en {output_file}")

def parse_execution_time(output):
    # Esta función debería analizar la salida del script `main.py`
    # y extraer el tiempo de ejecución real.
    # Ejemplo: si el tiempo está formateado en la salida como "Execution time: X seconds",
    # se podría usar una expresión regular o una búsqueda de string para obtener X.
    execution_time_line = [line for line in output.split('\n') if 'Execution time:' in line]
    if execution_time_line:
        execution_time = execution_time_line[0].split(':')[-1].strip()
        return float(execution_time)
    return 0.0

if __name__ == "__main__":
    main()