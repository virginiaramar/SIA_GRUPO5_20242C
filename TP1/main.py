import csv
import json
import time
from STATE import load_board_from_file
from MÉTODOS.NO_INFORMADOS import BFS, DFS, IDDFS
from MÉTODOS.INFORMADOS import GREEDY
from MÉTODOS.INFORMADOS.heuristics import H1, H2, H3, H4, H5
from TP1.MÉTODOS.INFORMADOS import A_STAR

def run_test(level_file, algorithm, parameters):
    board = load_board_from_file(level_file)
    
    if algorithm == "DFS":
        moves, nodes_expanded, frontier_size, depth = DFS.run(board)
    elif algorithm == "BFS":
        moves, nodes_expanded, frontier_size, depth = BFS.run(board)
    elif algorithm == "IDDFS":
        depth_step = parameters.get('depth_step', 10)
        iddfs_instance = IDDFS.IDDFS(board, depth_step)
        moves, nodes_expanded, frontier_size, depth = iddfs_instance.search()
    elif algorithm == "A_STAR":
        heuristic_fn = globals().get(parameters.get('heuristic'))
        moves, nodes_expanded, frontier_size, depth = A_STAR.run(board, heuristic_fn)
    elif algorithm == "GREEDY":
        heuristic_fn = globals().get(parameters.get('heuristic'))
        moves, nodes_expanded, frontier_size, depth = GREEDY.run(board, heuristic_fn)
    
    return moves, nodes_expanded, frontier_size, depth

def main():
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    test = config["test"]
    level_file = test["level"]
    algorithm = test["algorithm"]
    parameters = test.get("parameters", {})
    heuristic = test.get("heuristic")
    print_results = test.get("print", False)
    print_time = test.get("print_time", 0)
    
    results = []
    
    for _ in range(20):
        start_time = time.time()
        moves, nodes_expanded, frontier_size, depth = run_test(level_file, algorithm, parameters)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        result = {
            "algorithm": algorithm,
            "level": level_file,
            "execution_time": execution_time,
            "moves": moves,
            "nodes_expanded": nodes_expanded,
            "frontier_size": frontier_size,
            "depth": depth
        }
        
        results.append(result)
        
        if print_results and execution_time >= print_time:
            print(f"Results for {level_file}, {algorithm}, {heuristic}:")
            print(result)
    
    # Guardar resultados en CSV
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ["algorithm", "level", "execution_time", "moves", "nodes_expanded", "frontier_size", "depth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
