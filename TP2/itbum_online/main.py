import time
from src.configuration import Configuration
from src.genetic_algorithm import GeneticAlgorithm

def main():
    config = Configuration('config/config.json')
    ga_params = config.get_genetic_algorithm_params()
    time_limit = config.get_time_limit()

    ga = GeneticAlgorithm(ga_params)

    start_time = time.time()
    best_character = ga.evolve()
    end_time = time.time()

    print(f"Best character found:")
    print(best_character)
    print(f"Performance: {best_character.get_performance()}")
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()