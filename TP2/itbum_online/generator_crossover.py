import json
import os

def generate_crossover_variants():
    # Fixed base configuration values
    base_config = {
        "genetic_algorithm": {
            "population_size": 100,
            "offspring_count": 50,
            "mutation": {
                "type": "multigen",
                "rate": 0.02,
                "uniform": True
            },
            "replacement_method": "young_bias",
            "stop_criteria": {
                "max_generations": 100,
                "structure": 0.05,
                "content": 0.05,
                "optimal_fitness": 100
            },
            "character_class": 0,
            "total_points": 200,
            "time_limit": 120
        }
    }
    
    # Fixed selection methods
    selection_config = {
        "parents": {
            "method1": "universal",
            "method2": "ranking",
            "method1_proportion": 0.5
        },
        "replacement": {
            "method1": "roulette",
            "method2": "elite",
            "method1_proportion": 0.5
        },
        "tournament": {
            "type": "placeholder",
            "m": 0,
            "threshold": 0.0
        },
        "boltzmann": {
            "Tmin": 0.1,
            "Tmax": 3.0,
            "k": 0.1
        }
    }

    character_class = base_config['genetic_algorithm']['character_class']
    # Crossover methods and rates
    crossover_methods = ['one_point', 'two_point', 'uniform', 'anular']
    crossover_rate_range = [round(i * 0.1, 1) for i in range(11)]  # 0.0 to 1.0 in steps of 0.1

    # Create 'crossover_variants' directory if it doesn't exist
    os.makedirs('config/hibrido/crossover', exist_ok=True)
    file_counter = 1

    # Generate configurations for each combination of crossover method and rate
    for method in crossover_methods:
        for rate in crossover_rate_range:
            config = json.loads(json.dumps(base_config))  # Deep copy of the base configuration
            config['genetic_algorithm']['selection'] = selection_config  # Add the fixed selection methods
            config['genetic_algorithm']['crossover'] = {
                "type": method,
                "rate": rate
            }

            # Constructing the filename
            filename = f"C_{character_class}_crossover_{method}_rate_{str(rate).replace('.', '')}"
            config['outputName'] = filename

            # Save to a JSON file
            with open(f"config/hibrido/crossover/{filename}.json", "w") as json_file:
                json.dump(config, json_file, indent=4)
            file_counter += 1

    print(f"Generated {file_counter - 1} crossover variant configurations in the 'crossover_variants' folder.")

if __name__ == "__main__":
    generate_crossover_variants()
