import json
import os

def generate_replacement_variants():
    # Fixed base configuration values
    base_config = {
        "genetic_algorithm": {
            "population_size": 100,
            "offspring_count": 50,
            "crossover": {
                "type": "one_point",
                "rate": 0.3
            },
            "mutation": {
                "type": "gen",
                "rate": 0.01,
                "uniform": True
            },
            "stop_criteria": {
                "max_generations": 300,
                "structure": 0.01,
                "content": 0.01,
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
            "method1": "elite",
            "method2": "roulette",
            "method1_proportion": 0.5
        },
        "replacement": {
            "method1": "elite",
            "method2": "roulette",
            "method1_proportion": 0.5
        },
        "tournament": {
            "type": "placeholder",
            "m": 0,
            "threshold": 0.0
        },
        "boltzmann": {
            "Tmin": 0.0,
            "Tmax": 0.0,
            "k": 0.0
        }
    }

    # Replacement methods
    replacement_methods = ['traditional', 'young_bias']

    # Create 'replacement_variants' directory if it doesn't exist
    os.makedirs('config/hibrido/replacement', exist_ok=True)
    file_counter = 1
    
    character_class = base_config['genetic_algorithm']['character_class']

    # Generate configurations for each replacement method
    for method in replacement_methods:
        config = json.loads(json.dumps(base_config))  # Deep copy of the base configuration
        config['genetic_algorithm']['selection'] = selection_config  # Add the fixed selection methods
        config['genetic_algorithm']['replacement_method'] = method
        

        # Constructing the filename
        filename = f"C_{character_class}_replacement_{method}"
        config['outputName'] = filename

        # Save to a JSON file
        with open(f"config/hibrido/replacement/{filename}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        file_counter += 1

    print(f"Generated {file_counter - 1} replacement variant configurations in the 'replacement_variants' folder.")

if __name__ == "__main__":
    generate_replacement_variants()
