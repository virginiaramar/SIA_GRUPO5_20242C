import json
import os

def generate_hybrid_selection_configs():
    # Fixed values
    base_config = {
        "genetic_algorithm": {
            "population_size": 100,
            "offspring_count": 50,
            "crossover": {
                "type": "two_point",
                "rate": 0.8
            },
            "mutation": {
                "type": "multigen",
                "rate": 0.02,
                "uniform": True
            },
            "replacement_method": "young_bias",
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
    
    # Selection methods
    methods = ['tournament', 'roulette', 'universal', 'boltzmann', 'ranking', 'elite']
    
    # Tournament and Boltzmann settings
    tournament_types = ['deterministic', 'probabilistic']
    boltzmann_settings = {
        "Tmin": 0.5,
        "Tmax": 2.0,
        "k": 0.1
    }
    
    # Create 'generations' directory if it doesn't exist
    os.makedirs('config/hibrido/selection', exist_ok=True)
    file_counter = 1

    # Generate configurations for each combination of selection methods for parents and replacements
    for parent_method1 in methods:
        for parent_method2 in methods:
            for replacement_method1 in methods:
                for replacement_method2 in methods:
                    # Create a base configuration for each combination
                    config = json.loads(json.dumps(base_config))  # Deep copy
                    config['genetic_algorithm']['selection'] = {
                        "parents": {
                            "method1": parent_method1,
                            "method2": parent_method2,
                            "method1_proportion": 0.2
                        },
                        "replacement": {
                            "method1": replacement_method1,
                            "method2": replacement_method2,
                            "method1_proportion": 0.3
                        },
                        "tournament": {
                            "type": "placeholder",  # Placeholder for tournament
                            "m": 0,
                            "threshold": 0.0
                        },
                        "boltzmann":{
                            "Tmin": 0.5,
                            "Tmax": 2.0,
                            "k": 0.1
                        }
                    }

                    # Base filename for the configuration
                    character_class = base_config['genetic_algorithm']['character_class']
                    base_filename = f"C_{character_class}_P_{parent_method1}_{parent_method2}_R_{replacement_method1}_{replacement_method2}"

                    # Specific settings for tournaments
                    if 'tournament' in [parent_method1, parent_method2, replacement_method1, replacement_method2]:
                        for t_type in tournament_types:
                            config_copy = json.loads(json.dumps(config))  # Deep copy
                            config_copy['genetic_algorithm']['selection']['tournament'] = {
                                "type": t_type,
                                "m": 5,
                                "threshold": 0.75
                            }
                            filename = f"{base_filename}_T_{t_type}"
                            config_copy['outputName'] = filename
                            # Save to a JSON file
                            with open(f"config/hibrido/selection/{filename}.json", "w") as json_file:
                                json.dump(config_copy, json_file, indent=4)
                            file_counter += 1
                    # Specific settings for boltzmann
                    elif 'boltzmann' in [parent_method1, parent_method2, replacement_method1, replacement_method2]:
                        config['genetic_algorithm']['selection']['boltzmann'] = boltzmann_settings
                        filename = base_filename
                        config['outputName'] = filename
                        # Save to a JSON file
                        with open(f"config/hibrido/selection/{filename}.json", "w") as json_file:
                            json.dump(config, json_file, indent=4)
                        file_counter += 1
                    else:
                        filename = base_filename
                        config['outputName'] = filename
                        # Save to a JSON file
                        with open(f"config/hibrido/selection/{filename}.json", "w") as json_file:
                            json.dump(config, json_file, indent=4)
                        file_counter += 1

    print(f"Generated {file_counter - 1} hybrid selection configurations in the 'generations' folder.")

if __name__ == "__main__":
    generate_hybrid_selection_configs()
