import json
import pandas as pd
import numpy as np

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

if __name__ == "__main__":
    
    factory = PokemonFactory("pokemon.json")

    # Dataframe to store and analyse ALL the data 
    results_2b_noise = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect','health_points', 'noise', 'capture_success', 'capture_rate'])

    # Dataframe to store the summary after the computation of the data
    summary_2b_noise = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect','health_points', 'noise', 'average_capture_rate', 'effectiveness', 'variance'])

    # CData that will be used
    status_effects = [StatusEffect.NONE]
    pokemons = ["caterpie", "mewtwo"]
    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
    
    # Create loops for the pokemons, pokeballs and health condition
    for pokemon_name in pokemons:
        print(pokemon_name)
        for pokeball_type in pokeballs:
            print(pokeball_type)
            for status in status_effects:
                print(status)
                # Count for the effectiveness and array for probability
                success_count = 0
                capture_prob = []

                # It lowers the health from 100 to 1
                for i in range(100, 1, -1):
                    for _ in range(10):
                        # Create the pokemon with 100 level and different health
                        pokemon = factory.create(pokemon_name, 100, status, i/100)

                        # Attempt to catch with noise
                        capture_success, capture_rate = attempt_catch(pokemon, pokeball_type, 0.15)
                        capture_prob.append(capture_rate)

                        # If true, then count+1
                        if capture_success:
                            success_count += 1

                        # Add results to the dataframe
                        results_2b_noise = pd.concat([results_2b_noise, pd.DataFrame([{
                            'pokemon': pokemon_name,
                            'pokeball': pokeball_type,
                            'status_effect': status.name,
                            'health_points': i,
                            'noise': 0.0,
                            'capture_success': capture_success,
                            'capture_rate': capture_rate
                        }])], ignore_index=True)
                
                    # Calculate mean, effectiveness and variance
                    avg_capture_rate = np.mean(capture_prob)
                    effectiveness = success_count / 10
                    variance = np.var(capture_prob)

                    # Add summary of the data to the dataframe
                    summary_2b_noise = pd.concat([summary_2b_noise, pd.DataFrame([{
                        'pokemon': pokemon_name,
                        'pokeball': pokeball_type,
                        'status_effect': status.name,
                        'health_points': i,
                        'noise': 0.0,
                        'average_capture_rate': avg_capture_rate,
                        'effectiveness': effectiveness,
                        'variance': variance
                    }])], ignore_index=True)

    # Save each result to a CSV
    results_2b_noise.to_csv("capture_results_2b_noise.csv", index=False)

    # Save each result to a summary CSV
    summary_2b_noise.to_csv("capture_summary_2b_noise.csv", index=False)

    # Convert the final summary to JSON for better viewing
    summary_json_2b_noise = summary_2b_noise.to_dict(orient='records')
    with open("capture_summary_2b_noise.json", "w") as jsonfile:
        json.dump(summary_json_2b_noise, jsonfile, indent=4)


 
