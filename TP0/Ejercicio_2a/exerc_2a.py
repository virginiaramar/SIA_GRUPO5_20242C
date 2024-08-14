import json
import sys
import pandas as pd
import numpy as np


from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

# To run: python exerc_2a.py configs/alltypes.json

if __name__ == "__main__":
    
    factory = PokemonFactory("pokemon.json")

    # Dataframe to store and analyse ALL the data 
    results_2a = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect', 'noise', 'capture_success', 'capture_rate'])

    # Dataframe to store the summary after the computation of the data
    summary_2a = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect', 'noise', 'average_capture_rate', 'effectiveness', 'variance'])

    # Open the json with all the pokemons and pokeballs and assign to variables
    # To run: python exerc_2a.py configs/alltypes.json
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        pokemons = config["pokemons"]
        pokeballs = config["pokeballs"]
    
    # Create array with the different health conditions
    status_effects = [StatusEffect.NONE, StatusEffect.POISON, StatusEffect.BURN, StatusEffect.PARALYSIS, StatusEffect.SLEEP, StatusEffect.FREEZE]

    # Create loops for all the pokemons, pokeballs and health conditions
    for pokemon_name in pokemons:
        for pokeball_type in pokeballs:
            for status in status_effects:

                # Count for the effectiveness and array for probability
                success_count = 0
                capture_prob = []

                #   Run for each type 100 times
                for _ in range(100):

                    # Create the pokemon with 100 level and 100% health
                    pokemon = factory.create(pokemon_name, 100, status, 1.0)

                    # Attempt to catch without noise
                    capture_success, capture_rate = attempt_catch(pokemon, pokeball_type)
                    capture_prob.append(capture_rate)

                    # If true, then count+1
                    if capture_success:
                        success_count += 1

                    # Add results to the datadrame
                    results_2a = results_2a.append({
                        'pokemon': pokemon_name,
                        'pokeball': pokeball_type,
                        'status_effect': status.name,
                        'noise': 0.0,
                        'capture_success': capture_success,
                        'capture_rate': capture_rate
                    }, ignore_index=True)
                
                # Calculate mean, effectiveness and variance
                avg_capture_rate = np.mean(capture_prob)
                effectiveness = success_count / 100
                variance = np.var(capture_prob)

                # Add summary of the data to the dataframe
                summary_2a = summary_2a.append({
                    'pokemon': pokemon_name,
                    'pokeball': pokeball_type,
                    'status_effect': status.name,
                    'noise': 0.0,
                    'average_capture_rate': avg_capture_rate,
                    'effectiveness': effectiveness,
                    'variance': variance
                }, ignore_index=True)

    # Add each result to a csv
    results_2a.to_csv("capture_results.csv", index=False)

    # Add each result to a summary csv
    summary_2a.to_csv("capture_summary.csv", index=False)

    
    summary_json = summary_2a.to_dict(orient='records')
    with open("capture_summary.json", "w") as jsonfile:
        json.dump(summary_json, jsonfile, indent=4)

 

 
