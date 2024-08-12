import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect


if __name__ == "__main__":
    
    factory = PokemonFactory("pokemon.json")

    # Dataframe to store and analyse ALL the data 
    results_2b = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect','health_points', 'noise', 'capture_success', 'capture_rate'])

    # Dataframe to store the summary after the computation of the data
    summary_2b = pd.DataFrame(columns=['pokemon', 'pokeball', 'status_effect','health_points', 'noise', 'average_capture_rate', 'effectiveness', 'variance'])

    # Open the json with all the pokemons and pokeballs and assign to variables
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        pokemons = config["pokemons"]
        pokeballs = config["pokeballs"]
    
    # Create array with the different health conditions
    status_effects = [StatusEffect.NONE, StatusEffect.POISON, StatusEffect.BURN, StatusEffect.PARALYSIS, StatusEffect.SLEEP, StatusEffect.FREEZE]

    # Create loops for all the pokemons, pokeballs and health conditions
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

                        # Attempt to catch without noise
                        capture_success, capture_rate = attempt_catch(pokemon, pokeball_type)
                        capture_prob.append(capture_rate)

                        # If true, then count+1
                        if capture_success:
                            success_count += 1

                        # Add results to the datadrame
                        results_2b = results_2b.append({
                            'pokemon': pokemon_name,
                            'pokeball': pokeball_type,
                            'status_effect': status.name,
                            'health_points': i,
                            'noise': 0.0,
                            'capture_success': capture_success,
                            'capture_rate': capture_rate
                        }, ignore_index=True)
                
                    # Calculate mean, effectiveness and variance
                    avg_capture_rate = np.mean(capture_prob)
                    effectiveness = success_count / 10
                    variance = np.var(capture_prob)

                    # Add summary of the data to the dataframe
                    summary_2b = summary_2b.append({
                        'pokemon': pokemon_name,
                        'pokeball': pokeball_type,
                        'status_effect': status.name,
                        'health_points': i,
                        'noise': 0.0,
                        'average_capture_rate': avg_capture_rate,
                        'effectiveness': effectiveness,
                        'variance': variance
                    }, ignore_index=True)

    # Add each result to a csv
    results_2b.to_csv("capture_results_2b.csv", index=False)

    # Add each result to a summary csv
    summary_2b.to_csv("capture_summary_2b.csv", index=False)

    # Convertir el resumen final a formato JSON para ver mejor
    summary_json = summary_2b.to_dict(orient='records')
    with open("capture_summary_2b.json", "w") as jsonfile:
        json.dump(summary_json, jsonfile, indent=4)

    # STATEMENT DE FINALIZACION
    print("Datos guardados en 'capture_results.csv', 'capture_summary.csv', y 'capture_summary.json'.")
    # Graficar los resultados para un Pokémon específico (por ejemplo, Snorlax)
    pokemon_to_plot = "Snorlax"
    filtered_data = summary_2b[(summary_2b['pokemon'] == pokemon_to_plot)]
    
    plt.figure(figsize=(12, 6))
    for status_effect in filtered_data['status_effect'].unique():
        subset = filtered_data[filtered_data['status_effect'] == status_effect]
        plt.plot(subset['health_points'], subset['average_capture_rate'], label=status_effect)
        plt.fill_between(subset['health_points'], subset['average_capture_rate'] - np.sqrt(subset['variance']),
                         subset['average_capture_rate'] + np.sqrt(subset['variance']), alpha=0.2)

    plt.title(f'Evolución de la Tasa de Captura para {pokemon_to_plot}')
    plt.xlabel('Puntos de Salud')
    plt.ylabel('Promedio de Tasa de Captura')
    plt.legend(title='Estado')
    plt.show()

 
