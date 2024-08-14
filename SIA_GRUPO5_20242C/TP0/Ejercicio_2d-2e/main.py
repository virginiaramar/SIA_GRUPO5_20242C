import json
import sys
import csv

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        ball = config["pokeball"]
        # pokemon = factory.create(config["pokemon"], 1, StatusEffect.NONE, 1)

        # for i in range(100, 1, -1):
        #     pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, i / 100)
        #     print(pokemon.current_hp)
    successful_captures = 0


pokemon_name = config["pokemon"]  # Asumiendo que config["pokemon"] es el nombre del Pokémon

# Nombre del archivo CSV
csv_filename = f"{pokemon_name}_capture_results.csv"

# Abrimos el archivo CSV en modo escritura
with open(csv_filename, 'w', newline='') as csvfile:
    # Creamos el escritor CSV
    csvwriter = csv.writer(csvfile)
    
    # Escribimos el encabezado
    csvwriter.writerow([pokemon_name])
    csvwriter.writerow(["Level", "Success", "Capture Rate"])

    successful_captures = 0

    for i in range(100, 1, -1):
        pokemon = factory.create(config["pokemon"], i, StatusEffect.NONE, i/100)
        success, capture_rate = attempt_catch(pokemon, ball)
        if success:
            successful_captures += 1
        
        # Escribimos los resultados en el CSV
        csvwriter.writerow([pokemon.level, success, capture_rate])
        
        # Opcional: imprimir en consola también
        print(f"Level: {pokemon.level}")
        print(f"No noise: Success={success}, Capture Rate={capture_rate}")
        print("------")

        # Imprimimos el total de capturas exitosas al final del archivo
        print(f"Results saved to {csv_filename}")
     
        print(f"Total successful captures: {successful_captures}")

        
        # print("No noise: ", attempt_catch(pokemon, ball))
        #  for _ in range(10):
        #     print("Noisy: ", attempt_catch(pokemon, ball, 0.15))

#