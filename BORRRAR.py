
# json para leer y manejar los json
# sys para acceder nombres de los archivos de configs
import json
import sys

#El main al no estar en src necesitamos importar desde esa carpeta todas las funciones que vamos a utilizar
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    #esta manera de open es para abrir archivos, en este caso es 1 porque es el segundo argumento detras de python cuando lo corres
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        ball = config["pokeball"]
        pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)
        pokemonB = factory.create(config["pokemon"], 100, StatusEffect.SLEEP, 1)
        # for i in range(100, 1, -1):
        #     pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, i / 100)
        #     print(pokemon.current_hp)

        print("Pokemon A")
        print("No noise: ", attempt_catch(pokemon, ball))
        # se pone _ porque no necesitamos ese parametros como por ejempli i o j, entonces ponemos eso
        for _ in range(10):
            print("Noisy: ", attempt_catch(pokemon, ball, 0.15))

        print("Pokemon B")
        print("No noise: ", attempt_catch(pokemonB, ball))
        # se pone _ porque no necesitamos ese parametros como por ejempli i o j, entonces ponemos eso
        for _ in range(10):
            print("Noisy: ", attempt_catch(pokemonB, ball, 0.15))

# To run the code with a specific config
#pipenv run python main.py configs/caterpie.json



## 1A  ## 

# pokemons = ['snorlax']
#    pokeballs= ['pokeball']
#     for pokemon in pokemons:
#        for pokeball in pokeballs:
#            attempts = []
#            for _ in range(100):
#                attempts.append(
#                    attempt_catch(pokemon,pokeball)
#                )