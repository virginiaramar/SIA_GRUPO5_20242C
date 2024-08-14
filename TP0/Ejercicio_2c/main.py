import json
import sys

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        ball = config["pokeball"]

        status_effects = [StatusEffect.NONE, StatusEffect.BURN, StatusEffect.FREEZE, StatusEffect.PARALYSIS, StatusEffect.POISON, StatusEffect.SLEEP]
        
        min_rate = float('inf')
        max_rate = float('-inf')
        
        for status_effect in status_effects:
            pokemon = factory.create(config["pokemon"], 100, status_effect, 1)
            capture_rate = attempt_catch(pokemon, ball)[1]
            if capture_rate < min_rate:
                min_rate = capture_rate
            if capture_rate > max_rate:
                max_rate = capture_rate
        status_impact = max_rate - min_rate

        hp_percentages = [1.0, 0.5, 0.1]
        
        min_rate = float('inf')
        max_rate = float('-inf')
        
        for hp in hp_percentages:
            pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, hp)
            capture_rate = attempt_catch(pokemon, ball)[1]
            if capture_rate < min_rate:
                min_rate = capture_rate
            if capture_rate > max_rate:
                max_rate = capture_rate
        hp_impact = max_rate - min_rate

        levels = [1, 50, 100]
        
        min_rate = float('inf')
        max_rate = float('-inf')
        
        for lvl in levels:
            pokemon = factory.create(config["pokemon"], lvl, StatusEffect.NONE, 1)
            capture_rate = attempt_catch(pokemon, ball)[1]
            if capture_rate < min_rate:
                min_rate = capture_rate
            if capture_rate > max_rate:
                max_rate = capture_rate
        lvl_impact = max_rate - min_rate

        pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
        
        min_rate = float('inf')
        max_rate = float('-inf')
        
        for pokeball in pokeballs:
            pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)
            capture_rate = attempt_catch(pokemon, pokeball)[1]
            if capture_rate < min_rate:
                min_rate = capture_rate
            if capture_rate > max_rate:
                max_rate = capture_rate
        pokeball_impact = max_rate - min_rate

        print(f"Status Effect Impact: {status_impact}")
        print(f"HP Percentage Impact: {hp_impact}")
        print(f"Level Impact: {lvl_impact}")
        print(f"Pokéball Type Impact: {pokeball_impact}")
