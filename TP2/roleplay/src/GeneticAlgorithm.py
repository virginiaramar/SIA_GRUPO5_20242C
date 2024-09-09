import json
import random
import numpy.random as npr

from roleplay.src.Character import Character
from roleplay.src.CrossoverStrategy import *
from roleplay.src.Eve import Eve


class GeneticAlgorithm:
    def __init__(self, config_file: str):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        self.population_size = self.config.get('population_size')
        self.mutation_rate = self.config.get('mutation_rate')
        self.crossover_method = self.config.get('crossover_method')
        self.selection_method = self.config.get('selection_method')
        self.replacement_method = self.config.get('replacement_method')
        self.max_generations = self.config.get('max_generations')
        self.stopping_criteria = self.config.get('stopping_criteria')
        self.char_class = self.config.get('char_class')

        self.population = self.initialize_population(self.char_class)

        self.current_generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []

    def initialize_population(self, population_class):
        population = []
        for _ in range(self.population_size):
            character_class = population_class
            total_points = random.randint(100, 200)

            character = Character(character_class, total_points)
            population.append(character)

        return population

    def generate_offspring(self):
        new_population = self.selection()
        old_population = self.population
        while len(new_population) < self.population_size:
            parent1 = self.roulette_selection(old_population)
            parent2 = self.roulette_selection(old_population)
            while parent2 == parent1:
                parent2 = self.roulette_selection(old_population)

            old_population.remove(parent1)
            old_population.remove(parent2)

            offspring1, offspring2 = uniform_crossover(parent1, parent2)

            offspring1 = self.gen_mutation(offspring1)
            offspring2 = self.gen_mutation(offspring2)

            new_population.append(offspring1)
            new_population.append(offspring2)

        self.population = new_population[:self.population_size]

    def selection(self):
        new_population = []
        if self.selection_method == "elite":
            new_population = self.elite_selection(0.8)
        return new_population

    def select_parents(self, population):
        parent1, parent2 = None, None
        if self.replacement_method == "roulette":
            parent1, parent2 = self.roulette(population), self.roulette(population)
        return parent1, parent2

    def roulette_selection(self, population):
        max = sum([c.performance_score for c in population])
        selection_probs = [c.performance_score / max for c in population]
        return population[npr.choice(len(population), p=selection_probs)]

    def roulette(self, population):
        total_fitness = sum(individual.performance_score for individual in population)
        pick = random.uniform(0, total_fitness)

        current = 0
        for individual in population:
            current += individual.performance_score
            if current > pick:
                return individual

    def elite_selection(self, selection_rate):
        self.population.sort(key=lambda individual: individual.performance_score, reverse=True)

        num_elites = int(selection_rate * self.population_size)

        return self.population[:num_elites]

    def gen_mutation(self, character):
        attributes = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']
        new_character = Character(character.character_class, character.total_points)

        for attr in attributes:
            setattr(new_character, attr, getattr(character, attr))
        new_character.height = character.height

        if random.random() < self.mutation_rate:
            gene_to_mutate = random.choice(attributes + ['height'])

            if gene_to_mutate == 'height':
                mutation_step = random.uniform(-0.1, 0.1)
                new_character.height = max(1.3, min(2.0, character.height + mutation_step))
            else:
                value = getattr(character, gene_to_mutate)
                mutation_step = random.randint(-5, 5)
                mutated_value = max(0, min(100,
                                           value + mutation_step))
                setattr(new_character, gene_to_mutate, mutated_value)

        new_character_total = sum(
            [new_character.strength, new_character.dexterity, new_character.intelligence, new_character.vigor,
             new_character.constitution])
        adjust_attributes(new_character, new_character_total)

        new_character.calculate_attack()
        new_character.calculate_defense()
        new_character.performance_score = Eve().compute_performance_score(new_character)

        return new_character

    def multigen_mutation(self, character):
        attributes = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']
        new_character = Character(character.character_class, character.total_points)

        # Mutate integer attributes directly
        for attr in attributes:
            value = getattr(character, attr)
            if random.random() < self.mutation_rate:
                mutation_step = random.randint(-5, 5)
                mutated_value = max(0, min(100,
                                           value + mutation_step))
                setattr(new_character, attr, mutated_value)
            else:
                setattr(new_character, attr, value)

        if random.random() < self.mutation_rate:
            mutation_step = random.uniform(-0.1, 0.1)
            new_character.height = max(1.3, min(2.0, character.height + mutation_step))
        else:
            new_character.height = character.height

        new_character_total = sum(
            [new_character.strength, new_character.dexterity, new_character.intelligence, new_character.vigor,
             new_character.constitution])
        adjust_attributes(new_character, new_character_total)

        new_character.calculate_attack()
        new_character.calculate_defense()
        new_character.performance_score = Eve().compute_performance_score(new_character)

        return new_character


def main():
    ga = GeneticAlgorithm(config_file='config.json')
    max_fitness = 0
    best_character = None  # Variable to keep track of the character with the maximum fitness

    for _ in range(ga.max_generations):
        fitness_values = []
        ga.generate_offspring()

        for character in ga.population:
            fitness_values.append(character.performance_score)

        current_max = max(fitness_values)
        if current_max > max_fitness:
            max_fitness = current_max
            # Find the character with the maximum fitness
            best_character = next(c for c in ga.population if c.performance_score == current_max)
    print(f"Max Fitness: {max_fitness}")
    print(f"Best Character: {best_character}")


if __name__ == "__main__":
    main()
