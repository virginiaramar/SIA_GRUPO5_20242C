import json

from roleplay.src.CrossoverStrategy import *
from roleplay.src.Eve import Eve
from roleplay.src.SelectionStrategy import *


class GeneticAlgorithm:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        self.population_size = self.config.get('population_size')
        self.mutation_rate = self.config.get('mutation_rate')
        self.crossover_method = self.config.get('crossover_method')
        self.selection_method = self.config.get('selection_method')
        self.generational_gap = self.config.get('generational_gap')
        self.replacement_method = self.config.get('replacement_method')
        self.max_generations = self.config.get('max_generations')
        self.stopping_criteria = self.config.get('stopping_criteria')
        self.optimal_fitness = self.config.get('optimal_fitness')
        self.stagnant_population_fraction_limit = self.config.get('stagnant_population_fraction_limit')
        self.stagnant_generations_limit = self.config.get('stagnant_generations_limit')
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

    def generate_offspring(self, generation_number):
        new_population = self.selection(generation_number)
        old_population = self.population
        while len(new_population) < self.population_size:
            parent1 = self.replacement(old_population, generation_number)
            parent2 = self.replacement(old_population, generation_number)

            # in case same parents were selected
            while parent2 == parent1:
                parent2 = self.replacement(old_population, generation_number)

            old_population.remove(parent1)
            old_population.remove(parent2)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            offspring1 = self.gen_mutation(offspring1)
            offspring2 = self.gen_mutation(offspring2)

            new_population.append(offspring1)
            new_population.append(offspring2)

        self.population = new_population[:self.population_size]

    def selection(self, generation_number):
        new_population = []
        if self.selection_method == "elite":
            new_population = self.elite_selection()
        if self.selection_method == "ranking":
            while len(new_population) < self.generational_gap * self.population_size:
                new_population.append(ranking_selection(self.population))
        if self.selection_method == "roulette":
            while len(new_population) < self.generational_gap * self.population_size:
                new_population.append(roulette_selection(self.population))
        if self.selection_method == "boltzmann":
            while len(new_population) < self.generational_gap * self.population_size:
                new_population.append(boltzmann_selection(self.population, 100 / (1 + 0.1 * generation_number)))
        return new_population

    def replacement(self, population, generation_number):
        if self.selection_method == "ranking":
            return ranking_selection(population)
        if self.replacement_method == "roulette":
            return roulette_selection(population)
        elif self.replacement_method == "boltzmann":
            initial_temperature = 100
            cooling_rate = 0.1
            temperature = initial_temperature / (1 + cooling_rate * generation_number)
            return boltzmann_selection(population, temperature)
        else:
            return None

    def crossover(self, parent1, parent2):
        if self.crossover_method == "one-point":
            return one_point_crossover(parent1, parent2)
        elif self.crossover_method == "two-point":
            return two_point_crossover(parent1, parent2)
        elif self.crossover_method == "uniform":
            return uniform_crossover(parent1, parent2)
    def elite_selection(self):
        self.population.sort(key=lambda individual: individual.performance_score, reverse=True)

        num_elites = int(self.generational_gap * self.population_size)

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

    def run_algorithm(self):
        if self.stopping_criteria == "max_generations":
            max_fitness = 0
            best_character = None
            generation_number = 0
            for _ in range(self.max_generations):
                fitness_values = []
                self.generate_offspring(generation_number)
                generation_number += 1
                for character in self.population:
                    fitness_values.append(character.performance_score)

                current_max = max(fitness_values)
                if current_max > max_fitness:
                    max_fitness = current_max
                    best_character = next(c for c in self.population if c.performance_score == current_max)
            print(f"Max Fitness: {max_fitness}")
            print(f"Best Character: {best_character}")
        elif self.stopping_criteria == "acceptable_solution":
            max_fitness = 0
            best_character = None
            generation_number = 0

            while (max_fitness < self.optimal_fitness):
                fitness_values = []
                self.generate_offspring(generation_number)
                generation_number += 1

                for character in self.population:
                    fitness_values.append(character.performance_score)

                current_max = max(fitness_values)
                if current_max > max_fitness:
                    max_fitness = current_max
                    best_character = next(c for c in self.population if c.performance_score == current_max)
            print(f"Max Fitness: {max_fitness}")
            print(f"Best Character: {best_character}")
        elif self.stopping_criteria == "stagnant_content":
            max_fitness = 0
            best_character = None
            stagnant_generations = 0
            generation_number = 0

            while 1:
                fitness_values = []
                self.generate_offspring(generation_number)
                generation_number += 1

                for character in self.population:
                    fitness_values.append(character.performance_score)

                current_max = max(fitness_values)
                if current_max > max_fitness:
                    max_fitness = current_max
                    best_character = next(c for c in self.population if c.performance_score == current_max)
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

                if stagnant_generations >= self.stagnant_generations_limit:
                    print("Stopping due to stagnant best fitness.")
                    break

            print(f"Max Fitness: {max_fitness}")
            print(f"Best Character: {best_character}")

        elif self.stopping_criteria == "stagnant_structure":
            max_fitness = 0
            best_character = None
            unchanged_generations = 0
            generation_number = 0
            previous_population_state = [char.performance_score for char in self.population]

            while 1:
                self.generate_offspring(generation_number)
                generation_number += 1
                current_population_state = [char.performance_score for char in self.population]

                unchanged_individuals = sum(1 for i in range(len(current_population_state))
                                            if current_population_state[i] == previous_population_state[i])

                unchanged_fraction = unchanged_individuals / len(self.population)

                if unchanged_fraction >= self.stagnant_population_fraction_limit:
                    unchanged_generations += 1
                else:
                    unchanged_generations = 0

                previous_population_state = current_population_state

                if unchanged_generations >= self.stagnant_generations_limit:
                    print(f"Stopping after {generation_number} generations due to stagnant population.")
                    break

                fitness_values = [character.performance_score for character in self.population]
                current_max = max(fitness_values)

                if current_max > max_fitness:
                    max_fitness = current_max
                    best_character = next(c for c in self.population if c.performance_score == current_max)

            print(f"Max Fitness: {max_fitness}")
            print(f"Best Character: {best_character}")


def main():
    ga = GeneticAlgorithm()
    ga.run_algorithm()


if __name__ == "__main__":
    main()
