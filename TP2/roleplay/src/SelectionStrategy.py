import random


def elite_selection(self):
    self.population.sort(key=lambda individual: individual.fitness, reverse=True)

    num_elites = int(self.elitism_rate * self.population_size)

    return self.population[:num_elites]

def roulette_wheel_selection(self):
    total_fitness = self.calculate_total_fitness()

    pick = random.uniform(0, total_fitness)

    current = 0
    for individual in self.population:
        current += individual.fitness
        if current > pick:
            return individual