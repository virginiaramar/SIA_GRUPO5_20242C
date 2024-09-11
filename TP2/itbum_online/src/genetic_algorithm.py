import random
import math
from typing import List, Tuple, Callable
from src.character import Character
from src.eve import EVE

class GeneticAlgorithm:
    def __init__(self, config: dict):
        self.config = config
        self.population_size = config['population_size']
        self.crossover_type = config['crossover']['type']
        self.crossover_rate = config['crossover']['rate']
        self.mutation_type = config['mutation']['type']
        self.mutation_uniform = config['mutation']['uniform']
        self.mutation_rate = config['mutation']['rate']
        self.parent_selection = self.get_selection_method(config['selection']['parents'])
        self.replacement_selection = self.get_selection_method(config['selection']['replacement'])
        self.stop_criteria = config['stop_criteria']
        self.fixed_class = config.get('character_class')
        self.total_points = config.get('total_points')
        self.generation = 0

    def initialize_population(self) -> List[Character]:
        population = []
        for _ in range(self.population_size):
            items = {attr: random.uniform(0, 100) for attr in ["strength", "agility", "expertise", "endurance", "health"]}
            height = random.uniform(1.3, 2.0)
            class_index = self.fixed_class if self.fixed_class is not None else random.randint(0, 3)
            population.append(Character(items, height, class_index, self.total_points))
        return population

    def get_selection_method(self, config: dict) -> Callable:
        methods = {
            'tournament': self.tournament_selection,
            'roulette': self.roulette_selection,
            'universal': self.universal_selection,
            'boltzmann': self.boltzmann_selection,
            'ranking': self.ranking_selection,
            'elite': self.elite_selection
        }
        method1 = methods[config['method1']]
        method2 = methods[config['method2']]
        proportion = config['method1_proportion']
        
        def combined_method(population: List[Character], k: int) -> List[Character]:
            k1 = int(k * proportion)
            k2 = k - k1
            return method1(population, k1) + method2(population, k2)
        
        return combined_method

    def tournament_selection(self, population: List[Character], k: int) -> List[Character]:
        selected = []
        for _ in range(k):
            tournament = random.sample(population, 5)
            selected.append(max(tournament, key=lambda x: x.get_performance()))
        return selected

    def roulette_selection(self, population: List[Character], k: int) -> List[Character]:
        fitnesses = [c.get_performance() for c in population]
        total_fitness = sum(fitnesses)
        probabilities = [f/total_fitness for f in fitnesses]
        return random.choices(population, weights=probabilities, k=k)

    def universal_selection(self, population: List[Character], k: int) -> List[Character]:
        fitnesses = [c.get_performance() for c in population]
        total_fitness = sum(fitnesses)
        probabilities = [f/total_fitness for f in fitnesses]
        r = random.random() / k
        return [population[self.universal_selection_index(probabilities, r + i/k)] for i in range(k)]

    @staticmethod
    def universal_selection_index(probabilities: List[float], r: float) -> int:
        c = probabilities[0]
        i = 0
        while c < r:
            i += 1
            c += probabilities[i]
        return i

    def boltzmann_selection(self, population: List[Character], k: int) -> List[Character]:
        T = max(0.5, 1 - self.generation / self.stop_criteria['max_generations'])
        exp_values = [math.exp(c.get_performance() / T) for c in population]
        total = sum(exp_values)
        probabilities = [e/total for e in exp_values]
        return random.choices(population, weights=probabilities, k=k)

    def ranking_selection(self, population: List[Character], k: int) -> List[Character]:
        sorted_population = sorted(population, key=lambda x: x.get_performance(), reverse=True)
        ranks = list(range(1, len(population) + 1))
        return random.choices(sorted_population, weights=ranks, k=k)

    def elite_selection(self, population: List[Character], k: int) -> List[Character]:
        return sorted(population, key=lambda x: x.get_performance(), reverse=True)[:k]

    def crossover(self, parent1: Character, parent2: Character) -> Tuple[Character, Character]:
        if random.random() > self.crossover_rate:
            return parent1, parent2

        genotype1 = parent1.get_genotype()
        genotype2 = parent2.get_genotype()

        if self.crossover_type == 'one_point':
            point = random.randint(1, len(genotype1) - 1)
            child1 = genotype1[:point] + genotype2[point:]
            child2 = genotype2[:point] + genotype1[point:]
        elif self.crossover_type == 'two_point':
            points = sorted(random.sample(range(1, len(genotype1)), 2))
            child1 = genotype1[:points[0]] + genotype2[points[0]:points[1]] + genotype1[points[1]:]
            child2 = genotype2[:points[0]] + genotype1[points[0]:points[1]] + genotype2[points[1]:]
        elif self.crossover_type == 'uniform':
            child1 = [g1 if random.random() < 0.5 else g2 for g1, g2 in zip(genotype1, genotype2)]
            child2 = [g2 if random.random() < 0.5 else g1 for g1, g2 in zip(genotype1, genotype2)]
        elif self.crossover_type == 'arithmetic':
            alpha = random.random()
            child1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(genotype1, genotype2)]
            child2 = [(1 - alpha) * g1 + alpha * g2 for g1, g2 in zip(genotype1, genotype2)]

        return Character.from_genotype(child1, self.total_points), Character.from_genotype(child2, self.total_points)

    def mutate(self, character: Character) -> Character:
        genotype = character.get_genotype()
        
        if self.mutation_type == 'gen':
            if random.random() < self.mutation_rate:
                index = random.randint(0, len(genotype) - 1)
                genotype[index] = self.mutate_gene(genotype[index], index)
        elif self.mutation_type == 'multigen':
            for i in range(len(genotype)):
                if random.random() < self.mutation_rate:
                    genotype[i] = self.mutate_gene(genotype[i], i)

        if not self.mutation_uniform:
            self.mutation_rate *= 0.99  # Decrease mutation rate over time

        return Character.from_genotype(genotype, self.total_points)


    def mutate_gene(self, gene: float, index: int) -> float:
        if index < 5:  # Items
            return max(0, gene + random.uniform(-10, 10))  
        elif index == 5:  # Height
            return max(1.3, min(2.0, gene + random.uniform(-0.1, 0.1)))
        else:  # Class
            return float(random.randint(0, 3))


    def evolve(self) -> Character:
        population = self.initialize_population()
        best_fitness = 0
        generations_no_improve = 0

        while not self.should_stop(population, best_fitness):
            # Selección de padres
            parents = self.parent_selection(population, self.population_size)
            
            # Generación de hijos
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    offspring.extend([self.mutate(child1), self.mutate(child2)])

            # Aplicar método de reemplazo
            if self.config['replacement_method'] == 'traditional':
                # Método Tradicional (Fill-All)
                combined = population + offspring
                population = self.replacement_selection(combined, self.population_size)
            elif self.config['replacement_method'] == 'young_bias':
                # Método de Sesgo Joven (Fill-Parent)
                if len(offspring) > self.population_size:
                    population = self.replacement_selection(offspring, self.population_size)
                else:
                    remaining = self.population_size - len(offspring)
                    population = offspring + self.replacement_selection(population, remaining)
            
            current_best = max(population, key=lambda x: x.get_performance())
            if current_best.get_performance() > best_fitness:
                best_fitness = current_best.get_performance()
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            self.generation += 1

        return max(population, key=lambda x: x.get_performance())

    def should_stop(self, population: List[Character], best_fitness: float) -> bool:
        # Criterio 1: Máxima cantidad de generaciones
        if self.generation >= self.stop_criteria['max_generations']:
            print("Stopping: Maximum number of generations reached.")
            return True

        # Criterio 2: Estructura (convergencia de la población)
        fitnesses = [c.get_performance() for c in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        if (max_fitness - avg_fitness) / avg_fitness < self.stop_criteria['structure']:
            print("Stopping: Population structure converged.")
            return True

        # Criterio 3: Contenido (estancamiento del mejor fitness)
        if (best_fitness - avg_fitness) / avg_fitness < self.stop_criteria['content']:
            print("Stopping: Best fitness stagnated.")
            return True

        # Criterio 4: Entorno a un óptimo
        if best_fitness >= self.stop_criteria['optimal_fitness']:
            print(f"Stopping: Optimal fitness reached. Best fitness: {best_fitness}")
            return True

        return False