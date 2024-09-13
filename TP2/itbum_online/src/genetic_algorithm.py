import random
import math
from typing import List, Tuple, Callable
from src.character import Character
from src.eve import EVE
import time

class GeneticAlgorithm:
    def __init__(self, config: dict):
        self.config = config
        self.population_size = config['population_size']
        self.offspring_count = config['offspring_count']  # Nuevo parámetro añadido
        self.crossover_type = config['crossover']['type']
        self.crossover_rate = config['crossover']['rate']
        self.mutation_type = config['mutation']['type']
        self.mutation_uniform = config['mutation']['uniform']
        self.mutation_rate = config['mutation']['rate']
        self.parent_selection = self.get_selection_method(config['selection']['parents'])
        self.replacement_selection = self.get_selection_method(config['selection']['replacement'])
        self.stop_criteria = config['stop_criteria']
        self.fixed_class = config.get('character_class') if config.get('character_class') is not None else random.randint(0, 3)
        self.total_points = config.get('total_points') if  config.get('total_points') is not None else random.randint(100, 200)
        self.generation = 0
        self.generation_history = []
        self.time_limit = config['time_limit']
        self.start_time = time.time()

    def initialize_population(self) -> List[Character]:
        population = []
        for _ in range(self.population_size):
            items = {attr: random.uniform(0, 100) for attr in ["strength", "agility", "expertise", "endurance", "health"]}
            height = random.uniform(1.3, 2.0)
            class_index = self.fixed_class
            character = Character(items, height, class_index, self.total_points)
            population.append(character)
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
            
            selected1 = method1(population, k1)
            remaining_population = [ind for ind in population if ind not in selected1]
            selected2 = method2(remaining_population, k2)
            
            return selected1 + selected2
        
        return combined_method

    def tournament_selection(self, population: List[Character], k: int, tournament_size: int = 5, probabilistic: bool = False) -> List[Character]:
        selected = []
        for _ in range(k):
            tournament = random.sample(population, tournament_size)
            if probabilistic:
                threshold = 0.75
                if random.random() < threshold:
                    selected.append(max(tournament, key=lambda x: x.get_performance()))
                else:
                    selected.append(min(tournament, key=lambda x: x.get_performance()))
            else:
                selected.append(max(tournament, key=lambda x: x.get_performance()))
        return selected

    def roulette_selection(self, population: List[Character], k: int) -> List[Character]:
        fitnesses = [c.get_performance() for c in population]
        total_fitness = sum(fitnesses)
        probabilities = [f/total_fitness for f in fitnesses]
        selected = []
        for _ in range(k):
            r = random.random()
            for i, prob in enumerate(probabilities):
                if r <= prob:
                    selected.append(population[i])
                    break
                r -= prob
        return selected

    def universal_selection(self, population: List[Character], k: int) -> List[Character]:
        fitnesses = [c.get_performance() for c in population]
        total_fitness = sum(fitnesses)
        probabilities = [f/total_fitness for f in fitnesses]
        r = random.random() / k
        return [population[self.universal_selection_index(probabilities, r + i/k)] for i in range(k)]

    @staticmethod
    def universal_selection_index(probabilities: List[float], r: float) -> int:
        c = 0
        for i, prob in enumerate(probabilities):
            c += prob
            if c > r:
                return i
        return len(probabilities) - 1

    def boltzmann_selection(self, population: List[Character], k: int) -> List[Character]:
        T = self.boltzmann_temperature()
        exp_values = [math.exp(c.get_performance() / T) for c in population]
        total = sum(exp_values)
        probabilities = [e/total for e in exp_values]
        return random.choices(population, weights=probabilities, k=k)

    def boltzmann_temperature(self) -> float:
        Tmin = 0.5
        Tmax = 2.0
        k = 0.1    
        return Tmax - (Tmax - Tmin) * (1 - math.exp(-k * self.generation))

    def ranking_selection(self, population: List[Character], k: int) -> List[Character]:
        sorted_population = sorted(population, key=lambda x: x.get_performance(), reverse=True)
        N = len(population)
        ranks = [self.ranking_function(i, N) for i in range(1, N+1)]
        total = sum(ranks)
        probabilities = [r/total for r in ranks]
        return random.choices(sorted_population, weights=probabilities, k=k)

    @staticmethod
    def ranking_function(i: int, N: int) -> float:
        return 2 - 2 * (i - 1) / (N - 1)

    def elite_selection(self, population: List[Character], k: int) -> List[Character]:
        N = len(population)
        sorted_population = sorted(population, key=lambda x: x.get_performance(), reverse=True)
        selected = []
        
        for i in range(N):  # Iteramos solo sobre N
            n_i = math.ceil((k - i) / N)
            if n_i > 0:
                selected.extend([sorted_population[i]] * n_i)
            else:
                break  # Si n_i llega a 0 o menos, terminamos la selección
        
        # Si aún no hemos seleccionado k individuos, completamos con los mejores
        while len(selected) < k:
            selected.append(sorted_population[0])
        
        return selected[:k]  # Aseguramos que devolvemos exactamente k individuos

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

        return (Character.from_genotype(child1, parent1.class_index, self.total_points),
                Character.from_genotype(child2, parent2.class_index, self.total_points))

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

        return Character.from_genotype(genotype, character.class_index, self.total_points)

    def mutate_gene(self, gene: float, index: int) -> float:
        if index < 5:  # Items
            return max(0, gene + random.uniform(-10, 10))  
        elif index == 5:  # Height
            return max(1.3, min(2.0, gene + random.uniform(-0.1, 0.1)))
        else:  # No mutation for class
            return gene

    def evolve(self) -> Character:
        population = self.initialize_population()
        best_fitness = float('-inf')
        self.generation_history = []
        self.generation = 0
        generations_no_improve = 0

        while self.generation < self.stop_criteria['max_generations']:
            # Selección de padres
            parents = self.parent_selection(population, self.offspring_count)  # Cambiado a offspring_count para q sea igual
            
            # Generación de hijos
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    offspring.extend([self.mutate(child1), self.mutate(child2)])

            # Aplicar método de reemplazo
            if self.config['replacement_method'] == 'traditional':
                # Método Tradicional (Fill-All)
                print("Método tradicional. La nueva población está conformada por "+ (str(len(population))) +" individuos SELECCIONADOS del conjunto formado por "+ str(len(population) + len(offspring)) + " individuos que corresponde a la suma de la generación actual y los hijos generados.")
                combined = population + offspring
                population = self.replacement_selection(combined, self.population_size)
            elif self.config['replacement_method'] == 'young_bias':
                # Método de Sesgo Joven (Fill-Parent)
                if len(offspring) > self.population_size:
                    print("Hay sesgo Joven. La nueva población está conformada por "+ str(self.population_size) +" individuos SELECCIONADOS únicamente del conjunto de " + str(len(offspring)) + " HIJOS generados.")
                    population = self.replacement_selection(offspring, self.population_size)
                else:
                    remaining = self.population_size - len(offspring)
                    print("Hay sesgo Joven. La nueva población está conformada por " + str(len(offspring)) +" hijos generados (conjunto de todos los hijos) y  " + str(remaining) + " individuos SELECCIONADOS del conjunto formado por "+ str(len(population)) + " individuos de la generación actual")
                    population = offspring + self.replacement_selection(population, remaining)
            
            current_best = max(population, key=lambda x: x.get_performance())
            avg_fitness = sum(c.get_performance() for c in population) / len(population)
            
            self.generation_history.append({
                    'generation': self.generation,
                    'best_fitness': current_best.get_performance(),
                    'average_fitness': avg_fitness,
                })
            
            print(f"Generation {self.generation}: Best Fitness = {current_best.get_performance():.4f}, Avg Fitness = {avg_fitness:.4f}")
            
            if current_best.get_performance() > best_fitness:
                best_fitness = current_best.get_performance()
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            self.generation += 1

            # Verificar otros criterios de parada
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.time_limit:
                print(f"Tiempo límite alcanzado después de {self.generation} generaciones.")
                break
            
            if self.should_stop(population, best_fitness):
                break

        print(f"Evolution completed after {self.generation} generations.")
        print(f"Total generations recorded: {len(self.generation_history)}")
        return max(population, key=lambda x: x.get_performance())

    def should_stop(self, population: List[Character], best_fitness: float) -> bool:
        # Criterio 1: Máxima cantidad de generaciones
        if self.generation >= self.stop_criteria['max_generations']:
            print(f"Stopping: Maximum number of generations ({self.stop_criteria['max_generations']}) reached.")
            return True

        # Criterio 2: Estructura (convergencia de la población)
        fitnesses = [c.get_performance() for c in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        if (max_fitness - avg_fitness) / avg_fitness < self.stop_criteria['structure']:
            print(f"Stopping: Population structure converged. Difference: {(max_fitness - avg_fitness) / avg_fitness:.4f}")
            return True

        # Criterio 3: Contenido (estancamiento del mejor fitness)
        if (best_fitness - avg_fitness) / avg_fitness < self.stop_criteria['content']:
            print(f"Stopping: Best fitness stagnated. Difference: {(best_fitness - avg_fitness) / avg_fitness:.4f}")
            return True

        # Criterio 4: Entorno a un óptimo
        if best_fitness >= self.stop_criteria['optimal_fitness']:
            print(f"Stopping: Optimal fitness reached. Best fitness: {best_fitness:.4f}")
            return True

        return False
        
    def get_generation_history(self):
        return self.generation_history