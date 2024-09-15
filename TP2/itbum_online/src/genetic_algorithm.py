import random
import math
from typing import List, Tuple, Callable
from src.character import Character
from src.eve import EVE
import time
import numpy as np

class GeneticAlgorithm:
    def __init__(self, config: dict):
        self.config = config
        self.population_size = config['population_size']
        self.offspring_count = config['offspring_count']  # Nuevo parámetro añadido
        self.crossover_type = config['crossover']['type']
        self.crossover_rate = config['crossover']['rate']
        self.mutation_config = config['mutation']
        self.parent_selection = self.get_selection_method(config['selection']['parents'])
        self.replacement_selection = self.get_selection_method(config['selection']['replacement'])
        self.stop_criteria = config['stop_criteria']
        self.fixed_class = config.get('character_class') if config.get('character_class') is not None else random.randint(0, 3)
        self.total_points = config.get('total_points') if  config.get('total_points') is not None else random.randint(100, 200)
        self.generation = 0
        self.generation_history = []
        self.time_limit = config['time_limit']
        self.start_time = time.time()
        self.tournament_config = config['selection']['tournament']
        self.heuristic_enabled = config.get('heuristic_enabled', False)  



    def initialize_population(self) -> List[Character]:
        population = []
        for _ in range(self.population_size):
            items = {attr: random.uniform(0, 100) for attr in ["strength", "agility", "expertise", "endurance", "health"]}
            height = random.uniform(1.3, 2.0)
            class_index = self.fixed_class
            character = Character(items, height, class_index, self.total_points)

            if self.heuristic_enabled:
                character = self.apply_heuristic(character)


            population.append(character)
        return population
    
    def get_selection_method(self, config: dict) -> Callable:
        methods = {
            'elite': self.elite_selection,
            'roulette': self.roulette_selection,
            'universal': self.universal_selection,
            'ranking': self.ranking_selection,
            'boltzmann': self.boltzmann_selection,
            'tournament': self.tournament_selection,
        }
        method1 = methods[config['method1']]
        method2 = methods[config['method2']]
        proportion = config['method1_proportion']
        
        def combined_method(population: List[Character], k: int) -> List[Character]:
            k1 = int(k * proportion)
            k2 = k - k1
            
            # Método 1: Permitir repeticiones
            selected1 = method1(population, k1)
            
            # Método 2: Permitir repeticiones
            selected2 = method2(population, k2)
            
            print(f"Method 1 ({config['method1']}) selected: {len(selected1)}, Method 2 ({config['method2']}) selected: {len(selected2)}")
            
            return selected1 + selected2
        
        return combined_method
    

    def elite_selection(self, population: List[Character], k: int) -> List[Character]:
        n = len(population)
        sorted_population = sorted(population, key=lambda x: x.get_performance(), reverse=True)
        
        if k <= n:
            return sorted_population[:k]
        else:
            selected = []
            for i in range(n):
                repetitions = math.ceil((k - i) / n)
                for _ in range(repetitions):
                    if len(selected) == k:
                        return selected
                    selected.append(sorted_population[i])
            return selected

    def roulette_selection(self, population: List[Character], k: int) -> List[Character]:
        # Calculate relative fitness p_j
        total_fitness = sum(character.get_performance() for character in population)
        relative_fitness = [character.get_performance() / total_fitness for character in population]
        
        # Calculate cumulative relative fitness q_i
        cumulative_fitness = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
        
        selected = []
        for _ in range(k):
            r_j = random.uniform(0, 1)
            # Find the index i where q_i-1 < r_j <= q_i
            for i, q_i in enumerate(cumulative_fitness):
                if i == 0 and 0 <= r_j <= q_i:
                    selected.append(population[i])
                    break
                elif cumulative_fitness[i-1] < r_j <= q_i:
                    selected.append(population[i])
                    break
        return selected


    def universal_selection(self, population: List[Character], k: int) -> List[Character]:
        new_population = []
        
        # Calculate relative fitness p_j
        total_fitness = sum(character.get_performance() for character in population)
        relative_fitness = [character.get_performance() / total_fitness for character in population]
        
        # Calculate cumulative relative fitness q_i
        cumulative_fitness = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
        
        for j in range(k):
            r_j = (random.uniform(0, 1) + j) / k
            
            # Find the index i where q_i-1 < r_j <= q_i
            for i, q_i in enumerate(cumulative_fitness):
                if i == 0 and 0 <= r_j <= q_i:
                    new_population.append(population[i])
                    break
                elif cumulative_fitness[i-1] < r_j <= q_i:
                    new_population.append(population[i])
                    break
        
        return new_population
    
    def ranking_selection(self, population: List[Character], k: int) -> List[Character]:
        # Ordenar la población por fitness (de mayor a menor)
        sorted_population = sorted(population, key=lambda x: x.get_performance(), reverse=True)
        N = len(population)
        
        # Calcular el pseudo-fitness basado en el ranking
        pseudo_fitness = [(N - i) / N for i in range(N)]
        
        # Calcular las probabilidades relativas (p_j)
        total_pseudo_fitness = sum(pseudo_fitness)
        relative_probabilities = [fit / total_pseudo_fitness for fit in pseudo_fitness]
        
        # Calcular las probabilidades acumuladas (q_i)
        cumulative_probabilities = [sum(relative_probabilities[:i+1]) for i in range(N)]
        
        # Seleccionar k individuos usando el método de la ruleta
        selected = []
        for _ in range(k):
            r = random.random()
            for i, q_i in enumerate(cumulative_probabilities):
                if r <= q_i:
                    selected.append(sorted_population[i])
                    break
        
        return selected

    def boltzmann_selection(self, population: List[Character], k: int) -> List[Character]:
        T = self.boltzmann_temperature()
        
        # Calcular ExpVal para cada individuo
        fitnesses = [c.get_performance() for c in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        exp_values = [math.exp(f / T) for f in fitnesses]
        avg_exp = sum(exp_values) / len(exp_values)
        exp_vals = [ev / avg_exp for ev in exp_values]
        
        # Calcular probabilidades relativas
        total_exp_val = sum(exp_vals)
        probabilities = [ev / total_exp_val for ev in exp_vals]
        
        # Calcular probabilidades acumuladas
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        
        # Seleccionar k individuos usando el método de la ruleta
        selected = []
        for _ in range(k):
            r = random.random()
            for i, q_i in enumerate(cumulative_probabilities):
                if r <= q_i:
                    selected.append(population[i])
                    break
        
        return selected

    def boltzmann_temperature(self) -> float:
        Tmin = self.config['selection']['boltzmann']['Tmin']
        Tmax = self.config['selection']['boltzmann']['Tmax']
        k = self.config['selection']['boltzmann']['k']
        return Tmax - (Tmax - Tmin) * (1 - math.exp(-k * self.generation))

    def tournament_selection(self, population: List[Character], k: int) -> List[Character]:
        if self.tournament_config['type'] == 'deterministic':
            return self.deterministic_tournament(population, k, self.tournament_config['m'])
        elif self.tournament_config['type'] == 'probabilistic':
            return self.probabilistic_tournament(population, k, self.tournament_config['threshold'])
        else:
            raise ValueError("Invalid tournament type")

    def deterministic_tournament(self, population: List[Character], k: int, m: int) -> List[Character]:
        new_population = []
        for _ in range(k):
            selected_characters = random.sample(population, m)
            winner = max(selected_characters, key=lambda x: x.get_performance())
            new_population.append(winner)
        return new_population

    def probabilistic_tournament(self, population: List[Character], k: int, threshold: float) -> List[Character]:
        new_population = []
        for _ in range(k):
            selected_characters = random.sample(population, 2)
            selected_characters.sort(key=lambda x: x.get_performance(), reverse=True)
            if random.random() < threshold:
                new_population.append(selected_characters[0])  # Select the best
            else:
                new_population.append(selected_characters[1])  # Select the worst
        return new_population

        
    def crossover(self, parent1: Character, parent2: Character) -> Tuple[Character, Character]:
        if random.random() > self.crossover_rate:
            return parent1, parent2

        genotype1 = parent1.get_genotype()
        genotype2 = parent2.get_genotype()

        if self.crossover_type == 'one_point':
            child1, child2 = self.one_point_crossover(genotype1, genotype2)
        elif self.crossover_type == 'two_point':
            child1, child2 = self.two_point_crossover(genotype1, genotype2)
        elif self.crossover_type == 'uniform':
            child1, child2 = self.uniform_crossover(genotype1, genotype2)
        elif self.crossover_type == 'anular':
            child1, child2 = self.anular_crossover(genotype1, genotype2)
        else:
            raise ValueError(f"Invalid crossover type: {self.crossover_type}")

        return (Character.from_genotype(child1, parent1.class_index, self.total_points),
                Character.from_genotype(child2, parent2.class_index, self.total_points))

    def one_point_crossover(self, genes1: List[float], genes2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Cruce de un punto:
        Se elige un locus al azar y se intercambian los alelos a partir de ese locus.
        P = [0,S-1] ; S: Cantidad de genes
        """
        p = random.randint(0, len(genes1) - 1)
        child1 = genes1[:p] + genes2[p:]
        child2 = genes2[:p] + genes1[p:]
        return child1, child2

    def two_point_crossover(self, genes1: List[float], genes2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Cruce de dos puntos:
        Se eligen dos locus al azar y se intercambian los alelos entre ellos.
        P1 = [0,S-1] ; P2 = [0, S-1] ; P1 ≤ P2
        """
        p1, p2 = sorted(random.sample(range(len(genes1)), 2))
        child1 = genes1[:p1] + genes2[p1:p2] + genes1[p2:]
        child2 = genes2[:p1] + genes1[p1:p2] + genes2[p2:]
        return child1, child2

    def uniform_crossover(self, genes1: List[float], genes2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Cruce uniforme:
        Se produce un intercambio de alelos en cada gen con probabilidad P [0, 1].
        (Por lo general P = 0.5).
        Es el único tipo de cruce visto que no mantiene correlación posicional entre alelos.
        """
        child1, child2 = [], []
        for g1, g2 in zip(genes1, genes2):
            if random.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return child1, child2

    def anular_crossover(self, genes1: List[float], genes2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Cruce anular:
        Se elige un locus P al azar y una longitud L.
        Se intercambia el segmento de longitud L a partir de P.
        P = [0,S-1] ; L = [0, ⌈S/2⌉]
        """
        p = random.randint(0, len(genes1) - 1)
        length = random.randint(0, math.ceil(len(genes1)/2))
        length = min(length, len(genes1) - p)
        child1 = genes1[:p] + genes2[p:p+length] + genes1[p+length:]
        child2 = genes2[:p] + genes1[p:p+length] + genes2[p+length:]
        return child1, child2
    
    def mutate(self, population: List[Character]) -> List[Character]:
        mutation_type = self.mutation_config['type']
        mutation_rate = self.mutation_config['rate']
        is_uniform = self.mutation_config['uniform']

        if not is_uniform:
            mutation_rate = self.non_uniform_mutation_rate()

        if mutation_type == 'gen':
            return self.gene_mutation(population, mutation_rate)
        elif mutation_type == 'multigen':
            return self.multigen_mutation(population, mutation_rate)
        else:
            raise ValueError(f"Invalid mutation type: {mutation_type}")

    def gene_mutation(self, population: List[Character], mutation_rate: float) -> List[Character]:
        for character in population:
            if random.random() < mutation_rate:
                genotype = character.get_genotype()
                index = random.randint(0, len(genotype) - 1)
                genotype[index] = self.mutate_gene(genotype[index], index)
                character = Character.from_genotype(genotype, character.class_index, self.total_points)
        return population

    def multigen_mutation(self, population: List[Character], mutation_rate: float) -> List[Character]:
        for character in population:
            genotype = character.get_genotype()
            for i in range(len(genotype)):
                if random.random() < mutation_rate:
                    genotype[i] = self.mutate_gene(genotype[i], i)
            character = Character.from_genotype(genotype, character.class_index, self.total_points)
        return population

    def non_uniform_mutation_rate(self) -> float:
        return self.mutation_config['rate'] * (1 - self.generation / self.stop_criteria['max_generations'])

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
            parents = self.parent_selection(population, self.offspring_count)
            
            # Generación de hijos
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i+1])
                    offspring.extend([child1, child2])

            # Mutación
            offspring = self.mutate(offspring)
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
                    'best_character': current_best,
                    'variance_attrib': self.calculate_genetic_diversity(population)
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
    
    def calculate_genetic_diversity(self,population: List[Character]) -> dict:
        # Verificar que la población no esté vacía
        if not population:
            raise ValueError("La población está vacía.")
        
        # Inicializar listas para almacenar los atributos
        attributes = {
            "strength": [],
            "agility": [],
            "expertise": [],
            "endurance": [],
            "health": [],
            "height": []
        }
        
        # Extraer los atributos de cada Character en la población
        for character in population:
            items = character.items
            attributes["strength"].append(items.get("strength", 0))
            attributes["agility"].append(items.get("agility", 0))
            attributes["expertise"].append(items.get("expertise", 0))
            attributes["endurance"].append(items.get("endurance", 0))
            attributes["health"].append(items.get("health", 0))
            attributes["height"].append(character.height)
        
        # Calcular la varianza de cada atributo
        diversity = {}
        for key in attributes:
            # Convertir a array numpy para calcular la varianza
            attr_array = np.array(attributes[key])
            diversity[key] = np.var(attr_array, ddof=1)  # ddof=1 para varianza muestral

        return diversity