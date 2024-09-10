import random

from roleplay.src.Character import Character
from roleplay.src.Eve import Eve

def uniform_crossover(parent1, parent2):
    parent1_chromosome = [
        parent1.strength, parent1.dexterity, parent1.intelligence,
        parent1.vigor, parent1.constitution, parent1.height
    ]
    parent2_chromosome = [
        parent2.strength, parent2.dexterity, parent2.intelligence,
        parent2.vigor, parent2.constitution, parent2.height
    ]

    offspring1_chromosome = []
    offspring2_chromosome = []

    for gene1, gene2 in zip(parent1_chromosome, parent2_chromosome):
        if random.random() < 0.5:
            offspring1_chromosome.append(gene1)
            offspring2_chromosome.append(gene2)
        else:
            offspring1_chromosome.append(gene2)
            offspring2_chromosome.append(gene1)

    offspring1 = Character(parent1.character_class, parent1.total_points)
    offspring2 = Character(parent2.character_class, parent2.total_points)

    assign_attributes(offspring1, offspring1_chromosome)
    assign_attributes(offspring2, offspring2_chromosome)

    offspring1_total = sum([offspring1.strength, offspring1.dexterity, offspring1.intelligence, offspring1.vigor, offspring1.constitution])
    offspring2_total = sum([offspring2.strength, offspring2.dexterity, offspring2.intelligence, offspring2.vigor, offspring2.constitution])

    adjust_attributes(offspring1, offspring1_total)
    adjust_attributes(offspring2, offspring2_total)

    eve = Eve()
    offspring1.performance_score = eve.compute_performance_score(offspring1)
    offspring2.performance_score = eve.compute_performance_score(offspring2)

    return offspring1, offspring2

def two_point_crossover(parent1, parent2):
    parent1_chromosome = [
        parent1.strength, parent1.dexterity, parent1.intelligence,
        parent1.vigor, parent1.constitution, parent1.height
    ]
    parent2_chromosome = [
        parent2.strength, parent2.dexterity, parent2.intelligence,
        parent2.vigor, parent2.constitution, parent2.height
    ]

    chromosome_length = len(parent1_chromosome)

    point1 = random.randint(1, chromosome_length - 2)
    point2 = random.randint(point1 + 1, chromosome_length - 1)

    offspring1_chromosome = parent1_chromosome[:point1] + parent2_chromosome[point1:point2] + parent1_chromosome[point2:]
    offspring2_chromosome = parent2_chromosome[:point1] + parent1_chromosome[point1:point2] + parent2_chromosome[point2:]

    offspring1 = Character(parent1.character_class, parent1.total_points)
    offspring2 = Character(parent2.character_class, parent2.total_points)

    assign_attributes(offspring1, offspring1_chromosome)
    assign_attributes(offspring2, offspring2_chromosome)

    offspring1_total = sum([offspring1.strength, offspring1.dexterity, offspring1.intelligence, offspring1.vigor, offspring1.constitution])
    offspring2_total = sum([offspring2.strength, offspring2.dexterity, offspring2.intelligence, offspring2.vigor, offspring2.constitution])

    adjust_attributes(offspring1, offspring1_total)
    adjust_attributes(offspring2, offspring2_total)

    eve = Eve()
    offspring1.performance_score = eve.compute_performance_score(offspring1)
    offspring2.performance_score = eve.compute_performance_score(offspring2)

    return offspring1, offspring2


def one_point_crossover(parent1, parent2):
    parent1_chromosome = [
        parent1.strength, parent1.dexterity, parent1.intelligence,
        parent1.vigor, parent1.constitution, parent1.height
    ]
    parent2_chromosome = [
        parent2.strength, parent2.dexterity, parent2.intelligence,
        parent2.vigor, parent2.constitution, parent2.height
    ]

    chromosome_length = len(parent1_chromosome)

    point = random.randint(1, chromosome_length - 1)

    offspring1_chromosome = parent1_chromosome[:point] + parent2_chromosome[point:]
    offspring2_chromosome = parent2_chromosome[:point] + parent1_chromosome[point:]

    offspring1 = Character(parent1.character_class, parent1.total_points)
    offspring2 = Character(parent2.character_class, parent2.total_points)

    assign_attributes(offspring1, offspring1_chromosome)
    assign_attributes(offspring2, offspring2_chromosome)

    offspring1_total = sum([offspring1.strength, offspring1.dexterity, offspring1.intelligence, offspring1.vigor, offspring1.constitution])
    offspring2_total = sum([offspring2.strength, offspring2.dexterity, offspring2.intelligence, offspring2.vigor, offspring2.constitution])

    adjust_attributes(offspring1, offspring1_total)
    adjust_attributes(offspring2, offspring2_total)

    eve = Eve()
    offspring1.performance_score = eve.compute_performance_score(offspring1)
    offspring2.performance_score = eve.compute_performance_score(offspring2)

    return offspring1, offspring2


def assign_attributes(offspring, chromosome):
    offspring.strength = chromosome[0]
    offspring.dexterity = chromosome[1]
    offspring.intelligence = chromosome[2]
    offspring.vigor = chromosome[3]
    offspring.constitution = chromosome[4]
    offspring.height = chromosome[5]


def adjust_attributes(offspring, current_total):
    attributes = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']
    target_total = offspring.total_points
    diff = target_total - current_total
    while diff != 0:
        attr = random.choice(attributes)
        current_value = getattr(offspring, attr)

        if diff > 0:
            if current_value < 100:
                increment = min(diff, 100 - current_value)
                setattr(offspring, attr, current_value + increment)
                diff -= increment
        else:
            if current_value > 0:
                decrement = min(-diff, current_value)
                setattr(offspring, attr, current_value - decrement)
                diff += decrement

# Even distribution that doesn't work with the structur stopping criteria because the population doesn't converge
# def adjust_attributes(offspring, current_total):
#     attributes = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']
#     target_total = offspring.total_points
#     diff = target_total - current_total
#     num_attributes = len(attributes)
#
#     while diff != 0:
#         adjustment_per_attr = diff // num_attributes
#
#         for attr in attributes:
#             current_value = getattr(offspring, attr)
#
#             if diff > 0:
#                 if current_value < 100:
#                     increment = min(adjustment_per_attr, 100 - current_value)
#                     setattr(offspring, attr, current_value + increment)
#                     diff -= increment
#             else:
#                 if current_value > 0:
#                     decrement = min(-adjustment_per_attr, current_value)
#                     setattr(offspring, attr, current_value - decrement)
#                     diff += decrement
#
#             if diff == 0:
#                 break