import random
import math
import numpy.random as npr



def boltzmann_selection(population, temperature):
    fitness_scores = [character.performance_score for character in population]
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    boltzmann_probabilities = [math.exp(fitness / temperature) for fitness in fitness_scores]
    total_boltzmann_prob = sum(boltzmann_probabilities)
    selection_probs = [prob / total_boltzmann_prob for prob in boltzmann_probabilities]

    return population[npr.choice(len(population), p=selection_probs)]


def roulette_selection(population):
    max = sum([c.performance_score for c in population])
    selection_probs = [c.performance_score / max for c in population]
    return population[npr.choice(len(population), p=selection_probs)]


def ranking_selection(population):
    sorted_population = sorted(population, key=lambda character: character.performance_score)
    ranks = range(1, len(sorted_population) + 1)
    total_rank = sum(ranks)
    selection_probs = [rank / total_rank for rank in ranks]

    return population[npr.choice(len(population), p=selection_probs)]
