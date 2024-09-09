import math
import random

from roleplay.src.Eve import Eve


class Character:
    def __init__(self, character_class: str, total_points: int):

        self.character_class = character_class
        self.total_points = total_points
        self.height = random.uniform(1.3, 2.0)

        self.strength = 0
        self.dexterity = 0
        self.intelligence = 0
        self.vigor = 0
        self.constitution = 0

        self.distribute_points()

        self.attack = 0.0
        self.defense = 0.0

        self.calculate_attack()
        self.calculate_defense()

        self.performance_score = Eve().compute_performance_score(self)

    def distribute_points(self):
        attributes = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']
        num_attributes = len(attributes)

        # Generate random proportions using Dirichlet distribution
        proportions = [random.uniform(0, 1) for _ in range(num_attributes)]
        total = sum(proportions)

        # Normalize proportions to sum to 1
        proportions = [p / total for p in proportions]

        # Distribute points according to the generated proportions
        for i, attr in enumerate(attributes):
            points = int(self.total_points * proportions[i])
            setattr(self, attr, points)

        # Ensure the total points sum up to the desired amount
        current_total = sum(getattr(self, attr) for attr in attributes)
        while current_total < self.total_points:
            # Randomly add the remaining points to any attribute
            attr = random.choice(attributes)
            setattr(self, attr, getattr(self, attr) + 1)
            current_total += 1
        while current_total > self.total_points:
            # Randomly subtract the extra points from any attribute
            attr = random.choice(attributes)
            if getattr(self, attr) > 0:
                setattr(self, attr, getattr(self, attr) - 1)
                current_total -= 1

    def calculate_attack(self):
        strength_total = 100 * math.tanh(0.01 * self.strength)
        dexterity_total = math.tanh(0.01 * self.dexterity)
        intelligence_total = 0.6 * math.tanh(0.01 * self.intelligence)

        h = self.height
        ATM = 0.5 - (3 * h - 5) ** 4 + (3 * h - 5) ** 2 + h / 2

        self.attack = (dexterity_total + intelligence_total) * strength_total * ATM

    def calculate_defense(self):
        # Calculate attribute-modified values
        vigor_total = math.tanh(0.01 * self.vigor)
        constitution_total = 100 * math.tanh(0.01 * self.constitution)
        intelligence_total = 0.6 * math.tanh(0.01 * self.intelligence)

        # Compute the defense modifier (DEM) based on height
        h = self.height
        DEM = 2 + (3 * h - 5) ** 4 - (3 * h - 5) ** 2 - h / 2

        # Calculate Defense
        self.defense = (vigor_total + intelligence_total) * constitution_total * DEM

    def __str__(self):
        return (f"Class: {self.character_class}, Strength: {self.strength}, Dexterity: {self.dexterity}, "
                f"Intelligence: {self.intelligence}, Vigor: {self.vigor}, Constitution: {self.constitution}, "
                f"Height: {self.height:.2f}, Attack: {self.attack:.2f}, Defense: {self.defense:.2f}, "
                f"Performance Score: {self.performance_score:.2f}")
