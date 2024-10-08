import math
from abc import ABC, abstractmethod

class Class(ABC):
    @abstractmethod
    def get_performance(self, attack: float, defense: float) -> float:
        pass

class Warrior(Class):
    def get_performance(self, attack: float, defense: float) -> float:
        return 0.6 * attack + 0.4 * defense

class Archer(Class):
    def get_performance(self, attack: float, defense: float) -> float:
        return 0.9 * attack + 0.1 * defense

class Guardian(Class):
    def get_performance(self, attack: float, defense: float) -> float:
        return 0.1 * attack + 0.9 * defense

class Mage(Class):
    def get_performance(self, attack: float, defense: float) -> float:
        return 0.8 * attack + 0.3 * defense

class EVE:
    CLASSES = [Warrior, Archer, Guardian, Mage]

    @staticmethod
    def calculate_stats(items: dict) -> dict:
        return {
            "strength": 100 * math.tanh(0.01 * items["strength"]),
            "agility": math.tanh(0.01 * items["agility"]),
            "expertise": 0.6 * math.tanh(0.01 * items["expertise"]),
            "endurance": math.tanh(0.01 * items["endurance"]),
            "health": 100 * math.tanh(0.01 * items["health"])
        }

    @staticmethod
    def get_atm(height: float) -> float:
        return 0.5 - (3 * height - 5) ** 4 + (3 * height - 5) ** 2 + height / 2

    @staticmethod
    def get_dem(height: float) -> float:
        return 2 + (3 * height - 5) ** 4 - (3 * height - 5) ** 2 - height / 2

    @staticmethod
    def get_attack(stats: dict, height: float) -> float:
        return (stats["agility"] + stats["expertise"]) * stats["strength"] * EVE.get_atm(height)

    @staticmethod
    def get_defense(stats: dict, height: float) -> float:
        return (stats["endurance"] + stats["expertise"]) * stats["health"] * EVE.get_dem(height)

    @staticmethod
    def evaluate(items: dict, height: float, class_index: int) -> float:
        stats = EVE.calculate_stats(items)
        attack = EVE.get_attack(stats, height)
        defense = EVE.get_defense(stats, height)
        character_class = EVE.CLASSES[class_index]()
        return character_class.get_performance(attack, defense)