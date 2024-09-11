import random
from src.eve import EVE

class Character:
    def __init__(self, items: dict[str, float], height: float, class_index: int = None, total_points: int = None):
        self.total_points = total_points 
        self.items = self._normalize_items(items, self.total_points)
        self.height = height
        self.class_index = class_index 
        
    @staticmethod
    def _normalize_items(items: dict[str, float], total_points: int) -> dict[str, float]:
        total = sum(items.values())
        return {k: v * total_points / total for k, v in items.items()}

    @staticmethod
    def from_genotype(genotype: list[float], class_index: int, total_points: int = None):
        items = {
            "strength": genotype[0],
            "agility": genotype[1],
            "expertise": genotype[2],
            "endurance": genotype[3],
            "health": genotype[4]
        }
        height = genotype[5]
        return Character(items, height, class_index, total_points)

    def get_genotype(self) -> list[float]:
        genes = list(self.items.values())
        genes.append(self.height)
        return genes  # No incluimos class_index en el genotipo

    def get_performance(self) -> float:
        return EVE.evaluate(self.items, self.height, self.class_index)

    def get_class_name(self):
        class_names = ["Warrior", "Archer", "Guardian", "Mage"]
        return class_names[self.class_index]

    def __str__(self):
        return f"Character: \nClass: {self.get_class_name()} \nItems: {self.items} \nHeight: {self.height}\nTotal Points: {self.total_points}\n"