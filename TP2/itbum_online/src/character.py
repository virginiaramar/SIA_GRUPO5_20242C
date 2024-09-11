import random
from src.eve import EVE

class Character:
    def __init__(self, items: dict[str, float], height: float, class_index: int = None, total_points: int = None):
        self.total_points = total_points if total_points is not None else random.randint(100, 200)
        self.items = self._normalize_items(items, self.total_points)
        self.height = height
        self.class_index = class_index if class_index is not None else random.randint(0, 3)

    @staticmethod
    def _normalize_items(items: dict[str, float], total_points: int) -> dict[str, float]:
        total = sum(items.values())
        return {k: v * total_points / total for k, v in items.items()}

    @staticmethod
    def from_genotype(genotype: list[float], total_points: int = None):
        items = {
            "strength": genotype[0],
            "agility": genotype[1],
            "expertise": genotype[2],
            "endurance": genotype[3],
            "health": genotype[4]
        }
        height = genotype[5]
        class_index = int(genotype[6])
        return Character(items, height, class_index, total_points)

    def get_genotype(self) -> list[float]:
        genes = list(self.items.values())
        genes.append(self.height)
        genes.append(float(self.class_index))
        return genes

    def get_performance(self) -> float:
        return EVE.evaluate(self.items, self.height, self.class_index)

    def __str__(self):
        class_names = ["Warrior", "Archer", "Guardian", "Mage"]
        return f"Character: \nClass: {class_names[self.class_index]} \nItems: {self.items} \nHeight: {self.height}\nTotal Points: {self.total_points}\n"