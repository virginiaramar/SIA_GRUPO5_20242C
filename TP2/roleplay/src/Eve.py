class Eve:
    def compute_performance_score(self, character):
        performance_score = 0
        if character.character_class == "Warrior":
            performance_score = 0.6 * character.attack + 0.4 * character.defense
        elif character.character_class == "Archer":
            performance_score = 0.9 * character.attack + 0.1 * character.defense
        elif character.character_class == "Guardian":
            performance_score = 0.1 * character.attack + 0.9 * character.defense
        elif character.character_class == "Mage":
            performance_score = 0.8 * character.attack + 0.3 * character.defense
        return performance_score
