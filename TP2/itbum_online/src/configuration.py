import json

class Configuration:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as file:
            self.config = json.load(file)
    
    def get_genetic_algorithm_params(self):
        return self.config['genetic_algorithm']
    
    def get_time_limit(self):
        return self.config['time_limit']