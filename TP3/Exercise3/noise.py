import numpy as np

class NoiseGenerator:
    def add_noise(self, data, noise_level=0.1):
        noisy_data = data.copy()
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = np.clip(noisy_data + noise, 0, 1)
        return noisy_data

    def add_salt_and_pepper_noise(self, data, salt_prob=0.05, pepper_prob=0.05):
        noisy_data = data.copy()
        salt_mask = np.random.choice([0, 1], size=data.shape, p=[1 - salt_prob, salt_prob])
        pepper_mask = np.random.choice([0, 1], size=data.shape, p=[1 - pepper_prob, pepper_prob])
        noisy_data[salt_mask == 1] = 1
        noisy_data[pepper_mask == 1] = 0
        return noisy_data

    def add_50_percent_noise(self, data):
        noisy_data = data.copy()
        mask = np.random.choice([0, 1], size=data.shape, p=[0.5, 0.5])
        noise = np.random.rand(*data.shape)
        noisy_data[mask == 1] = noise[mask == 1]
        return noisy_data

    def add_20_percent_noise(self, data):
        noisy_data = data.copy()
        mask = np.random.choice([0, 1], size=data.shape, p=[0.8, 0.2])
        noise = np.random.rand(*data.shape)
        noisy_data[mask == 1] = noise[mask == 1]
        return noisy_data

    def add_100_percent_noise(self, data):
        noisy_data = data.copy()
        noise = np.random.rand(*data.shape)
        noisy_data = noise
        return noisy_data
