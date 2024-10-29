import numpy as np
from matplotlib import pyplot as plt


class HopfieldNetwork:
    def train(self, patterns):
        """Train the Hopfield network using the provided patterns."""
        num_patterns = len(patterns)
        num_neurons = patterns[0].size
        self.num_neurons = num_neurons

        self.weights = np.zeros((num_neurons, num_neurons))

        avg_activity = np.mean([np.sum(p) for p in patterns]) / num_neurons

        for pattern in patterns:
            centered_pattern = pattern - avg_activity
            self.weights += np.outer(centered_pattern, centered_pattern)

        np.fill_diagonal(self.weights, 0)

        self.weights /= num_patterns

    def predict(self, test_patterns, iterations=20, threshold=0, async_update=False):
        self.iterations = iterations
        self.threshold = threshold
        self.async_update = async_update

        return [self._run(np.copy(pattern)) for pattern in test_patterns]

    def _run(self, state):
        previous_energy = self._compute_energy(state)

        for _ in range(self.iterations):
            if self.async_update:
                for _ in range(100):
                    i = np.random.randint(self.num_neurons)
                    state[i] = np.sign(np.dot(self.weights[i], state) - self.threshold)
            else:
                state = np.sign(np.dot(self.weights, state) - self.threshold)

            current_energy = self._compute_energy(state)
            if current_energy == previous_energy:
                break
            previous_energy = current_energy

        return state

    def _compute_energy(self, state):
        """Calculate the energy of the given state in the Hopfield network."""
        return -0.5 * np.dot(state, np.dot(self.weights, state))

    def visualize_weights(self, filename="weights.png"):
        plt.figure(figsize=(6, 5))
        weight_plot = plt.imshow(self.weights, cmap='coolwarm')
        plt.colorbar(weight_plot)
        plt.title("Weight Matrix of the Network")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
