import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # Initialize weights with random values
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step_activation(self, x):
        return 1 if x >= 0 else -1

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]  # Add bias term
        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                pred = self.step_activation(np.dot(x, self.W))
                error = target - pred
                self.W += self.alpha * error * x

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]  # Add bias term
        return [self.step_activation(np.dot(x, self.W)) for x in X]
