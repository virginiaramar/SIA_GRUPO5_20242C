import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


class NonLinearPerceptron:
    def __init__(self, N, alpha=0.1):
        # Initialize weights
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def fit(self, X, y, n_epochs=10):
        # Add bias term
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(n_epochs):
            for (x, target) in zip(X, y):
                p = tanh(np.dot(x, self.W))
                error = target - p
                self.W += self.alpha * error * tanh_derivative(p) * x

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        predictions = []

        for x in X:
            pred = tanh(np.dot(x, self.W))
            # Threshold the sigmoid output at 0.5
            predictions.append(1 if pred >= 0 else -1)
        return np.array(predictions)
