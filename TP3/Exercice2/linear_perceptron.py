import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.01, epsilon=1e-5):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        self.epsilon = epsilon

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            epoch_loss = 0
            for (x, target) in zip(X, y):
                pred = np.dot(x, self.W)
                error = target - pred
                self.W += self.alpha * error * x
                if abs(error) < self.epsilon:
                    break
                epoch_loss += error ** 2

            mse = epoch_loss / len(X)

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        return [np.dot(x, self.W) for x in X]

