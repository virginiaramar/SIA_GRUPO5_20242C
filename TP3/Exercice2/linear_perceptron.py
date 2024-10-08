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
            converged = True
            for (x, target) in zip(X, y):
                pred = np.dot(x, self.W)
                error = target - pred
                self.W += self.alpha * error * x
                if abs(error) >= self.epsilon:
                    converged = False
                epoch_loss += error ** 2
            mse = epoch_loss / len(X)
            if converged:
                print(f"Converged after {epoch + 1} epochs with MSE: {mse:.4f}")
                break

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        return [np.dot(x, self.W) for x in X]


if __name__ == '__main__':
    X_AND = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_AND = np.array([-1, -1, -1, 1])

    perceptron_and = Perceptron(N=2)
    perceptron_and.fit(X_AND, y_AND, epochs=10)

    predictions = perceptron_and.predict(X_AND)
    print("Predictions for AND:", predictions)
