import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)  # N is the nb of input features
        self.alpha = alpha

    def step_activation(self, x):
        return 1 if x >= 0 else -1

    def fit(self, X, y, n_epochs=10):
        # column of 1s to account for the bias
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in range(n_epochs):
            for (x, target) in zip(X, y):
                p = self.step_activation(np.dot(x, self.W))
                if target != p:
                    error = target - p
                    self.W += self.alpha * error * x

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        predictions = []
        for x in X:
            pred = self.step_activation(np.dot(x, self.W))
            predictions.append(pred)
        return predictions


if __name__ == '__main__':
    X_AND = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_AND = np.array([-1, -1, -1, 1])
    perceptron_and = Perceptron(N=2)
    perceptron_and.fit(X_AND, y_AND)
    predictions_and = perceptron_and.predict(X_AND)
    print("Predictions AND:", predictions_and)

    X_XOR = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_XOR = np.array([1, 1, -1, -1])

    perceptron_xor = Perceptron(N=2)
    perceptron_xor.fit(X_XOR, y_XOR)
    predictions_xor = perceptron_xor.predict(X_XOR)
    print("Predictions XOR:", predictions_xor)
