import numpy as np
import matplotlib.pyplot as plt


def step_activation(x):
    return 1 if x >= 0 else -1


class Perceptron:
    def __init__(self, N, alpha=0.1, epsilon=1e-4):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        self.epsilon = epsilon

    def fit_with_plot(self, X, y, n_epochs=10, plot_interval=1, title=""):
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in range(n_epochs):
            for (x, target) in zip(X, y):
                p = step_activation(np.dot(x, self.W))
                error = target - p
                if abs(error) > self.epsilon:
                    self.W += self.alpha * error * x
            if epoch % plot_interval == 0:
                self.plot_decision_boundary(X[:, :-1], y, epoch, title)

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        predictions = []
        for x in X:
            pred = step_activation(np.dot(x, self.W))



# Main
if __name__ == '__main__':
    X_AND = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_AND = np.array([-1, -1, -1, 1])

    X_XOR = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_XOR = np.array([1, 1, -1, -1])

    perceptron_and = Perceptron(N=2, epsilon=1e-4)
    perceptron_and.fit_with_plot(X_AND, y_AND, n_epochs=10, plot_interval=1, title="AND Problem")

    perceptron_xor = Perceptron(N=2, epsilon=1e-4)
    perceptron_xor.fit_with_plot(X_XOR, y_XOR, n_epochs=10, plot_interval=1, title="XOR Problem")
