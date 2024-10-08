import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, N, alpha=0.1, epsilon=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)  # N is the number of input features
        self.alpha = alpha
        self.epsilon = epsilon

    def step_activation(self, x):
        return 1 if x >= 0 else -1

    def fit_with_plot(self, X, y, n_epochs=10, plot_interval=1, title=""):
        X = np.c_[X, np.ones((X.shape[0]))]  # Add column of 1s to account for bias
        for epoch in range(n_epochs):
            converged = True
            for (x, target) in zip(X, y):
                p = self.step_activation(np.dot(x, self.W))
                error = target - p
                if error != 0:
                    self.W += self.alpha * error * x
                    if abs(error) > self.epsilon:
                        converged = False
            if converged:
                break
            if epoch % plot_interval == 0:
                self.plot_decision_boundary(X[:, :-1], y, epoch, title)

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        predictions = []
        for x in X:
            pred = self.step_activation(np.dot(x, self.W))
            predictions.append(pred)
        return predictions

    def plot_decision_boundary(self, X, y, epoch, title):
        # Plotting data points
        plt.figure(figsize=(8, 6))
        for i, target in enumerate(y):
            if target == 1:
                plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o', label="Class 1" if i == 0 else "")
            else:
                plt.scatter(X[i, 0], X[i, 1], color='red', marker='x', label="Class -1" if i == 0 else "")

        # Plotting decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = np.c_[xx.ravel(), yy.ravel()]
        Z = np.dot(np.c_[Z, np.ones(Z.shape[0])], self.W)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['red', 'blue'], alpha=0.2)

        # Titles and legends
        plt.title(f"{title} - Epoch {epoch}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()


# Main
if __name__ == '__main__':
    X_AND = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_AND = np.array([-1, -1, -1, 1])

    X_XOR = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_XOR = np.array([1, 1, -1, -1])

    # Train and plot for AND
    perceptron_and = Perceptron(N=2)
    perceptron_and.fit_with_plot(X_AND, y_AND, n_epochs=10, plot_interval=1, title="AND Problem")

    # Train and plot for XOR
    perceptron_xor = Perceptron(N=2)
    perceptron_xor.fit_with_plot(X_XOR, y_XOR, n_epochs=10, plot_interval=1, title="XOR Problem")
