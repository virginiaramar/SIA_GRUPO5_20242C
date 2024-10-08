import csv
import numpy as np
from matplotlib import pyplot as plt

from Exercice2.fitting import load_data
from Exercice2.nonlinear_perceptron import NonLinearPerceptron



def train_fixed_split(X, y, model_class, n_epochs=20, split_ratio=0.8):
    split_idx = int(split_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = model_class(N=X_train.shape[1])
    model.fit(X_train, y_train, n_epochs)

    train_predictions = model.predict(X_train)
    mse_train = np.mean((train_predictions - y_train) ** 2)

    test_predictions = model.predict(X_test)
    mse_test = np.mean((test_predictions - y_test) ** 2)

    return mse_train, mse_test

def cross_validation(X, y, model_class, alpha, n_splits=5, n_epochs=20):
    split_size = len(X) // n_splits
    mses = []

    for i in range(n_splits):
        X_test = X[i * split_size:(i + 1) * split_size]
        y_test = y[i * split_size:(i + 1) * split_size]
        X_train = np.concatenate((X[:i * split_size], X[(i + 1) * split_size:]), axis=0)
        y_train = np.concatenate((y[:i * split_size], y[(i + 1) * split_size:]), axis=0)

        model = model_class(N=X_train.shape[1], alpha=alpha)
        model.fit(X_train, y_train, n_epochs)

        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mses.append(mse)

    return np.mean(mses)

if __name__ == '__main__':
    X, y = load_data('TP3-ej2-conjunto.csv')

    cross_val_mses = []
    fixed_split_mses = []

    for _ in range(100):
        mse_non_linear = cross_validation(X, y, NonLinearPerceptron, alpha=0.01, n_splits=5, n_epochs=20)
        cross_val_mses.append(mse_non_linear)

        mse_train, mse_test = train_fixed_split(X, y, NonLinearPerceptron, n_epochs=20, split_ratio=0.2)
        fixed_split_mses.append(mse_test)

    avg_fixed_split_mse = np.mean(fixed_split_mses)
    avg_cross_val_mse = np.mean(cross_val_mses)
    worst_fixed_split_mse = np.max(fixed_split_mses)
    worst_cross_val_mse = np.max(cross_val_mses)

    print(f'Cross-validation average MSE: {avg_cross_val_mse:.4f}')
    print(f'Fixed split average MSE: {avg_fixed_split_mse:.4f}')
    print(f'Cross-validation worst MSE: {worst_cross_val_mse:.4f}')
    print(f'Fixed split worst MSE: {worst_fixed_split_mse:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 101), fixed_split_mses, label='Fixed Split MSEs', marker='o', linestyle='-', color='blue', markersize=3)
    plt.plot(range(1, 101), cross_val_mses, label='Cross-validation MSEs', marker='x', linestyle='--', color='orange', markersize=3)
    plt.title('MSE for Each Iteration: Fixed Split vs Cross-validation')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
