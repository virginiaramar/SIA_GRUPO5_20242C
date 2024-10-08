import csv
import numpy as np
import matplotlib.pyplot as plt
from Exercice2.linear_perceptron import Perceptron
from Exercice2.nonlinear_perceptron import NonLinearPerceptron


# Load CSV data manually
def load_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    X = []
    y = []

    for row in data[1:]:  # Skipping the header
        values = row.strip().split(',')
        features = [float(values[0]), float(values[1]), float(values[2])]
        target = float(values[3])

        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    y_min, y_max = np.min(y), np.max(y)
    y_normalized = 2 * (y - y_min) / (y_max - y_min) - 1

    return X, y_normalized


def custom_train_test_split(X, y, test_size=0.5):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X, y, model_class, alpha=0.01, n_epochs=20, test_size=0.5):
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=test_size)

    model = model_class(N=X_train.shape[1], alpha=alpha)
    model.fit(X_train, y_train, n_epochs=n_epochs)

    train_predictions = model.predict(X_train)
    mse_train = np.mean((train_predictions - y_train) ** 2)
    mae_train = np.mean(np.abs(train_predictions - y_train))
    r2_train = 1 - (np.sum((train_predictions - y_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))

    test_predictions = model.predict(X_test)
    mse_test = np.mean((test_predictions - y_test) ** 2)
    mae_test = np.mean(np.abs(test_predictions - y_test))
    r2_test = 1 - (np.sum((test_predictions - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    return mse_train, mse_test, mae_train, mae_test, r2_train, r2_test


if __name__ == '__main__':
    X, y = load_data('TP3-ej2-conjunto.csv')

    test_sizes = []
    mse_train_avg = []
    mse_test_avg = []
    mae_train_avg = []
    mae_test_avg = []
    r2_train_avg = []
    r2_test_avg = []

    n_simulations = 100

    for i in range(1, 10):
        ii = i / 10
        test_sizes.append(ii)
        mse_train_total = 0
        mse_test_total = 0
        mae_train_total = 0
        mae_test_total = 0
        r2_train_total = 0
        r2_test_total = 0

        print(f"Running simulations for test size: {ii:.1f}")

        for _ in range(n_simulations):
            mse_train, mse_test, mae_train, mae_test, r2_train, r2_test = train_and_evaluate(
                X, y, NonLinearPerceptron, alpha=0.01, n_epochs=20, test_size=ii
            )
            mse_train_total += mse_train
            mse_test_total += mse_test
            mae_train_total += mae_train
            mae_test_total += mae_test
            r2_train_total += r2_train
            r2_test_total += r2_test

        mse_train_avg.append(mse_train_total / n_simulations)
        mse_test_avg.append(mse_test_total / n_simulations)
        mae_train_avg.append(mae_train_total / n_simulations)
        mae_test_avg.append(mae_test_total / n_simulations)
        r2_train_avg.append(r2_train_total / n_simulations)
        r2_test_avg.append(r2_test_total / n_simulations)

    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, mse_train_avg, label='Average Training MSE', marker='o')
    plt.plot(test_sizes, mse_test_avg, label='Average Test MSE', marker='o')
    plt.title('Average MSE vs Test Size Over 100 Simulations')
    plt.xlabel('Test Size')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(test_sizes)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting the MAE values
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, mae_train_avg, label='Average Training MAE', marker='o')
    plt.plot(test_sizes, mae_test_avg, label='Average Test MAE', marker='o')
    plt.title('Average MAE vs Test Size Over 100 Simulations')
    plt.xlabel('Test Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xticks(test_sizes)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting the R² values
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, r2_train_avg, label='Average Training R²', marker='o')
    plt.plot(test_sizes, r2_test_avg, label='Average Test R²', marker='o')
    plt.title('Average R² vs Test Size Over 100 Simulations')
    plt.xlabel('Test Size')
    plt.ylabel('R²')
    plt.xticks(test_sizes)
    plt.grid(True)
    plt.legend()
    plt.show()
