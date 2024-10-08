import numpy as np
import matplotlib.pyplot as plt

from Exercice2.fitting import load_data
from Exercice2.linear_perceptron import Perceptron
from Exercice2.nonlinear_perceptron import NonLinearPerceptron


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def run_simulations(X, y, n_simulations=100, epochs=20):
    mse_linear_list = []
    mse_non_linear_list = []
    mae_linear_list = []
    mae_non_linear_list = []
    r2_linear_list = []
    r2_non_linear_list = []

    for _ in range(n_simulations):
        linear_perceptron = Perceptron(N=X.shape[1], alpha=0.01)
        linear_perceptron.fit(X, y, epochs=epochs)
        predictions_linear = linear_perceptron.predict(X)

        non_linear_perceptron = NonLinearPerceptron(N=X.shape[1], alpha=0.01)
        non_linear_perceptron.fit(X, y, n_epochs=epochs)
        predictions_non_linear = non_linear_perceptron.predict(X)

        mse_linear = np.mean((predictions_linear - y) ** 2)
        mse_non_linear = np.mean((predictions_non_linear - y) ** 2)
        mae_linear = mean_absolute_error(y, predictions_linear)
        mae_non_linear = mean_absolute_error(y, predictions_non_linear)
        r2_linear = r_squared(y, predictions_linear)
        r2_non_linear = r_squared(y, predictions_non_linear)

        mse_linear_list.append(mse_linear)
        mse_non_linear_list.append(mse_non_linear)
        mae_linear_list.append(mae_linear)
        mae_non_linear_list.append(mae_non_linear)
        r2_linear_list.append(r2_linear)
        r2_non_linear_list.append(r2_non_linear)

    return mse_linear_list, mse_non_linear_list, mae_linear_list, mae_non_linear_list, r2_linear_list, r2_non_linear_list


def plot_results(mse_linear, mse_non_linear, mae_linear, mae_non_linear, r2_linear, r2_non_linear):
    iterations = range(1, len(mse_linear) + 1)

    plt.figure(figsize=(14, 8))

    # MSE plot
    plt.subplot(3, 1, 1)
    plt.plot(iterations, mse_linear, label='Linear Perceptron MSE', color='blue', marker='o')
    plt.plot(iterations, mse_non_linear, label='Non-Linear Perceptron MSE', color='orange', marker='x')
    plt.title('MSE Comparison')
    plt.xlabel('Simulation')
    plt.ylabel('MSE')
    plt.legend(loc="upper left")
    plt.legend()

    # MAE plot
    plt.subplot(3, 1, 2)
    plt.plot(iterations, mae_linear, label='Linear Perceptron MAE', color='green', marker='o')
    plt.plot(iterations, mae_non_linear, label='Non-Linear Perceptron MAE', color='red', marker='x')
    plt.title('MAE Comparison')
    plt.xlabel('Simulation')
    plt.ylabel('MAE')
    plt.legend()

    # R² plot
    plt.subplot(3, 1, 3)
    plt.plot(iterations, r2_linear, label='Linear Perceptron R²', color='purple', marker='o')
    plt.plot(iterations, r2_non_linear, label='Non-Linear Perceptron R²', color='brown', marker='x')
    plt.title('R² Comparison')
    plt.xlabel('Simulation')
    plt.ylabel('R²')
    plt.legend()

    plt.tight_layout(pad=5.0)

    plt.show()


if __name__ == '__main__':
    X, y = load_data('TP3-ej2-conjunto.csv')

    mse_linear, mse_non_linear, mae_linear, mae_non_linear, r2_linear, r2_non_linear = run_simulations(X, y,
                                                                                                       n_simulations=100,
                                                                                                       epochs=20)
    print("Average MSE of a linear perceptron", round(sum(mse_linear) / len(mse_linear), 3))
    print("Average MSE of a non linear perceptron", round(sum(mse_non_linear) / len(mse_non_linear), 3))
    print("Average MAE of a linear perceptron", round(sum(mae_linear) / len(mae_linear), 3))
    print("Average MAE of a non linear perceptron", round(sum(mae_non_linear) / len(mae_non_linear), 3))
    print("Average R2 of a linear perceptron", round(sum(r2_linear) / len(r2_linear), 3))
    print("Average R2 of a non linear perceptron", round(sum(r2_non_linear) / len(r2_non_linear), 3))

    plot_results(mse_linear, mse_non_linear, mae_linear, mae_non_linear, r2_linear, r2_non_linear)
