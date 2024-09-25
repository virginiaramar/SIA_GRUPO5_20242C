import csv
import numpy as np

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

    # Normalize y values to range [-1, 1]
    y_min, y_max = np.min(y), np.max(y)
    y_normalized = 2 * (y - y_min) / (y_max - y_min) - 1
    y_threshold = np.where(y_normalized >= 0, 1, -1)

    return X, y_threshold



def cross_validation(X, y, model_class, alpha, n_splits=5, n_epochs=10):
    split_size = len(X) // n_splits
    accuracies = []

    for i in range(n_splits):
        X_test = X[i * split_size:(i + 1) * split_size]
        y_test = y[i * split_size:(i + 1) * split_size]
        X_train = np.concatenate((X[:i * split_size], X[(i + 1) * split_size:]), axis=0)
        y_train = np.concatenate((y[:i * split_size], y[(i + 1) * split_size:]), axis=0)

        model = model_class(N=X_train.shape[1], alpha=alpha)
        model.fit(X_train, y_train, n_epochs)

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print("accuracy",accuracy)
        accuracies.append(accuracy)

    return np.mean(accuracies)


if __name__ == '__main__':
    # Load dataset
    X, y = load_data('TP3-ej2-conjunto.csv')

    accuracy_linear = cross_validation(X, y, Perceptron, alpha=0.1, n_splits=5, n_epochs=10)
    print(f'Linear aerceptron accuracy: {accuracy_linear * 100:.2f}%')

    accuracy_non_linear = cross_validation(X, y, NonLinearPerceptron, alpha=0.1, n_splits=5, n_epochs=10)
    print(f'Non-Linear perceptron accuracy: {accuracy_non_linear * 100:.2f}%')
