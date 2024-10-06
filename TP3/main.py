import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import seaborn as sns


from multilayer_perceptron import multilayer_perceptron 

def plot_decision_boundary(mlp, X, y):
    # Define plot limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict for each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=ListedColormap(cmap_bold), edgecolor='k', s=100)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary learned by MLP")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

def run_xor_exercise():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    mlp = multilayer_perceptron('config.json')
    mlp.multilayer_algorithm(X, y)
    mlp.evaluate(X, y)
    plot_decision_boundary(mlp, X, y)

def run_3b_exercise():
    data = np.genfromtxt('data/TP3-ej3-digitos.txt', delimiter=' ')
    data_flatten = np.array([data[i:i+7].flatten() for i in range(0, 70, 7)])
    labels = np.eye(10)  # One-hot encoding for 10 classes

    np.savetxt('data/digits_flatten.txt', data_flatten, fmt='%d', delimiter=' ')

    perceptron = multilayer_perceptron(config_file='config.json')
    print("Entrenando la red neuronal...")
    perceptron.multilayer_algorithm(data_flatten, labels)
    print("Evaluando la red neuronal...")
    perceptron.evaluate(data_flatten, labels)

def run_3c_exercise():
    data = np.genfromtxt('data/TP3-ej3-digitos.txt', delimiter=' ')
    data_flatten = np.array([data[i:i+7].flatten() for i in range(0, 70, 7)])
    labels = np.eye(10)  # One-hot encoding for 10 classes

    perceptron = multilayer_perceptron(config_file='config.json')

    print("Entrenando la red neuronal para reconocimiento de dígitos...")
    perceptron.multilayer_algorithm(data_flatten, labels)

    print("Evaluando la red neuronal...")
    accuracy = perceptron.evaluate(data_flatten, labels)

    # Generar predicciones
    predictions = perceptron.predict(data_flatten)

    # Crear heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
    plt.title('Mapa de calor de predicciones de dígitos')
    plt.xlabel('Dígito predicho')
    plt.ylabel('Dígito real')
    
    # Agregar texto con la suma de la diagonal
    diagonal_sum = np.trace(predictions)
    plt.text(0.5, 1.05, f'Suma de la diagonal: {diagonal_sum:.2f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12, color='red')

    # Mostrar información sobre la arquitectura
    plt.text(0.5, 1.1, f'Capa 0: {perceptron.architecture[0]} inputs\n'
                       f'Capa 1: {perceptron.architecture[1]} nodos\n'
                       f'Capa Final: {perceptron.architecture[-1]} Outputs', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig('heatmap_digitos.png')
    plt.show()

    print("Probando con datos ruidosos...")
    noisy_data = add_noise(data_flatten)
    noisy_accuracy = perceptron.evaluate(noisy_data, labels)

    # Generar heatmap para datos ruidosos
    noisy_predictions = perceptron.predict(noisy_data)
    plt.figure(figsize=(12, 10))
    sns.heatmap(noisy_predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
    plt.title('Mapa de calor de predicciones de dígitos (datos ruidosos)')
    plt.xlabel('Dígito predicho')
    plt.ylabel('Dígito real')
    
    # Agregar texto con la suma de la diagonal
    noisy_diagonal_sum = np.trace(noisy_predictions)
    plt.text(0.5, 1.05, f'Suma de la diagonal: {noisy_diagonal_sum:.2f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig('heatmap_digitos_ruidosos.png')
    plt.show()

    print("Heatmaps saved as 'heatmap_digitos.png' and 'heatmap_digitos_ruidosos.png'")

def add_noise(data, noise_level=0.1):
    noisy_data = data.copy()
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = np.clip(noisy_data + noise, 0, 1)
    return noisy_data

if __name__ == "__main__":
    exercise = int(input("Ingrese el número de ejercicio a ejecutar (1, 2 o 3): "))
    
    if exercise == 1:
        run_xor_exercise()
    elif exercise == 2:
        run_3b_exercise()
    elif exercise == 3:
        run_3c_exercise()