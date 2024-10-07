import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import seaborn as sns
from noise import NoiseGenerator


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
    mlp.multilayer_algorithm()
    mlp.evaluate()
    plot_decision_boundary(mlp, X, y)
    

def run_3b_exercise():
    data = np.genfromtxt('data/TP3-ej3-digitos.txt', delimiter=' ')
    # Convert the data
    data_flatten = np.array([data[i:i+7].flatten() for i in range(0, 70, 7)])
    np.savetxt('data/digits_flatten.txt', data_flatten, fmt='%d', delimiter=' ')



    perceptron = multilayer_perceptron(config_file='config.json')
    print("Entrenando la red neuronal...")
    perceptron.multilayer_algorithm()
    print("Evaluando la red neuronal...")
    perceptron.evaluate()
    #perceptron.plot_error_history()
    #perceptron.plot_error_history_cross()
    #perceptron.plot_metrics_history()
    perceptron.plot_prediction_comparison()

def run_3c_exercise(noise_type=None):
    # Cargar los datos originales
    data = np.genfromtxt('data/TP3-ej3-digitos.txt', delimiter=' ')
    data_flatten = np.array([data[i:i+7].flatten() for i in range(0, 70, 7)])
    labels = np.eye(10)  # One-hot encoding for 10 clases

    # Entrenamiento y evaluación con datos originales (sin ruido)
    perceptron = multilayer_perceptron(config_file='config.json')
    
    print("Entrenando la red neuronal con datos originales (sin ruido)...")
    perceptron.multilayer_algorithm(data_flatten, labels)
    
    print("Evaluando la red neuronal con datos originales...")
    accuracy = perceptron.evaluate()

    # Generar heatmap para datos originales
    predictions = perceptron.predict(data_flatten)
    plt.figure(figsize=(12, 10))
    sns.heatmap(predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
    plt.title('Mapa de calor de predicciones de dígitos (datos originales)')
    plt.xlabel('Dígito predicho')
    plt.ylabel('Dígito real')
    
    # Agregar texto con la suma de la diagonal
    diagonal_sum = np.trace(predictions)
    plt.text(0.5, 1.05, f'Suma de la diagonal: {diagonal_sum:.2f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig('heatmap_digitos.png')
    plt.show()

    print("Heatmap guardado como 'heatmap_digitos.png'.")
    
    # Si se especifica un tipo de ruido, aplicarlo a los datos
    if noise_type is not None:
        print(f"Probando con datos ruidosos ({noise_type})...")
        noise_generator = NoiseGenerator()
        if noise_type == '50_percent':
            noisy_data = noise_generator.add_50_percent_noise(data_flatten)
        elif noise_type == '20_percent':
            noisy_data = noise_generator.add_20_percent_noise(data_flatten)
        elif noise_type == '100_percent':
            noisy_data = noise_generator.add_100_percent_noise(data_flatten)
        elif noise_type == 'salt_and_pepper':
            noisy_data = noise_generator.add_salt_and_pepper_noise(data_flatten)
        elif noise_type == 'normal':
            noisy_data = noise_generator.add_noise(data_flatten)

        # Evaluación con datos ruidosos
        noisy_accuracy = perceptron.evaluate(noisy_data, labels)

        # Generar heatmap para datos ruidosos
        noisy_predictions = perceptron.predict(noisy_data)
        plt.figure(figsize=(12, 10))
        sns.heatmap(noisy_predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
        plt.title(f'Mapa de calor de predicciones de dígitos (datos ruidosos - {noise_type})')
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

        print("Heatmap guardado como 'heatmap_digitos_ruidosos.png'.")




if __name__ == "__main__":
    exercise = int(input("Ingrese el número de ejercicio a ejecutar (1, 2 o 3): "))
    
    if exercise == 1:
        run_xor_exercise()
    elif exercise == 2:
        run_3b_exercise()
    elif exercise == 3:
        #noise_type = input("Ingrese el tipo de ruido ('50_percent', '20_percent', '100_percent', 'salt_and_pepper', 'normal' o 'none'): ").strip()
        noise_type='normal'
        if noise_type == 'none':
            run_3c_exercise()
        else:
            run_3c_exercise(noise_type)