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


def calculate_metrics(y_true, y_pred):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Accuracy
    accuracy = np.mean(y_true_classes == y_pred_classes)

    # Precision, Recall, F1-Score
    num_classes = y_true.shape[1]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for class_idx in range(num_classes):
        true_positives = np.sum((y_true_classes == class_idx) & (y_pred_classes == class_idx))
        false_positives = np.sum((y_true_classes != class_idx) & (y_pred_classes == class_idx))
        false_negatives = np.sum((y_true_classes == class_idx) & (y_pred_classes != class_idx))

        precision[class_idx] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall[class_idx] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1[class_idx] = 2 * (precision[class_idx] * recall[class_idx]) / (precision[class_idx] + recall[class_idx]) if (precision[class_idx] + recall[class_idx]) > 0 else 0

    # Weighted average for multiclass
    class_counts = np.sum(y_true, axis=0)
    weights = class_counts / np.sum(class_counts)
    precision_weighted = np.sum(precision * weights)
    recall_weighted = np.sum(recall * weights)
    f1_weighted = np.sum(f1 * weights)

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    return accuracy, precision_weighted, recall_weighted, f1_weighted, mse

def run_3c_exercise(noise_type=None):
    # Cargar los datos originales
    data = np.genfromtxt('data/TP3-ej3-digitos.txt', delimiter=' ')
    data_flatten = np.array([data[i:i+7].flatten() for i in range(0, 70, 7)])
    labels = np.eye(10)  # One-hot encoding for 10 clases

    # Inicializar el perceptrón
    perceptron = multilayer_perceptron(config_file='config.json')
    
    print("Entrenando la red neuronal con datos originales (sin ruido)...")
    
    # Listas para almacenar métricas durante el entrenamiento
    train_accuracy_history = []
    train_precision_history = []
    train_recall_history = []
    train_f1_history = []
    train_mse_history = []
    
    # Si se especifica un tipo de ruido, generar datos ruidosos
    if noise_type is not None:
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
        
        # Listas para almacenar métricas de generalización (datos ruidosos)
        gen_accuracy_history = []
        gen_precision_history = []
        gen_recall_history = []
        gen_f1_history = []
        gen_mse_history = []

    for epoch in range(perceptron.epochs):
        # Entrenar una época
        perceptron._train_epoch(data_flatten, labels)
        
        # Calcular métricas de entrenamiento
        train_predictions = perceptron.predict(data_flatten)
        train_accuracy, train_precision, train_recall, train_f1, train_mse = calculate_metrics(labels, train_predictions)
        
        train_accuracy_history.append(train_accuracy)
        train_precision_history.append(train_precision)
        train_recall_history.append(train_recall)
        train_f1_history.append(train_f1)
        train_mse_history.append(train_mse)
        
        # Calcular métricas de generalización si hay datos ruidosos
        if noise_type is not None:
            gen_predictions = perceptron.predict(noisy_data)
            gen_accuracy, gen_precision, gen_recall, gen_f1, gen_mse = calculate_metrics(labels, gen_predictions)
            
            gen_accuracy_history.append(gen_accuracy)
            gen_precision_history.append(gen_precision)
            gen_recall_history.append(gen_recall)
            gen_f1_history.append(gen_f1)
            gen_mse_history.append(gen_mse)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train MSE: {train_mse:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Generar gráficos individuales para cada métrica
    metrics = [
        ('Accuracy', train_accuracy_history, gen_accuracy_history if noise_type else None),
        ('Precision', train_precision_history, gen_precision_history if noise_type else None),
        ('Recall', train_recall_history, gen_recall_history if noise_type else None),
        ('F1-Score', train_f1_history, gen_f1_history if noise_type else None),
        ('MSE', train_mse_history, gen_mse_history if noise_type else None)
    ]

    for metric_name, train_metric, gen_metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metric, label='Train')
        if gen_metric:
            plt.plot(gen_metric, label='Generalization (Noisy)')
        plt.title(f'Model {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'model_{metric_name.lower().replace("-", "_")}.png')
        plt.show()
        print(f"Gráfica de {metric_name} guardada como 'model_{metric_name.lower().replace('-', '_')}.png'.")

    # Evaluación final y heatmap para datos originales
    print("\nEvaluando la red neuronal con datos originales...")
    final_predictions = perceptron.predict(data_flatten)
    accuracy, precision, recall, f1, mse = calculate_metrics(labels, final_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")

    # Generar heatmap para datos originales
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
    plt.title('Mapa de calor de predicciones de dígitos (datos originales)')
    plt.xlabel('Dígito predicho')
    plt.ylabel('Dígito real')
    diagonal_sum = np.trace(final_predictions)
    plt.text(0.5, 1.05, f'Suma de la diagonal: {diagonal_sum:.2f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12, color='red')
    plt.tight_layout()
    plt.savefig('heatmap_digitos_originales.png')
    plt.show()
    print("Heatmap de datos originales guardado como 'heatmap_digitos_originales.png'.")
    
    # Evaluación final y heatmap para datos ruidosos (si se especificó)
    if noise_type is not None:
        print(f"\nEvaluando con datos ruidosos ({noise_type})...")
        noisy_predictions = perceptron.predict(noisy_data)
        noisy_accuracy, noisy_precision, noisy_recall, noisy_f1, noisy_mse = calculate_metrics(labels, noisy_predictions)
        
        print(f"Accuracy (noisy): {noisy_accuracy:.4f}")
        print(f"Precision (noisy): {noisy_precision:.4f}")
        print(f"Recall (noisy): {noisy_recall:.4f}")
        print(f"F1-Score (noisy): {noisy_f1:.4f}")
        print(f"MSE (noisy): {noisy_mse:.4f}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(noisy_predictions, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probabilidad'})
        plt.title(f'Mapa de calor de predicciones de dígitos (datos ruidosos - {noise_type})')
        plt.xlabel('Dígito predicho')
        plt.ylabel('Dígito real')
        noisy_diagonal_sum = np.trace(noisy_predictions)
        plt.text(0.5, 1.05, f'Suma de la diagonal: {noisy_diagonal_sum:.2f}', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=12, color='red')
        plt.tight_layout()
        plt.savefig('heatmap_digitos_ruidosos.png')
        plt.show()
        print("Heatmap de datos ruidosos guardado como 'heatmap_digitos_ruidosos.png'.")

    return perceptron

if __name__ == "__main__":
    exercise = int(input("Ingrese el número de ejercicio a ejecutar (1, 2 o 3): "))
    
    if exercise == 1:
        run_xor_exercise()
    elif exercise == 2:
        run_3b_exercise()
    elif exercise == 3:
        #noise_type = input("Ingrese el tipo de ruido ('50_percent', '20_percent', '100_percent', 'salt_and_pepper', 'normal' o 'none'): ").strip()
        noise_type='100_percent'
        if noise_type == 'none':
            run_3c_exercise()
        else:
            run_3c_exercise(noise_type)