import re
import numpy as np
import matplotlib.pyplot as plt
import json

# Función para leer el archivo font.h
def read_font_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {file_path}.")
        return []

    pattern = re.compile(r'\{(.*?)\}')
    font_patterns = []
    
    for line in lines:
        match = pattern.search(line)
        if match:
            hex_values = [int(value.strip(), 16) for value in match.group(1).split(',')]
            font_patterns.append(hex_values)
    
    return font_patterns

# Convertir cada carácter a un vector binario de 35 elementos
def preprocess_patterns(patterns):
    binary_patterns = []
    for pattern in patterns:
        binary_pattern = [int(bit) for hex_val in pattern for bit in f"{hex_val:05b}"]
        binary_patterns.append(binary_pattern)
    return np.array(binary_patterns, dtype=float)

# Inicialización de pesos y sesgos con Xavier
def initialize_weights(input_dim, hidden_dim, latent_dim):
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(1 / hidden_dim)
    b2 = np.zeros((1, latent_dim))
    W3 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(1 / latent_dim)
    b3 = np.zeros((1, hidden_dim))
    W4 = np.random.randn(hidden_dim, input_dim) * np.sqrt(1 / hidden_dim)
    b4 = np.zeros((1, input_dim))
    return W1, b1, W2, b2, W3, b3, W4, b4

# Funciones de activación
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Funciones de pérdida
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return y_pred - y_true

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_loss_derivative(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Propagación hacia adelante
def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, activation_fn):
    Z1 = np.dot(X, W1) + b1
    A1 = activation_fn(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = activation_fn(Z2)  # Espacio latente
    Z3 = np.dot(A2, W3) + b3
    A3 = activation_fn(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = sigmoid(Z4)  # Salida con sigmoid para probabilidades
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "Z4": Z4, "A4": A4}

# Retropropagación
def backward_propagation(X, outputs, W2, W3, W4, activation_derivative_fn, loss_derivative_fn):
    A1, A2, A3, A4 = outputs["A1"], outputs["A2"], outputs["A3"], outputs["A4"]
    dA4 = loss_derivative_fn(X, A4)
    dZ4 = dA4 * sigmoid_derivative(A4)
    dW4 = np.dot(A3.T, dZ4) / X.shape[0]
    db4 = np.sum(dZ4, axis=0, keepdims=True) / X.shape[0]
    
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * activation_derivative_fn(A3)
    dW3 = np.dot(A2.T, dZ3) / X.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]
    
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * activation_derivative_fn(A2)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * activation_derivative_fn(A1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3, "dW4": dW4, "db4": db4}

# Actualización de parámetros
def update_parameters(params, grads, learning_rate):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    params["W3"] -= learning_rate * grads["dW3"]
    params["b3"] -= learning_rate * grads["db3"]
    params["W4"] -= learning_rate * grads["dW4"]
    params["b4"] -= learning_rate * grads["db4"]
    return params

# Decaimiento exponencial del learning rate
def exponential_decay_lr(epoch, initial_lr, decay_rate):
    return initial_lr * (decay_rate ** epoch)

# Bucle de entrenamiento con tasa de aprendizaje variable
def train_autoencoder(X, params, learning_rate, epochs, activation_fn, activation_derivative_fn, loss_fn, loss_derivative_fn, variable_lr=None):
    """
    Entrena un autoencoder con la opción de tasa de aprendizaje variable.
    
    Args:
        X (np.ndarray): Datos de entrada.
        params (dict): Diccionario con los pesos y sesgos iniciales.
        learning_rate (float): Tasa de aprendizaje inicial.
        epochs (int): Número de iteraciones de entrenamiento.
        activation_fn (function): Función de activación.
        activation_derivative_fn (function): Derivada de la función de activación.
        loss_fn (function): Función de pérdida.
        loss_derivative_fn (function): Derivada de la función de pérdida.
        variable_lr (function): Función para calcular tasa de aprendizaje variable (opcional).
    
    Returns:
        dict: Parámetros entrenados.
        list: Lista de pérdidas (losses) durante el entrenamiento.
    """
    losses = []
    for epoch in range(epochs):
        # Calcula el learning rate actual
        lr = variable_lr(epoch) if variable_lr else learning_rate

        # Propagación hacia adelante
        outputs = forward_propagation(X, params["W1"], params["b1"], 
                                      params["W2"], params["b2"], 
                                      params["W3"], params["b3"], 
                                      params["W4"], params["b4"], 
                                      activation_fn)
        
        # Calcular la pérdida
        reconstruction = outputs["A4"]
        loss = loss_fn(X, reconstruction)
        losses.append(loss)

        # Retropropagación
        grads = backward_propagation(X, outputs, params["W2"], params["W3"], params["W4"], activation_derivative_fn, loss_derivative_fn)
        
        # Actualizar parámetros
        params = update_parameters(params, grads, lr)

        # Registro de progreso
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Época {epoch + 1}/{epochs}, Pérdida: {loss:.6f}, Learning Rate: {lr:.6f}")
    
    return params, losses
# Visualizar comparación entre los caracteres originales y reconstrucciones
def compare_original_vs_reconstruction(X, reconstructed_data, num_chars=3):
    """
    Compara los caracteres originales con sus reconstrucciones.
    
    Args:
        X (np.ndarray): Datos originales.
        reconstructed_data (np.ndarray): Reconstrucciones generadas.
        num_chars (int): Número de caracteres a visualizar.
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_chars):
        # Carácter original
        plt.subplot(num_chars, 2, 2 * i + 1)
        plt.imshow(X[i].reshape(7, 5), cmap="binary", interpolation="nearest")
        plt.title(f"Original - Índice {i}")
        plt.axis("off")
        
        # Carácter reconstruido
        plt.subplot(num_chars, 2, 2 * i + 2)
        plt.imshow(reconstructed_data[i].reshape(7, 5), cmap="binary", interpolation="nearest")
        plt.title(f"Reconstrucción - Índice {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Leer configuraciones desde el archivo JSON
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Leer y procesar los patrones
file_path = config["file_path"]
patterns = read_font_file(file_path)
font_data = preprocess_patterns(patterns)

# Configurar dimensiones del modelo
input_dim = config["input_dim"]
hidden_dim = config["hidden_dim"]
latent_dim = config["latent_dim"]

# Inicializar pesos y sesgos
W1, b1, W2, b2, W3, b3, W4, b4 = initialize_weights(input_dim, hidden_dim, latent_dim)

# Seleccionar funciones de activación
if config["activation_fn"] == "tanh":
    activation_fn = tanh
    activation_derivative_fn = tanh_derivative
elif config["activation_fn"] == "relu":
    activation_fn = relu
    activation_derivative_fn = relu_derivative
elif config["activation_fn"] == "sigmoid":
    activation_fn = sigmoid
    activation_derivative_fn = sigmoid_derivative
else:
    raise ValueError(f"Función de activación desconocida: {config['activation_fn']}")

# Seleccionar función de pérdida
if config["loss_function"] == "mse":
    loss_fn = mse_loss
    loss_derivative_fn = mse_loss_derivative
elif config["loss_function"] == "cross_entropy":
    loss_fn = cross_entropy_loss
    loss_derivative_fn = cross_entropy_loss_derivative
else:
    raise ValueError(f"Función de pérdida desconocida: {config['loss_function']}")

# Entrenamiento
learning_rate = config["learning_rate"]
initial_lr = config["initial_lr"]
decay_rate = config["decay_rate"]
variable_lr = None

if config["variable_lr"]:
    variable_lr = lambda epoch: exponential_decay_lr(epoch, initial_lr, decay_rate)
    
epochs = config["epochs"]

params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}

trained_params, training_losses = train_autoencoder(
    font_data, 
    params, 
    learning_rate, 
    epochs, 
    activation_fn, 
    activation_derivative_fn, 
    loss_fn, 
    loss_derivative_fn, 
    variable_lr=variable_lr  # Aquí se pasa la función variable_lr configurada
)

# Determinar detalles del learning rate
if config["variable_lr"]:
    lr_info = f"Variable (Inicial: {config['initial_lr']}, Decay Rate: {config['decay_rate']})"
else:
    lr_info = f"Constante ({config['learning_rate']})"

# Graficar la pérdida con detalles
plt.figure(figsize=(8, 6))
plt.plot(training_losses, label="Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title(f"Progreso del Entrenamiento\nFunción de Pérdida: {config['loss_function'].replace('_', ' ').title()} | Learning Rate: {lr_info}")
plt.legend()
plt.show()

# Evaluar reconstrucciones
outputs = forward_propagation(font_data, trained_params["W1"], trained_params["b1"], 
                              trained_params["W2"], trained_params["b2"], 
                              trained_params["W3"], trained_params["b3"], 
                              trained_params["W4"], trained_params["b4"], 
                              activation_fn)
reconstructed_data = outputs["A4"]

# Comparar los tres primeros caracteres
compare_original_vs_reconstruction(font_data, reconstructed_data, num_chars=3)