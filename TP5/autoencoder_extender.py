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
def initialize_weights(input_dim, hidden_dim1, hidden_dim2, latent_dim):
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim1) * np.sqrt(1 / input_dim)
    b1 = np.zeros((1, hidden_dim1))
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(1 / hidden_dim1)
    b2 = np.zeros((1, hidden_dim2))
    W3 = np.random.randn(hidden_dim2, latent_dim) * np.sqrt(1 / hidden_dim2)
    b3 = np.zeros((1, latent_dim))
    W4 = np.random.randn(latent_dim, hidden_dim2) * np.sqrt(1 / latent_dim)
    b4 = np.zeros((1, hidden_dim2))
    W5 = np.random.randn(hidden_dim2, hidden_dim1) * np.sqrt(1 / hidden_dim2)
    b5 = np.zeros((1, hidden_dim1))
    W6 = np.random.randn(hidden_dim1, input_dim) * np.sqrt(1 / hidden_dim1)
    b6 = np.zeros((1, input_dim))
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6

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


    return y_pred - y_true

def backward_propagation(X, outputs, W3, W4, W5, W6, activation_derivative_fn, loss_derivative_fn):
    """
    Realiza la retropropagación para calcular los gradientes utilizando la arquitectura extendida.
    
    Args:
        X (np.ndarray): Datos de entrada.
        outputs (dict): Salidas de la propagación hacia adelante.
        W3, W4, W5, W6: Pesos relevantes para calcular los gradientes.
        activation_derivative_fn (function): Derivada de la función de activación.
        loss_derivative_fn (function): Derivada de la función de pérdida.
    
    Returns:
        dict: Gradientes para actualizar pesos y sesgos.
    """
    A1, A2, A3, A4, A5, A6 = outputs["A1"], outputs["A2"], outputs["A3"], outputs["A4"], outputs["A5"], outputs["A6"]

    # Gradiente en la salida
    dA6 = loss_derivative_fn(X, A6)
    dZ6 = dA6 * sigmoid_derivative(A6)
    dW6 = np.dot(A5.T, dZ6) / X.shape[0]
    db6 = np.sum(dZ6, axis=0, keepdims=True) / X.shape[0]

    # Gradientes de la capa 5
    dA5 = np.dot(dZ6, W6.T)
    dZ5 = dA5 * activation_derivative_fn(A5)
    dW5 = np.dot(A4.T, dZ5) / X.shape[0]
    db5 = np.sum(dZ5, axis=0, keepdims=True) / X.shape[0]

    # Gradientes de la capa 4
    dA4 = np.dot(dZ5, W5.T)
    dZ4 = dA4 * activation_derivative_fn(A4)
    dW4 = np.dot(A3.T, dZ4) / X.shape[0]
    db4 = np.sum(dZ4, axis=0, keepdims=True) / X.shape[0]

    # Gradientes de la capa 3
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * activation_derivative_fn(A3)
    dW3 = np.dot(A2.T, dZ3) / X.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]

    # Gradientes de la capa 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * activation_derivative_fn(A2)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

    # Gradientes de la capa 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * activation_derivative_fn(A1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    return {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3,
        "dW4": dW4, "db4": db4,
        "dW5": dW5, "db5": db5,
        "dW6": dW6, "db6": db6
    }

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, activation_fn):
    """
    Realiza la propagación hacia adelante en el autoencoder extendido.

    Args:
        X (np.ndarray): Datos de entrada.
        W1, b1, W2, b2, ..., W6, b6: Pesos y sesgos del autoencoder.
        activation_fn (function): Función de activación para las capas ocultas.

    Returns:
        dict: Salidas intermedias y finales de cada capa.
    """
    # Codificador
    Z1 = np.dot(X, W1) + b1
    A1 = activation_fn(Z1)  # Primera capa oculta

    Z2 = np.dot(A1, W2) + b2
    A2 = activation_fn(Z2)  # Segunda capa oculta

    Z3 = np.dot(A2, W3) + b3
    A3 = activation_fn(Z3)  # Espacio latente

    # Decodificador
    Z4 = np.dot(A3, W4) + b4
    A4 = activation_fn(Z4)  # Primera capa oculta del decodificador

    Z5 = np.dot(A4, W5) + b5
    A5 = activation_fn(Z5)  # Segunda capa oculta del decodificador

    Z6 = np.dot(A5, W6) + b6
    A6 = sigmoid(Z6)  # Salida final con sigmoid (reconstrucción)

    return {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3,
        "Z4": Z4, "A4": A4,
        "Z5": Z5, "A5": A5,
        "Z6": Z6, "A6": A6
    }

# Actualización de parámetros para la nueva arquitectura
def update_parameters(params, grads, learning_rate):
    """
    Actualiza los pesos y sesgos usando gradientes descendentes para la arquitectura extendida.

    Args:
        params (dict): Diccionario de pesos y sesgos actuales.
        grads (dict): Gradientes calculados.
        learning_rate (float): Tasa de aprendizaje.

    Returns:
        dict: Parámetros actualizados.
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    params["W3"] -= learning_rate * grads["dW3"]
    params["b3"] -= learning_rate * grads["db3"]
    params["W4"] -= learning_rate * grads["dW4"]
    params["b4"] -= learning_rate * grads["db4"]
    params["W5"] -= learning_rate * grads["dW5"]
    params["b5"] -= learning_rate * grads["db5"]
    params["W6"] -= learning_rate * grads["dW6"]
    params["b6"] -= learning_rate * grads["db6"]
    return params

# Decaimiento exponencial del learning rate
def exponential_decay_lr(epoch, initial_lr, decay_rate):
    return initial_lr * (decay_rate ** epoch)

# Bucle de entrenamiento con tasa de aprendizaje variable
def train_autoencoder(X, params, learning_rate, epochs, activation_fn, activation_derivative_fn, loss_fn, loss_derivative_fn, variable_lr=None):
    """
    Entrena un autoencoder con la nueva arquitectura extendida y la opción de tasa de aprendizaje variable.
    
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
                                      params["W5"], params["b5"], 
                                      params["W6"], params["b6"], 
                                      activation_fn)
        
        # Calcular la pérdida
        reconstruction = outputs["A6"]  # La salida final es A6
        loss = loss_fn(X, reconstruction)
        losses.append(loss)

        # Retropropagación
        grads = backward_propagation(X, outputs, params["W3"], params["W4"], params["W5"], params["W6"], activation_derivative_fn, loss_derivative_fn)
        
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

# Función para comparar visualmente
def compare_characters(indices, title):
    """
    Visualiza los caracteres originales y sus reconstrucciones para índices dados.
    
    Args:
        indices (list): Lista de índices de los caracteres a comparar.
        title (str): Título del gráfico.
    """
    num_chars = len(indices)
    plt.figure(figsize=(12, 4 * num_chars))

    for i, idx in enumerate(indices):
        # Carácter original
        plt.subplot(num_chars, 2, 2 * i + 1)
        plt.imshow(font_data[idx].reshape(7, 5), cmap="binary", interpolation="nearest")
        plt.title(f"{title} - Original (Índice {idx})")
        plt.axis("off")
        
        # Carácter reconstruido
        plt.subplot(num_chars, 2, 2 * i + 2)
        plt.imshow(reconstructed_data[idx].reshape(7, 5), cmap="binary", interpolation="nearest")
        plt.title(f"{title} - Reconstrucción (Índice {idx})")
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
input_dim = config["input_dim"]        # Dimensión de entrada (35)
hidden_dim1 = config["hidden_dim1"]    # Dimensión de la primera capa oculta (16)
hidden_dim2 = config["hidden_dim2"]    # Dimensión de la segunda capa oculta (8)
latent_dim = config["latent_dim"]      # Dimensión del espacio latente (2)

# Inicializar pesos y sesgos
W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, latent_dim)

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

params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5, "W6": W6, "b6": b6}

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
                              trained_params["W5"], trained_params["b5"], 
                              trained_params["W6"], trained_params["b6"], 
                              activation_fn)
reconstructed_data = outputs["A6"]  # Ahora la salida final es A6

# Comparar los tres primeros caracteres
compare_original_vs_reconstruction(font_data, reconstructed_data, num_chars=3)

# Espacio latente
latent_space = outputs["A3"]  # La salida del espacio latente es A3 ahora

plt.figure(figsize=(8, 6))
plt.scatter(latent_space[:, 0], latent_space[:, 1], c=np.arange(len(latent_space)), cmap="tab20", s=50)
plt.colorbar(label="Índice de Carácter")
plt.title("Representación en el Espacio Latente")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.grid()
plt.show()

# Error de reconstrucción por carácter
reconstruction_error = np.sum(np.abs(font_data - reconstructed_data), axis=1) / font_data.shape[1]

# Detectar los tres mejores y los tres peores
best_indices = np.argsort(reconstruction_error)[:3]  # Tres mejores (menor error)
worst_indices = np.argsort(reconstruction_error)[-3:]  # Tres peores (mayor error)

print("Tres mejores caracteres (menor error):", best_indices)
print("Tres peores caracteres (mayor error):", worst_indices)

# Comparar los tres mejores
compare_characters(best_indices, "Mejores")

# Comparar los tres peores
compare_characters(worst_indices, "Peores")


plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(reconstruction_error)), reconstruction_error)
plt.axhline(1, color='red', linestyle='--', label="Límite de 1 píxel")
plt.title("Error de Reconstrucción por Carácter")
plt.xlabel("Índice de Carácter")
plt.ylabel("Error Promedio por Píxel")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(reconstruction_error, bins=10, alpha=0.7, color="blue")
plt.axvline(1, color='red', linestyle='--', label="Límite de 1 píxel")
plt.title("Distribución del Error de Reconstrucción")
plt.xlabel("Error Promedio por Píxel")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()