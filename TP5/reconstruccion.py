import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from utils import (
    relu, relu_derivative, sigmoid, sigmoid_derivative,
    tanh, tanh_derivative, mse_loss, mse_loss_derivative,
    cross_entropy_loss, cross_entropy_loss_derivative
)

# Cargar el espacio latente
latent_space_df = pd.read_csv('C:/Users/raama/Desktop/final_tp5/configs/cfg1/latent_space.csv')

def reconstruct_from_latent(autoencoder, latent_point):
    """
    Reconstruye una imagen (letra) a partir de un punto en el espacio latente.

    Args:
        autoencoder: Modelo del autoencoder entrenado.
        latent_point (np.ndarray): Punto en el espacio latente.

    Returns:
        np.ndarray: Imagen reconstruida.
    """
    activations = {f"A{autoencoder.latent_layer_index}": latent_point}
    
    for i in range(autoencoder.latent_layer_index + 1, len(autoencoder.layers)):
        W = autoencoder.weights[f"W{i}"]
        b = autoencoder.biases[f"b{i}"]
        Z = np.dot(activations[f"A{i-1}"], W) + b
        
        if i == len(autoencoder.layers) - 1:
            A = sigmoid(Z)  # Última capa usa sigmoid
        else:
            A = autoencoder.activation_fn(Z)  # Capas intermedias usan activación seleccionada
        
        activations[f"A{i}"] = A
    
    return activations[f"A{len(autoencoder.layers) - 1}"]

# Ejemplo: Seleccionar un punto del espacio latente para reconstruir
label_to_generate = 'A'  # Cambia esto por la letra que deseas generar
latent_point = latent_space_df[latent_space_df['Etiqueta'] == label_to_generate].iloc[0, :-1].values.reshape(1, -1)

# Reconstrucción de la letra
reconstructed_image = reconstruct_from_latent(Autoencoder, latent_point)

# Visualizar la imagen reconstruida
plt.imshow(reconstructed_image.reshape((28, 28)), cmap="gray")  # Ajusta las dimensiones según tu conjunto de datos
plt.title(f"Letra Reconstruida: {label_to_generate}")
plt.show()