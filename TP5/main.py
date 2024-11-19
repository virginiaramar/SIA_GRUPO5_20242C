import json
import numpy as np
import matplotlib.pyplot as plt
import os
from autoencoder import Autoencoder

def load_font_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return None
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                vector = [int(bit) for bit in line.split()]
                data.append(vector)
    return np.array(data, dtype=float)



def compare_original_vs_reconstruction(X, reconstructed_data, character_labels, num_chars=None):
    if num_chars is None:
        num_chars = X.shape[0]
    num_chars = min(num_chars, X.shape[0])

    fig, axes = plt.subplots(2, num_chars, figsize=(2 * num_chars, 7))  # 2 filas, num_chars columnas

    # Ajustar los espacios entre las subgráficas
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(num_chars):
        # Carácter original
        ax = axes[0, i]  # Seleccionamos el subgráfico para la imagen original
        ax.imshow(X[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        # Título para la letra debajo de la imagen original (ajustado cerca)
        ax.text(0.5, -0.2, f"{character_labels[i]}", ha='center', va='top', fontsize=12, transform=ax.transAxes)

        # Carácter reconstruido
        ax = axes[1, i]  # Seleccionamos el subgráfico para la imagen reconstruida
        ax.imshow(reconstructed_data[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        # Título para la letra debajo de la imagen reconstruida (ajustado cerca)
        ax.text(0.5, -0.2, f"{character_labels[i]}", ha='center', va='top', fontsize=12, transform=ax.transAxes)

    # Títulos generales
    fig.text(0.5, 0.85, 'Originales', ha='center', va='bottom', fontsize=14)
    fig.text(0.5, 0.4, 'Reconstruidos', ha='center', va='bottom', fontsize=14)

    # Mostrar el gráfico con los títulos correctos
    plt.tight_layout()
    plt.show()

def plot_latent_space(latent_space, labels=None):
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(latent_space):
        plt.scatter(point[0], point[1], marker='o')
        label = labels[i] if labels is not None else str(i)
        plt.text(point[0]+0.02, point[1]+0.02, label, fontsize=9)
    plt.title("Representación en el Espacio Latente")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True)
    plt.show()

def plot_reconstruction_error(errors, character_labels=None):
    plt.figure(figsize=(10, 6))
    
    # Si no se proporcionan las etiquetas de los caracteres, usamos los índices numéricos
    if character_labels is None:
        character_labels = [str(i) for i in range(len(errors))]
    
    # Crear el gráfico de barras
    plt.bar(range(len(errors)), errors)

    # Líneas para resaltar el límite de error
    plt.axhline(1, color='red', linestyle='--', label="Límite de 1 píxel")
    
    # Añadir etiquetas en el eje X para cada barra (usando las letras de los caracteres)
    plt.xticks(range(len(errors)), character_labels, rotation=90)

    plt.title("Error de Reconstrucción por Carácter")
    plt.xlabel("Carácter")
    plt.ylabel("Error Total")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Leer configuración
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Cargar datos
    font_data = load_font_data(config["file_path"])
    if font_data is None:
        return

    # Inicializar autoencoder
    autoencoder = Autoencoder(
        layers=config["layers"],
        activation_fn_name=config["activation_fn"],
        loss_fn_name=config["loss_function"],
        optimizer=config.get("optimizer", "gd"),
        learning_rate=config["learning_rate"],
        initial_lr=config.get("initial_lr"),
        decay_rate=config.get("decay_rate"),
        variable_lr=config.get("variable_lr", False)
    )

    # Entrenar autoencoder
    losses = autoencoder.train(font_data, epochs=config["epochs"])

    # Reconstruir datos
    reconstructed_data = autoencoder.reconstruct(font_data)

    character_labels = [
    '`','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~','DEL'
]
    # Comparar caracteres originales y reconstruidos
    compare_original_vs_reconstruction(font_data, reconstructed_data, character_labels)

    # Obtener representación en el espacio latente
    latent_space = autoencoder.get_latent_space(font_data)

    # Generar etiquetas para los caracteres (opcional)
    labels = character_labels[:len(latent_space)] 

    # Graficar espacio latente
    plot_latent_space(latent_space, labels=labels)

    # Calcular error de reconstrucción por carácter
    reconstruction_error = np.sum(np.abs(font_data - reconstructed_data), axis=1)

    # Graficar error de reconstrucción
    plot_reconstruction_error(reconstruction_error, character_labels=character_labels)

if __name__ == "__main__":
    main()



