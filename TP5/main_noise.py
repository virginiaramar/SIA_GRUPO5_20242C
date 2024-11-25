import json
import numpy as np
import matplotlib.pyplot as plt
import os
from autoencoder_noise import Autoencoder

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

def add_noise(X, noise_type="salt_and_pepper", noise_level=0.2):
    """Aplica ruido a los datos de entrada"""
    noisy_X = X.copy()

    if noise_type == "salt_and_pepper":
        # Añadir ruido tipo sal y pimienta
        num_noise = int(noise_level * X.size)
        salt_indices = np.random.choice(X.size, num_noise // 2, replace=False)
        pepper_indices = np.random.choice(X.size, num_noise // 2, replace=False)

        noisy_X.ravel()[salt_indices] = 1  # Sal (ruido blanco)
        noisy_X.ravel()[pepper_indices] = 0  # Pimienta (negro)

    elif noise_type == "gaussian":
        # Añadir ruido gaussiano
        noise = np.random.normal(0, noise_level, X.shape)
        noisy_X = np.clip(noisy_X + noise, 0, 1)  # Asegurar que los valores estén en el rango [0, 1]

    elif noise_type == "masking":
        # Masking noise (ocultando algunas partes aleatorias de los datos)
        mask = np.random.rand(*X.shape) < noise_level
        noisy_X[mask] = 0  # Poner a cero los valores seleccionados por el mask

    return noisy_X

def compare_original_vs_reconstruction(X, noisy_data, reconstructed_data, character_labels, num_chars=None, save_path=None):
    if num_chars is None:
        num_chars = X.shape[0]
    num_chars = min(num_chars, X.shape[0])

    fig, axes = plt.subplots(3, num_chars, figsize=(2 * num_chars, 10))  # 3 filas, num_chars columnas

    # Ajustar los espacios entre las subgráficas
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(num_chars):
        # Carácter original
        ax = axes[0, i]  # Seleccionamos el subgráfico para la imagen original
        ax.imshow(X[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        # Título para la letra debajo de la imagen original (ajustado cerca)
        ax.text(0.5, -0.2, f"{character_labels[i]}", ha='center', va='top', fontsize=12, transform=ax.transAxes)

        # Carácter con ruido
        ax = axes[1, i]  # Seleccionamos el subgráfico para la imagen con ruido
        ax.imshow(noisy_data[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        # Título para la letra debajo de la imagen con ruido
        ax.text(0.5, -0.2, f"{character_labels[i]}", ha='center', va='top', fontsize=12, transform=ax.transAxes)

        # Carácter reconstruido
        ax = axes[2, i]  # Seleccionamos el subgráfico para la imagen reconstruida
        ax.imshow(reconstructed_data[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        # Título para la letra debajo de la imagen reconstruida (ajustado cerca)
        ax.text(0.5, -0.2, f"{character_labels[i]}", ha='center', va='top', fontsize=12, transform=ax.transAxes)

    # Títulos generales
    fig.text(0.5, 0.92, 'Originales', ha='center', va='bottom', fontsize=14)
    fig.text(0.5, 0.6, 'Con Ruido', ha='center', va='bottom', fontsize=14)
    fig.text(0.5, 0.28, 'Reconstruidos', ha='center', va='bottom', fontsize=14)

    # Guardar o mostrar el gráfico
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.tight_layout()
        plt.show()



def plot_latent_space(latent_space, labels=None, save_path=None):
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(latent_space):
        plt.scatter(point[0], point[1], marker='o')
        label = labels[i] if labels is not None else str(i)
        plt.text(point[0]+0.02, point[1]+0.02, label, fontsize=9)
    plt.title("Representación en el Espacio Latente")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True)

    # Guardar o mostrar el gráfico
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()


def plot_reconstruction_error(errors, character_labels=None, save_path=None):
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

    # Guardar o mostrar el gráfico
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def main():
    # Leer configuración
    with open("config_noise.json", "r") as config_file:
        config = json.load(config_file)

    # Cargar datos
    font_data = load_font_data(config["file_path"])
    if font_data is None:
        return

    # Cargar ruido y aplicar
    noisy_font_data = add_noise(font_data, noise_type=config["noise_type"], noise_level=config["noise_level"])

    # Inicializar autoencoder
    autoencoder = Autoencoder(
        layers=config["layers"],
        activation_fn_name=config["activation_fn"],
        loss_fn_name=config["loss_function"],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        initial_lr=config["initial_lr"],
        decay_rate=config["decay_rate"],
        variable_lr=config["variable_lr"],
        seed=42
    )

    # Cargar etiquetas de caracteres
    character_labels = [
        "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", 
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"
    ]

    # Entrenar autoencoder
    # Entrenar autoencoder
    losses = autoencoder.train(noisy_font_data, font_data, epochs=config["epochs"])

    # Reconstruir datos
    reconstructed_data = autoencoder.reconstruct(noisy_font_data)

    # Comparar caracteres originales y reconstruidos
    compare_original_vs_reconstruction(font_data, noisy_font_data, reconstructed_data, character_labels, save_path="results_1b/or_vs_recons_90salt.png")

    # Obtener representación en el espacio latente
    latent_space = autoencoder.get_latent_space(noisy_font_data)

    # Graficar espacio latente
    plot_latent_space(latent_space, labels=character_labels[:len(latent_space)], save_path="results_1b/latent_space_90salt.png")

    # Calcular error de reconstrucción por carácter
    reconstruction_error = np.sum(np.abs(font_data - reconstructed_data), axis=1)

    # Graficar error de reconstrucción
    plot_reconstruction_error(reconstruction_error, character_labels=character_labels, save_path="results_1b/reconspix_90salt_error.png")

if __name__ == "__main__":
    main()
