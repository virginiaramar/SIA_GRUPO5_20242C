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

def compare_original_vs_reconstruction(X, reconstructed_data, character_labels, num_chars=None, save_path=None):
    """
    Compara los caracteres originales con sus reconstrucciones, con letras negras y fondo blanco.

    Args:
        X (np.ndarray): Datos originales (caracteres).
        reconstructed_data (np.ndarray): Reconstrucciones generadas por el modelo.
        character_labels (list): Etiquetas de los caracteres.
        num_chars (int, optional): Número de caracteres a mostrar. Muestra todos si es None.
        save_path (str, optional): Ruta para guardar el gráfico. Si es None, muestra el gráfico.
    """
    if num_chars is None:
        num_chars = X.shape[0]
    num_chars = min(num_chars, X.shape[0])

    # Crear subgráficos
    fig, axes = plt.subplots(2, num_chars, figsize=(2.5 * num_chars, 10))  # 2 filas, num_chars columnas

    # Ajustar los espacios entre las subgráficas
    plt.subplots_adjust(hspace=0.7, wspace=0.4)  # Más espacio vertical y horizontal

    for i in range(num_chars):
        # Carácter original (invertir colores para letra negra, fondo blanco)
        ax = axes[0, i]
        ax.imshow(1 - X[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(character_labels[i], fontsize=12)  # Títulos más grandes

        # Carácter reconstruido (invertir colores para letra negra, fondo blanco)
        ax = axes[1, i]
        ax.imshow(1 - reconstructed_data[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(character_labels[i], fontsize=12)  # Títulos más grandes

    # Títulos generales
    fig.text(0.5, 0.92, 'Originales', ha='center', va='center', fontsize=30)  # Tamaño de texto aumentado
    fig.text(0.5, 0.47, 'Reconstruidos', ha='center', va='center', fontsize=30)  # Tamaño de texto aumentado

    # Guardar o mostrar el gráfico
    if save_path:
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajustar el espacio para incluir los títulos
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Imagen guardada en: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_latent_space(latent_space, labels=None, save_path=None, save_data_path="latent_space.csv"):
    """
    Grafica el espacio latente con etiquetas opcionales, guarda el gráfico y siempre guarda los datos.

    Args:
        latent_space (np.ndarray): Coordenadas en el espacio latente.
        labels (list, optional): Etiquetas de los puntos en el espacio latente.
        save_path (str, optional): Ruta para guardar el gráfico. Si es None, muestra el gráfico.
        save_data_path (str): Ruta para guardar los datos en formato CSV.
    """
    import pandas as pd

    # Graficar el espacio latente
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(latent_space):
        plt.scatter(point[0], point[1], marker='o')
        label = labels[i] if labels is not None else str(i)
        plt.text(point[0] + 0.02, point[1] + 0.02, label, fontsize=9)
    plt.title("Representación en el Espacio Latente")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True)

    # Guardar o mostrar el gráfico
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Imagen guardada en: {save_path}")
    else:
        plt.show()

    plt.close()  # Cierra la figura para liberar memoria

    # Guardar los datos del espacio latente
    data = {
        "Dimensión 1": latent_space[:, 0],
        "Dimensión 2": latent_space[:, 1],
        "Etiqueta": labels if labels is not None else [f"Punto {i}" for i in range(len(latent_space))]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_data_path, index=False, encoding="utf-8")
    print(f"Datos del espacio latente guardados en: {save_data_path}")

def plot_reconstruction_error(errors, character_labels=None, save_path=None, save_txt_path=None):
    """
    Grafica y guarda el error de reconstrucción por carácter.

    Args:
        errors (np.ndarray): Array con los errores de reconstrucción.
        character_labels (list): Etiquetas de los caracteres.
        save_path (str, optional): Ruta para guardar la gráfica.
        save_txt_path (str, optional): Ruta para guardar los errores en un archivo de texto.
    """
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

    # Guardar la gráfica si se proporciona una ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Imagen guardada en: {save_path}")
    plt.close()

    # Guardar los errores en un archivo de texto si se proporciona una ruta
    if save_txt_path:
        with open(save_txt_path, "w") as file:
            file.write("Carácter\tError\n")  # Encabezado
            for label, error in zip(character_labels, errors):
                file.write(f"{label}\t{error:.6f}\n")
        print(f"Errores guardados en el archivo: {save_txt_path}")

def main():
    # Leer configuración desde config.json
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Cargar datos desde el archivo especificado en la configuración
    font_data = load_font_data(config["file_path"])
    if font_data is None:
        return

    # Etiquetas de caracteres
    character_labels = [
        '`','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~','DEL'
    ]

    # Número de repeticiones
    num_repetitions=config.get("n_runs")
    for repetition in range(1, num_repetitions + 1):
        print(f"\nRepetición {repetition}/{num_repetitions}")
        
        # Inicializar autoencoder con la configuración
        autoencoder = Autoencoder(
            layers=config["layers"],
            activation_fn_name=config["activation_fn"],
            loss_fn_name=config["loss_function"],
            optimizer=config.get("optimizer", "gd"),
            learning_rate=config["learning_rate"],
            initial_lr=config.get("initial_lr"),
            decay_rate=config.get("decay_rate"),
            variable_lr=config.get("variable_lr", False),
            seed=config.get("seed"), 
            init_method=config.get("init_method", "uniform") 
        )

        weights_path = f"autoencoder_weights_repetition_{repetition}.pkl"
        
        # Entrenar autoencoder, pasando la iteración para nombrar los archivos
        losses = autoencoder.train(
            font_data, 
            epochs=config["epochs"], 
            iteration=repetition, 
            save_prefix=f"training_repetition_{repetition}"
        )

        autoencoder.save_weights(weights_path)
        print(f"Pesos guardados en: {weights_path}")

        # Reconstruir datos
        reconstructed_data = autoencoder.reconstruct(font_data)

        compare_original_vs_reconstruction(
            font_data, reconstructed_data, 
            character_labels, 
            num_chars=32,  # Opcional, si quieres limitar el número de caracteres
            save_path=f"comparison_repetition_{repetition}.png"
        )
                
        # Obtener representación en el espacio latente
        latent_space = autoencoder.get_latent_space(font_data)

        # Graficar espacio latente
        plot_latent_space(
            latent_space, 
            labels=character_labels[:len(latent_space)], 
            save_path=f"latent_space_repetition_{repetition}.png"
        )

        # Calcular error de reconstrucción por carácter
        reconstruction_error = np.sum(np.abs(font_data - reconstructed_data), axis=1)

        # Graficar y guardar error de reconstrucción
        plot_reconstruction_error(
            reconstruction_error, 
            character_labels=character_labels,
            save_path=f"reconstruction_error_repetition_{repetition}.png",
            save_txt_path=f"reconstruction_errors_repetition_{repetition}.txt"
        )


        # Generar nuevas letras desde puntos aleatorios del espacio latente
        print("Generando nuevas letras desde puntos aleatorios del espacio latente...")
        latent_dim = autoencoder.layers[autoencoder.latent_layer_index]
        num_new_letters = 10  # Número de nuevas letras a generar
        random_latent_points = autoencoder.generate_latent_points_in_grid(
            num_points=num_new_letters,
            dim1_range=(0, 100),  # Rango para Dim1
            dim2_range=(-10, 250)  # Rango para Dim2
        )

        # Mostrar los puntos generados (x, y)
        print("Puntos generados en el espacio latente:")
        for i, (x, y) in enumerate(random_latent_points):
            print(f"Punto {i+1}: x={x:.2f}, y={y:.2f}")

        # Decodificar los puntos latentes
        generated_letters = autoencoder.decode_latent_points(random_latent_points)

        # Visualizar las nuevas letras generadas
        fig, axes = plt.subplots(1, num_new_letters, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(generated_letters[i].reshape(7, 5), cmap="gray")
            ax.axis("off")
            ax.set_title(f"Letra {i+1}")
        plt.savefig(f"generated_letters_repetition_{repetition}.png", dpi=300)
        print(f"Letras generadas guardadas en: generated_letters_repetition_{repetition}.png")
        plt.close()    


        # Obtener representación en el espacio latente
        latent_space = autoencoder.get_latent_space(font_data)

        # Visualizar el espacio latente combinado con puntos generados
        autoencoder.plot_latent_space_with_generated(
            training_latent_points=latent_space,
            character_labels=character_labels[:len(latent_space)],
            num_generated_points=10,  # Número de nuevas letras a generar
            save_path="combined_latent_space_plot.png"  # Ruta para guardar el gráfico
        )

    print(f"\nFinalizado: Se completaron las {num_repetitions} repeticiones.")

if __name__ == "__main__":
    main()



