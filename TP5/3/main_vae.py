import json
import numpy as np
from vae import VAE
import matplotlib.pyplot as plt
from emoji import emojis

def load_emojis(emojis):
    data = np.array(emojis, dtype=np.float32)
    data = data / 1.0  # Asegura que los valores están en el rango [0, 1]
    return data

def visualize_latent_transition(vae, X, index1, index2, num_steps=10):
    """
    Genera una gráfica con emojis interpolados entre todos los pares.
    """
    # Codificar los datos originales al espacio latente
    mu, log_var, z, _, _, _, _, _ = vae.forward(X)
    
    # Seleccionar los puntos latentes
    z1 = z[index1]
    z2 = z[index2]
    
    # Generar puntos intermedios en el espacio latente
    latent_points = [z1 + (t / (num_steps - 1)) * (z2 - z1) for t in range(num_steps)]
    
    # Decodificar los puntos latentes
    reconstructed_emojis = []
    for point in latent_points:
        _, decoder_Z_values = vae.decoder_forward(point)
        last_layer_key = max(decoder_Z_values.keys(), key=lambda x: int(x[1:]))
        reconstructed = decoder_Z_values[last_layer_key]
        reconstructed_emojis.append(np.clip(reconstructed, 0, 1))
    
    # Visualizar la transición
    fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2))
    for i, emoji in enumerate(reconstructed_emojis):
        axes[i].imshow(emoji.reshape(25, 22), cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()




def compare_original_vs_reconstruction(X, reconstructed_data, num_chars=5):
    fig, axes = plt.subplots(2, num_chars, figsize=(2 * num_chars, 5))

    for i in range(num_chars):
        # Original
        ax = axes[0, i]
        ax.imshow(X[i].reshape(25, 22), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Original {i+1}")

        # Reconstruido
        ax = axes[1, i]
        ax.imshow(reconstructed_data[i].reshape(25, 22), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Reconstruido {i+1}")

    plt.tight_layout()
    plt.show()

def plot_latent_space(latent_space):
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_space[:, 0], latent_space[:, 1])
    plt.title("Espacio Latente")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True)
    plt.show()

def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Cargar datos
    X = load_emojis(emojis)

    # Instanciar el modelo
    vae = VAE(
        input_dim=config["input_dim"],
        encoder_layers=config["encoder_layers"],
        latent_dim=config["latent_dim"],
        decoder_layers=config["decoder_layers"],
        activation_fn_name=config["activation_fn"],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        initial_lr=config.get("initial_lr", None),
        decay_rate=config.get("decay_rate", None),
        variable_lr=config.get("variable_lr", False)
    )

    # Entrenar el modelo
    vae.train(X, epochs=config["epochs"])

    # Reconstruir datos
    mu, log_var, z, reconstructed_X, _, _, _, _ = vae.forward(X)

    # Visualizar reconstrucciones y espacio latente
    compare_original_vs_reconstruction(X, reconstructed_X)
    plot_latent_space(z)

    visualize_latent_transition(vae, X, index1=4, index2=5, num_steps=5)

if __name__ == "__main__":
    main()
