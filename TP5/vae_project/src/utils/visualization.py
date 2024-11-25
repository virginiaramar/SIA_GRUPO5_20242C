import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import os

def plot_reconstruction(original, reconstructed, save_dir="output/images"):
    """Visualiza y guarda la comparación de imágenes originales y reconstruidas"""
    fig = plt.figure(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    
    # Original
    plt.subplot(1, 2, 1)
    img = original[0].reshape(64, 64, 3)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    # Reconstruida
    plt.subplot(1, 2, 2)
    img = reconstructed[0].reshape(64, 64, 3)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title('Reconstruida')
    plt.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'reconstruccion.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Reconstrucción guardada en:", save_path)

def plot_loss(history, save_dir="output/images"):
    """Visualiza y guarda el historial de pérdida"""
    plt.figure(figsize=(10, 6))
    
    # Filtrar valores infinitos y NaN
    valid_history = np.array(history)[np.isfinite(history)]
    epochs = range(1, len(valid_history) + 1)
    
    # Graficar pérdida
    plt.plot(epochs, valid_history, 'b-', linewidth=2, label='Pérdida Total')
    
    # Personalización
    plt.title('Evolución de la Pérdida Durante el Entrenamiento', fontsize=14, pad=20)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Anotar mínimo
    min_loss = np.min(valid_history)
    min_epoch = np.argmin(valid_history) + 1
    plt.annotate(f'Mínimo: {min_loss:.2f}',
                xy=(min_epoch, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->'))
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Gráfico de pérdida guardado en:", save_path)

def plot_latent_space(vae, data, filenames, save_dir="output/images"):
    """Visualiza el espacio latente con nombres de archivo"""
    # Configurar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Obtener representaciones latentes
    latent_representations = []
    for img in data:
        z = vae.encode(img.reshape(1, -1))
        latent_representations.append(z[0])
    
    latent_representations = np.array(latent_representations)
    
    # PCA si es necesario
    if latent_representations.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_representations)
        explained_variance = pca.explained_variance_ratio_
        print(f"Varianza explicada: {explained_variance * 100}%")
    else:
        latent_2d = latent_representations
    
    x = latent_2d[:, 0]
    y = latent_2d[:, 1]
    
    # Plotear puntos
    scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', 
                        s=100, alpha=0.6)
    
    # Añadir nombres
    for i, filename in enumerate(filenames):
        name = os.path.splitext(filename)[0]
        ax.annotate(name,
                   (x[i], y[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   color='white',
                   fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7))
    
    # Personalización
    ax.set_title('Distribución en el Espacio Latente',
                fontsize=14, pad=20, color='white')
    ax.set_xlabel('Componente Principal 1', fontsize=12, color='white')
    ax.set_ylabel('Componente Principal 2', fontsize=12, color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.set_aspect('equal')
    
    # Colorbar
    plt.colorbar(scatter, label='Índice de imagen')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'latent_space.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300,
                facecolor='black', edgecolor='none')
    plt.close()
    print("Espacio latente guardado en:", save_path)

def show_latent_traversal(vae, save_dir="output/images", n_steps=8):
    """Muestra cómo cambian las imágenes al atravesar el espacio latente"""
    # Crear grid para las dos primeras dimensiones
    x = np.linspace(-2, 2, n_steps)
    y = np.linspace(-2, 2, n_steps)
    
    fig, axes = plt.subplots(n_steps, n_steps, figsize=(15, 15))
    fig.suptitle('Traversal del Espacio Latente\n(Primeras 2 dimensiones)', fontsize=16)
    
    # Generar imágenes
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            # Crear vector latente completo (todas las dimensiones)
            z = np.zeros((1, vae.latent_dim))
            z[0, 0] = xi  # Primera dimensión
            z[0, 1] = yi  # Segunda dimensión
            # Las otras dimensiones quedan en 0
            
            try:
                img = vae.decode(z)
                img = img.reshape(64, 64, 3)
                img = np.clip(img, 0, 1)
            except Exception as e:
                print(f"Error en posición ({i}, {j}): {str(e)}")
                img = np.zeros((64, 64, 3))  # Imagen negra en caso de error
            
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            # Añadir valores de coordenadas
            if i == 0:
                axes[i, j].set_title(f'y: {yi:.1f}')
            if j == 0:
                axes[i, 0].set_ylabel(f'x: {xi:.1f}')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'latent_traversal.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Traversal del espacio latente guardado en:", save_path)

def visualize_all(vae, data, history, filenames, save_dir="output/images"):
    """Genera todas las visualizaciones"""
    try:
        print("\nGenerando visualizaciones...")
        
        # Reconstrucción
        print("1. Generando reconstrucción...")
        original_sample = data[:1]
        reconstructed_sample = vae.forward(original_sample)
        plot_reconstruction(original_sample, reconstructed_sample, save_dir)
        
        # Loss
        print("2. Generando gráfico de pérdida...")
        plot_loss(history, save_dir)
        
        # Espacio latente
        print("3. Generando visualización del espacio latente...")
        plot_latent_space(vae, data, filenames, save_dir)
        
        # Traversal
        print("4. Generando traversal del espacio latente...")
        show_latent_traversal(vae, save_dir)
        
        print("\nTodas las visualizaciones han sido guardadas en:", save_dir)
        
    except Exception as e:
        print(f"\nError generando visualización: {str(e)}")
        import traceback
        traceback.print_exc()