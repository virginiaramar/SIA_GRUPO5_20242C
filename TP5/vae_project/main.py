import json
import os
import sys
import numpy as np
from src.model.vae import VAE
from src.utils.data import load_images_with_filenames
from src.utils.visualization import plot_reconstruction, plot_loss, plot_latent_space, show_latent_traversal

# Configuración por defecto
DEFAULT_CONFIG = {
    "data": {
        "input_size": [64, 64],
        "channels": 3,
        "batch_size": 2,
        "validation_split": 0.2
    },
    "training": {
        "epochs": 500,
        "learning_rate": 0.001,
        "batch_size": 2,
        "optimizer": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        }
    }
}

def load_config(config_path='config.json'):
    """Carga y valida la configuración"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Asegurar que existan todos los campos necesarios
        if 'training' not in config:
            config['training'] = DEFAULT_CONFIG['training']
        if 'batch_size' not in config['training']:
            config['training']['batch_size'] = DEFAULT_CONFIG['training']['batch_size']
        
        return config
    except Exception as e:
        print(f"Error cargando configuración: {str(e)}")
        raise

def create_directories(config):
    """Crea los directorios necesarios"""
    directories = [
        config['paths']['save_dir'],
        config['paths']['log_dir'],
        'output/images'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directorio creado/verificado: {directory}")

def print_training_info(config, data_shape):
    """Imprime información sobre el entrenamiento"""
    print("\nConfiguración del entrenamiento:")
    print(f"- Tamaño de datos: {data_shape}")
    print(f"- Batch size: {config['training']['batch_size']}")
    print(f"- Learning rate: {config['training']['learning_rate']}")
    print(f"- Épocas: {config['training']['epochs']}")
    print(f"- Dimensión latente: {config['model']['latent_dim']}")
    
    print("\nArquitectura del encoder:")
    for i, layer in enumerate(config['model']['encoder_layers']):
        print(f"- Capa {i+1}: {layer['units']} unidades, {layer['activation']}")
    
    print("\nArquitectura del decoder:")
    for i, layer in enumerate(config['model']['decoder_layers']):
        print(f"- Capa {i+1}: {layer['units']} unidades, {layer['activation']}")

def visualize_all(vae, data, history, filenames, save_dir="output/images"):
    """Genera y guarda todas las visualizaciones"""
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
    
    # Traversal del espacio latente
    print("4. Generando traversal del espacio latente...")
    show_latent_traversal(vae, save_dir)
    
    print("\nTodas las visualizaciones han sido guardadas en:", save_dir)

def main(config_path):
    # Cargar configuración
    config = load_config(config_path)
    print(f"\nUsando configuración: {config_path}")
    
    # Crear directorios necesarios
    create_directories(config)
    
    # Cargar datos
    print("\nCargando datos...")
    data, filenames = load_images_with_filenames(
        config['paths']['data_dir'],
        target_size=tuple(config['data']['input_size'])
    )
    print(f"Datos cargados: {len(data)} imágenes")
    
    # Imprimir información del entrenamiento
    print_training_info(config, data.shape)
    
    # Crear modelo
    print("\nCreando modelo...")
    input_dim = np.prod(config['data']['input_size']) * config['data']['channels']
    vae = VAE(
        input_dim=input_dim,
        latent_dim=config['model']['latent_dim'],
        encoder_layers=config['model']['encoder_layers'],
        decoder_layers=config['model']['decoder_layers']
    )
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    history = vae.train(
        data,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Generar visualizaciones
    visualize_all(vae, data, history, filenames)
    
    print("\n¡Proceso completado exitosamente!")
    print("Revisa las visualizaciones en el directorio 'output/images/'")

def handle_error(e):
    """Maneja y reporta errores de manera amigable"""
    error_types = {
        KeyError: "Error en la configuración: falta un campo requerido",
        FileNotFoundError: "Error: no se encontró un archivo necesario",
        ValueError: "Error: valor inválido en la configuración",
        Exception: "Error inesperado"
    }
    
    error_type = type(e)
    error_message = error_types.get(error_type, error_types[Exception])
    
    print(f"\n{'='*50}")
    print(f"ERROR: {error_message}")
    print(f"Detalles: {str(e)}")
    print(f"{'='*50}\n")
    
    import traceback
    print("Traceback completo:")
    traceback.print_exc()

if __name__ == "__main__":
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_path)
    except Exception as e:
        handle_error(e)
        sys.exit(1)