import numpy as np
from PIL import Image
import os

def load_images_with_filenames(data_dir, target_size=(64, 64)):
    """
    Carga imágenes y devuelve tanto los datos como los nombres de archivo
    """
    images = []
    filenames = []
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    
    # Asegurarse de que el directorio existe
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio {data_dir} no existe")
    
    # Listar y ordenar archivos para consistencia
    files = sorted(os.listdir(data_dir))
    
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(data_dir, filename)
            try:
                # Cargar y preprocesar imagen
                img = Image.open(img_path)
                img = img.convert('RGB')  # Asegurar que es RGB
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convertir a array y normalizar
                img_array = np.array(img) / 255.0
                
                # Asegurar rango [0, 1]
                img_array = np.clip(img_array, 0, 1)
                
                # Aplanar la imagen
                img_flat = img_array.reshape(-1)
                
                # Guardar imagen y nombre
                images.append(img_flat)
                filenames.append(filename)
                
                print(f"Imagen {filename} cargada. Shape: {img_array.shape}, "
                      f"Rango: [{img_flat.min():.3f}, {img_flat.max():.3f}]")
                
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
    
    if not images:
        raise ValueError("No se encontraron imágenes válidas en el directorio")
    
    return np.array(images), filenames

def split_data(data, validation_split=0.2):
    """
    Divide los datos en conjuntos de entrenamiento y validación
    """
    if len(data) == 0:
        raise ValueError("No hay datos para dividir")
        
    n_validation = max(1, int(len(data) * validation_split))
    indices = np.random.permutation(len(data))
    
    validation_indices = indices[:n_validation]
    training_indices = indices[n_validation:]
    
    validation_data = data[validation_indices]
    training_data = data[training_indices]
    
    return training_data, validation_data

def prepare_batches(data, batch_size):
    """
    Prepara los datos en batches para el entrenamiento
    """
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    for start_idx in range(0, len(data), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        batch = data[excerpt]
        if len(batch) < batch_size:
            # Rellenar el último batch si es necesario
            pad_size = batch_size - len(batch)
            batch = np.pad(batch, ((0, pad_size), (0, 0)), mode='edge')
        yield batch