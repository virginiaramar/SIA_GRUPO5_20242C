# TP SIA - GRUPO 5 

# Integrantes
 Madero Torres, Eduardo Federico - 59494
 Ramos Marca, Mar´ıa Virginia - 67200
 Pluss, Ramiro - 66254
 Kuchukhidze, Giorgi - 67262

 # Implementación de un Autoencoder Básico - Ejercicio 1B

## Descripción General
Este proyecto implementa un **autoencoder** desde cero utilizando Python. El objetivo principal es representar caracteres binarios en un espacio latente de dos dimensiones, reconstruyendo los datos originales con un error máximo de 1 píxel y explorando capacidades generativas. También se incluyen herramientas para visualizar el espacio latente, evaluar la reconstrucción y generar nuevas letras.

---

## Archivos Principales

### `main.py`
- Archivo principal que ejecuta el flujo del proyecto:
  - Carga los datos.
  - Inicializa y entrena el autoencoder.
  - Guarda los pesos del modelo.
  - Reconstruye datos originales y los compara con las reconstrucciones.
  - Genera gráficos del espacio latente y evalúa errores de reconstrucción.

### `autoencoder.py`
- Define la clase `Autoencoder`, que implementa:
  - Inicialización de pesos (Xavier, He, Uniforme).
  - Propagación hacia adelante y retropropagación.
  - Actualización de parámetros mediante Adam o Descenso de Gradiente.
  - Métodos para reconstrucción, generación de datos y manejo del espacio latente.

### `config.json`
- Archivo de configuración para personalizar los hiperparámetros del modelo:
  - Arquitectura de la red.
  - Función de activación (e.g., ReLU, Sigmoid).
  - Función de pérdida (e.g., Cross Entropy, MSE).
  - Método de optimización (Adam o GD).
  - Learning Rate y Decay Rate.
  - Semilla para reproducibilidad.

### `reconstruccion.py`
- Reconstruye imágenes (caracteres) desde puntos específicos del espacio latente.
- Incluye métodos para visualizar reconstrucciones específicas.

### `utils.py`
- Contiene funciones auxiliares, como:
  - Funciones de activación (ReLU, Sigmoid, Tanh) y sus derivadas.
  - Funciones de pérdida (MSE, Cross Entropy) y sus derivadas.

---

## Requisitos del Sistema
- **Python** 3.8+
- Bibliotecas necesarias:
  - `numpy`
  - `matplotlib`
  - `pandas` (opcional para algunas funciones)

---

## Configuración
Edita el archivo `config.json` para personalizar la configuración del modelo. Ejemplo:
```json
{
    "file_path": "input/font_vectors.txt",
    "layers": [35, 25, 10, 2, 10, 25, 35],
    "initial_lr": 0.01,
    "decay_rate": 0.9995,
    "learning_rate": 0.01,
    "epochs": 5000,
    "activation_fn": "relu",
    "variable_lr": false,
    "loss_function": "cross_entropy_loss",
    "optimizer": "adam",
    "n_runs": 1,
    "seed": 42,
    "init_method": "uniform"
}
```
---

# Implementación de un Denoising Autoencoder - Ejercicio 1B

## Descripción General
Este proyecto implementa un **denoising autoencoder** siguiendo el autoencoder anterior. Además, se analiza la calidad de las reconstrucciones, la distribución del espacio latente y la sensibilidad del modelo a diferentes tipos de ruido.

---

## Archivos Principales

### `main_noise.py`
- Archivo principal que ejecuta el flujo del proyecto:
  - Carga y aplica ruido a los datos.
  - Inicializa y entrena el denoising autoencoder.
  - Guarda los pesos del modelo entrenado.
  - Reconstruye los datos originales a partir de los datos con ruido.
  - Genera gráficos del espacio latente y evalúa errores de reconstrucción.

### `autoencoder_noise.py`
- Define la clase `Autoencoder`, que implementa:
  - Inicialización de pesos (Xavier, He, Uniforme).
  - Propagación hacia adelante y retropropagación.
  - Actualización de parámetros mediante Adam o Descenso de Gradiente.
  - Métodos específicos para la reconstrucción y generación de datos desde el espacio latente.
  - Funciones de entrenamiento con visualización de curvas de pérdida.

### `config_noise.json`
- Archivo de configuración para personalizar los hiperparámetros del modelo:
  - Arquitectura de la red.
  - Función de activación (e.g., ReLU, Sigmoid).
  - Función de pérdida (e.g., Cross Entropy, MSE).
  - Método de optimización (Adam o GD).
  - Parámetros de ruido, como tipo (`salt_and_pepper`, gaussiano, masking) y nivel.
  - Learning Rate y Decay Rate.
  - Semilla para reproducibilidad.


---

## Configuración
Edita el archivo `config_noise.json` para personalizar la configuración del modelo. Ejemplo:
```json
{
    "file_path": "input/font_vectors.txt",
    "layers": [35, 25, 15, 7, 2, 7, 15, 25, 35],
    "initial_lr": 0.01,
    "decay_rate": 0.999,
    "learning_rate": 0.01,
    "epochs": 9000,
    "activation_fn": "relu",
    "variable_lr": false,
    "loss_function": "cross_entropy_loss",
    "optimizer": "adam",
    "noise_type": "salt_and_pepper",
    "noise_level": 0.9,
    "seed": 43
}
```
