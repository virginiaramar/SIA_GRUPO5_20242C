# TP SIA - GRUPO 5 

# Integrantes
 Madero Torres, Eduardo Federico - 59494;
 Ramos Marca, María Virginia - 67200;
 Pluss, Ramiro - 66254;
 Kuchukhidze, Giorgi - 67262;

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
# Variational Autoencoder (VAE) para Emojis

Este proyecto implementa un Autoencoder Variacional (VAE) para generar y reconstruir imágenes de emojis. El VAE es capaz de aprender una representación comprimida (espacio latente) de las imágenes y generar nuevas muestras similares a los datos de entrenamiento.

## Estructura del Proyecto
```
vae_project/
├── config.json          # Archivo de configuración
├── input/               
│   └── emojis/          # Directorio con imágenes de entrada
├── src/
│   ├── model/           # Implementación del VAE
│   └── utils/           # Utilidades y visualización
├── output/
│   ├── images/          # Imágenes generadas
│   └── models/          # Modelos guardados
└── main.py              # Punto de entrada
```

## Configuración (config.json)

### 1. Parámetros de Datos
```json
"data": {
    "input_size": [64, 64],    // Tamaño de las imágenes
    "channels": 3,             // Canales de color (RGB)
    "batch_size": 2,           // Tamaño del batch
    "validation_split": 0.2    // Proporción de datos para validación
}
```

### 2. Arquitectura del Modelo
```json
"model": {
    "latent_dim": 16,          // Dimensión del espacio latente
    "encoder_layers": [         // Capas del encoder
        {"units": 512, "activation": "relu"},
        {"units": 256, "activation": "relu"},
        {"units": 128, "activation": "relu"}
    ],
    "decoder_layers": [         // Capas del decoder
        {"units": 128, "activation": "relu"},
        {"units": 256, "activation": "relu"},
        {"units": 512, "activation": "cosh"}
    ]
}
```

### 3. Parámetros de Entrenamiento
```json
"training": {
    "epochs": 1000,            // Número máximo de épocas
    "learning_rate": 0.001,    // Tasa de aprendizaje
    "optimizer": {
        "type": "adam",        // Tipo de optimizador
        "beta1": 0.9,          // Parámetro β₁ de Adam
        "beta2": 0.999,        // Parámetro β₂ de Adam
        "epsilon": 1e-8        // Epsilon para estabilidad numérica
    }
}
```

## Funcionamiento del VAE

### 1. Encoder
- **Entrada**: Imagen RGB (64×64×3 = 12,288 dimensiones)
- **Proceso**: 
  1. Comprime la imagen a través de capas densas
  2. Genera dos vectores: media (μ) y log-varianza (log σ²)
  3. Usa el "reparametrization trick": z = μ + σ * ε
- **Salida**: Vector en el espacio latente (ej: 16 dimensiones)

### 2. Decoder
- **Entrada**: Vector del espacio latente
- **Proceso**: 
  1. Expande el vector a través de capas densas
  2. Reconstruye la imagen original
- **Salida**: Imagen reconstruida (64×64×3)

### 3. Función de Pérdida (Loss)
La pérdida total tiene dos componentes:
1. **Pérdida de Reconstrucción**: Mide qué tan bien se reconstruye la imagen
   ```
   L_recon = -Σ(x * log(x̂) + (1-x) * log(1-x̂))
   ```
2. **Pérdida KL**: Regulariza el espacio latente
   ```
   L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
   ```

## Visualizaciones Generadas

### 1. Reconstrucción (reconstruccion.png)
- **Qué muestra**: Imagen original vs. imagen reconstruida
- **Interpretación**: Evalúa la calidad de la reconstrucción
- **Ubicación**: output/images/reconstruccion.png

### 2. Espacio Latente (latent_space.png)
- **Qué muestra**: Distribución de las imágenes en el espacio latente
- **Interpretación**: 
  - Cercanía indica similitud entre imágenes
  - Clusters sugieren agrupaciones naturales
- **Ubicación**: output/images/latent_space.png

### 3. Gráfico de Loss (loss.png)
- **Qué muestra**: Evolución de la pérdida durante el entrenamiento
- **Interpretación**:
  - Debe decrecer con el tiempo
  - Estabilización indica convergencia
  - Fluctuaciones grandes sugieren problemas
- **Ubicación**: output/images/loss.png

### 4. Traversal del Espacio Latente (latent_traversal.png)
- **Descripción**: Muestra variaciones en el espacio latente
- **Componentes**:
  - Grid de imágenes generadas
  - Ejes X/Y: Primeras dos dimensiones latentes
  - Valores: Coordenadas en el espacio latente
- **Interpretación**:
  - Muestra transiciones suaves entre características
  - Indica qué aprende cada dimensión latente

  
## Ejecución

1. Preparar datos:
   ```bash
   # Colocar imágenes en input/emojis/
   ```

2. Configurar parámetros:
   ```bash
   # Editar config.json según necesidades
   ```

3. Ejecutar:
   ```bash
   python main.py
   ```

## Consejos de Configuración

1. **Espacio Latente**:
   - Dimensión pequeña (8-16): Más compresión, menos detalles
   - Dimensión grande (32-64): Más detalles, menos generalización

2. **Arquitectura**:
   - Más capas: Mejor capacidad de aprendizaje, más tiempo
   - Menos capas: Entrenamiento más rápido, menos expresividad

3. **Entrenamiento**:
   - Learning rate alto (0.001): Aprendizaje rápido, posible inestabilidad
   - Learning rate bajo (0.0001): Más estable, convergencia más lenta

## Ejemplos de Configuraciones

### 1. Configuración Básica
```json
{
    "model": {
        "latent_dim": 8,
        "encoder_layers": [
            {"units": 128, "activation": "relu"},
            {"units": 64, "activation": "relu"}
        ]
    },
    "training": {
        "epochs": 300,
        "learning_rate": 0.001
    }
}
```

### 2. Configuración Profunda
```json
{
    "model": {
        "latent_dim": 16,
        "encoder_layers": [
            {"units": 512, "activation": "relu"},
            {"units": 256, "activation": "relu"},
            {"units": 128, "activation": "relu"}
        ]
    },
    "training": {
        "epochs": 500,
        "learning_rate": 0.0001
    }
}
```

### 3. Configuración para Detalles Finos
```json
{
    "model": {
        "latent_dim": 32,
        "encoder_layers": [
            {"units": 1024, "activation": "relu"},
            {"units": 512, "activation": "relu"}
        ]
    },
    "training": {
        "epochs": 1000,
        "learning_rate": 0.00001
    }
}
```

## Solución de Problemas

1. **Loss no decrece**:
   - Reducir learning rate
   - Aumentar tamaño de la red
   - Verificar preprocesamiento de datos

2. **Reconstrucciones borrosas**:
   - Aumentar dimensión latente
   - Profundizar la red
   - Ajustar peso de pérdida KL

3. **Entrenamiento inestable**:
   - Reducir learning rate
   - Ajustar parámetros de Adam
   - Verificar inicialización de pesos

