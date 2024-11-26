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

