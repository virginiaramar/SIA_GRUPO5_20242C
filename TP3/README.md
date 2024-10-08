## TP SIA - GRUPO 5 

## Integrantes
 Madero Torres, Eduardo Federico - 59494
 Ramos Marca, Mar´ıa Virginia - 67200
 Pluss, Ramiro - 66254
 Kuchukhidze, Giorgi - 67262

## Descripción del Trabajo

**Trabajo Práctico Número 3 - Sistemas de Inteligencia Artificial - ITBA**

Este proyecto implementa dos perceptrones, simple y multicapa. A su vez, el simple también pueede ser lineal y no lineal. Con ellos vamos a estar aprendiendo y validnado diferentes conjuntos de datos.

## Ejecución de los ejercicios

### Ejercicio 1



### Ejercicio 2



### Ejercicio 3

En este ejercicio se implementa un perceptrón multicapa en el cual se utiliza un config para poder meterle todos los datos de entrada.

**Estructura del archivo config.json**

```json
{
    "data": {
        "input": "<string: 'data/digits_flatten.txt'>",
        "output": "<string: 'data/TP3-ej3b-realoutput.txt', 'data/TP3-ej3c-realoutput.txt'>",
        "problem_type": "<string: 'multiclass', 'binary'>",
    },
    "initial_parameters": {
        "architecture": [35, 64, 32, 10],
        "learning_rate": "<float:>",
        "epochs": "<int>",
        "mode": "<string: 'batch', 'mini-batch', 'online'>",
        "minibatch_size": "<int>"
    },
    "weights": {
        "initialization": "<string: 'zero', 'random', 'normal', 'xavier', 'he'>"
    },
    "activation_function": {
        "function": "<string: 'sigmoid', 'tanh', 'relu'>",
        "output_function": "<string: 'sigmoid', 'softmax'>",
        "beta": "<float:>"
    },
    "error": {
        "threshold": "<float:>"
    },
    "optimizer": {
        "type": "<string: 'gradient_descent', 'adam', 'momentum'>",
        "adaptive_learning_rate": "<bool: true - false",
        "lr_adjustment_value": "<float:>",
        "adam": {
            "beta1": "<float:>",
            "beta2": "<float:>",
            "epsilon": "<float:>"
            "alpha": "<float:>"
        }
    },
    "cross_validation": {
        "use_cross_validation": "<bool: true - false",
        "k_folds": "<int>",
        "shuffle": "<bool: true - false",
        "random_seed": "<int>"
    }
}
```

**Opciones de Configuración**

### Datos de Entrada y Salida

- `input`: Archivo de entrada que contiene los datos a procesar.
  - Ejemplo: `"data/digits_flatten.txt"`
- `output`: Archivos de salida donde se guardarán los resultados.
  - Ejemplo: `["data/TP3-ej3b-realoutput.txt", "data/TP3-ej3c-realoutput.txt"]`
- `problem_type`: Tipo de problema que se está resolviendo.
  - Opciones: `"multiclass"`, `"binary"`

### Parámetros Iniciales

- `architecture`: Lista de enteros que define el número de neuronas en cada capa de la red.
  - Ejemplo: `[35, 64, 32, 10]`
- `learning_rate`: Tasa de aprendizaje, que controla la magnitud de la actualización de los pesos en cada iteración.
  - Ejemplo: `0.01`
- `epochs`: Número de épocas de entrenamiento.
  - Ejemplo: `100`
- `mode`: Modo de entrenamiento.
  - Opciones: `"batch"`, `"mini-batch"`, `"online"`
- `minibatch_size`: Tamaño del mini-batch (si se usa mini-batch).
  - Ejemplo: `32`

### Inicialización de Pesos

- `initialization`: Método para inicializar los pesos de la red.
  - Opciones: `"zero"`, `"random"`, `"normal"`, `"xavier"`, `"he"`

### Función de Activación

- `function`: Función de activación utilizada en las capas intermedias de la red.
  - Opciones: `"sigmoid"`, `"tanh"`, `"relu"`
- `output_function`: Función de activación para la capa de salida.
  - Opciones: `"sigmoid"`, `"softmax"`
- `beta`: Parámetro de ajuste (beta) para las funciones de activación, en caso de ser necesario.
  - Ejemplo: `1.0`

### Umbral de Error

- `threshold`: Valor del umbral de error para el criterio de parada.
  - Ejemplo: `0.001`

### Optimizador

- `type`: Tipo de optimizador a utilizar.
  - Opciones: `"gradient_descent"`, `"adam"`, `"momentum"`
- `adaptive_learning_rate`: Indica si se ajustará dinámicamente la tasa de aprendizaje.
  - Opciones: `true`, `false`
- `lr_adjustment_value`: Valor de ajuste para la tasa de aprendizaje en caso de usar un learning rate adaptativo.
  - Ejemplo: `0.001`
- **Parámetros del optimizador Adam**:
  - `beta1`: Parámetro beta1 para el optimizador Adam.
    - Ejemplo: `0.9`
  - `beta2`: Parámetro beta2 para el optimizador Adam.
    - Ejemplo: `0.999`
  - `epsilon`: Valor de epsilon para la estabilidad numérica.
    - Ejemplo: `1e-08`
  - `alpha`: Tasa de aprendizaje específica para Adam.
    - Ejemplo: `0.001`

### Validación Cruzada

- `use_cross_validation`: Indica si se utilizará validación cruzada.
  - Opciones: `true`, `false`
- `k_folds`: Número de particiones (folds) a utilizar en la validación cruzada.
  - Ejemplo: `5`
- `shuffle`: Indica si los datos deben mezclarse antes de realizar la validación cruzada.
  - Opciones: `true`, `false`
- `random_seed`: Semilla aleatoria para garantizar la reproducibilidad del shuffle.
  - Ejemplo: `42`

**Ejecución del programa**

Para ejecutar el programa:

1. Asegúrate de tener Python instalado.
2. Navega hasta el directorio del proyecto en la terminal.
3. Ejecuta el siguiente comando:

```
python main.py
```

**Selección del apartado**

Después de ejecutar el programa, se te presentará una opción en la terminal para seleccionar cuál de los ejercicios deseas ejecutar. A continuación, ingresa el número correspondiente a la tarea que quieres realizar:

1. **Ejercicio 1**: Aprendizaje de la función lógica XOR.
2. **Ejercicio 2**: Discriminación de paridad (determinar si un número es par o impar).
3. **Ejercicio 3**: Discriminación de dígitos (identificación de dígitos a partir de su representación).

Para seleccionar un ejercicio, ingresa el número correspondiente y presiona `Enter`.
