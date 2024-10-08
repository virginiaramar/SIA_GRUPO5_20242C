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

# Simple Perceptron Implementation

## Descripción

Este proyecto implementa un perceptrón simple utilizando las librerías NumPy y Matplotlib. El perceptrón es un modelo básico de aprendizaje automático que puede clasificar datos linealmente separables. Esta implementación permite entrenar el perceptrón en problemas lógicos como el **AND** y **XOR**, visualizando cómo la frontera de decisión cambia durante el entrenamiento.

## Estructura del Código

El código está compuesto por las siguientes partes principales:

### 1. **Clase `Perceptron`**
   - **Inicialización (`__init__`)**: Inicializa los pesos aleatoriamente y recibe un parámetro `alpha` que define la tasa de aprendizaje y `epsilon` umbral pequeño para controlar la actualización del peso
   - **Método `step_activation(x)`**: Implementa la función de activación escalón que devuelve 1 si la entrada es mayor o igual a 0, o -1 en caso contrario.
   - **Método `fit_with_plot(X, y, n_epochs, plot_interval, title)`**: Entrena el perceptrón utilizando un conjunto de datos `X` y sus etiquetas `y`. Además, genera un gráfico de la frontera de decisión en intervalos definidos por `plot_interval`.
   - **Método `predict(X)`**: Realiza predicciones para nuevas muestras `X`.
   - **Método `plot_decision_boundary(X, y, epoch, title)`**: Visualiza los datos y la frontera de decisión aprendida por el perceptrón.

### 2. **Main (Entrenamiento y Visualización)**
   - El código principal crea conjuntos de datos para los problemas lógicos **AND** y **XOR**. 
   - Se entrena un perceptrón para cada problema, visualizando la frontera de decisión en cada época.

## Requisitos

Este código requiere las siguientes bibliotecas de Python:
- **NumPy**: Para operaciones matriciales y manejo de datos.
- **Matplotlib**: Para la visualización de los datos y las fronteras de decisión.


## Ejecución

Para ejecutar el código:
1. Asegúrate de tener un entorno de Python configurado.
2. Corre el script principal `simple_perceptron.py`


Durante la ejecución, el perceptrón se entrenará y se visualizará la frontera de decisión en intervalos regulares de épocas para los problemas **AND** y **XOR**.

## Resultados Esperados

- Para el problema **AND**, el perceptrón debería converger y aprender la frontera de decisión adecuada.
- Para el problema **XOR**, como los datos no son linealmente separables, el perceptrón no logrará encontrar una solución adecuada.

## Personalización

Puedes personalizar el número de épocas, la tasa de aprendizaje (`alpha`), y la frecuencia de visualización modificando los parámetros en las llamadas al método `fit_with_plot`.



### Ejercicio 2

# Perceptron Linear & Non-Linear Classification

## Descripción

Este proyecto implementa y compara la capacidad de clasificación de un perceptrón lineal y un perceptrón no lineal en un conjunto de datos proporcionado en el archivo **TP3-ej2-conjunto.csv**. Los modelos se evalúan utilizando validación cruzada para medir su precisión en la clasificación. Además, se incluyen simulaciones para evaluar métricas adicionales como el error cuadrático medio (MSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²). La comparación se realiza utilizando un perceptrón simple con función de activación escalón y otro con función de activación tangente hiperbólica (tanh).

## Archivos Principales

### 1. **`linear_perceptron.py`**:
   - Implementa un perceptrón lineal clásico.
   - **Clase `Perceptron`**: Se inicializan los pesos aleatoriamente, y la función de activación escalón se utiliza para clasificar los datos.
   - **Método `fit(X, y, epochs)`**: Entrena el modelo ajustando los pesos en función del error en cada época.
   - **Método `predict(X)`**: Realiza predicciones basadas en los pesos aprendidos.

### 2. **`nonlinear_perceptron.py`**:
   - Implementa un perceptrón no lineal utilizando la función de activación tangente hiperbólica (`tanh`).
   - **Clase `NonLinearPerceptron`**: Inicializa los pesos aleatoriamente y utiliza la función `tanh` como activación.
   - **Método `fit(X, y, n_epochs)`**: Entrena el modelo ajustando los pesos utilizando la derivada de `tanh` para calcular los gradientes.
   - **Método `predict(X)`**: Predice etiquetas ajustando el valor de salida de la tangente hiperbólica a -1 o 1, dependiendo de si la salida es mayor o menor a 0.

### 3. **`training_and_eval.py`**:
   - Carga el conjunto de datos del archivo **TP3-ej2-conjunto.csv**.
   - Aplica validación cruzada de 5 particiones para ambos modelos (lineal y no lineal) y calcula la precisión de cada uno.
   - **Función `cross_validation(X, y, model_class, alpha, n_splits, n_epochs)`**: Implementa la validación cruzada para dividir el conjunto de datos, entrenar el modelo y evaluar su precisión.

### 4. **`perceptron_comparison.py`**:
   - Implementa simulaciones para comparar los modelos lineales y no lineales en términos de MSE, MAE, y R².
   - **Función `run_simulations(X, y, n_simulations, epochs)`**: Corre varias simulaciones para evaluar la precisión de ambos modelos.
   - **Función `plot_results(...)`**: Genera gráficos comparativos de MSE, MAE y R² entre el perceptrón lineal y no lineal.

### 4. **`fitting.py`**:
   - Este código tiene como objetivo comparar las diferentes métricas utilizando un tamaño de prueba del 20%.
      - **Función `load_data(filename)`**: Carga y normaliza los datos de entrada, ajustando las etiquetas a un rango de [-1, 1].
      - **Función `train_and_evaluate(X, y, model_class, alpha=0.01, n_epochs=100, test_size=0.5)`**: Entrena y evalúa un modelo especificado sobre los datos, calculando MSE, MAE y R² para los conjuntos de entrenamiento y prueba.


### 5. **`TP3-ej2-conjunto.csv`**:
   - Contiene el conjunto de datos utilizado para entrenar y evaluar los modelos de perceptrón.
   - Los datos están formateados con tres características (features) y una etiqueta de clase como salida.

## Requisitos

Este proyecto depende de las siguientes bibliotecas:
- **NumPy**: Para el manejo de matrices y operaciones numéricas.
- **CSV**: Para la lectura y procesamiento del conjunto de datos.
- **Matplotlib**: Para la visualización de los resultados.

## Ejecución

1. Asegúrate de tener todos los archivos en el mismo directorio.
2. Corre el script principal `training_and_eval.py` para entrenar ambos modelos y obtener sus precisiones.
3. Corre el script principal `perceptron_comparison.py` para comparar los modelos lineal y no lineal.
4. Corre el script principal `fitting.py` para comparar las diferentes métricas utilizando un tamaño de prueba del 20%.



### Ejercicio 3

# Multilayer Perceptron Implementation

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
