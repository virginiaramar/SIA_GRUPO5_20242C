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

