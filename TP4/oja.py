import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pca import perform_pca_analysis
from sklearn.decomposition import PCA

def parse_csv(file_path):
    parsed_file = pd.read_csv(file_path)

    # Extraer la lista de países
    countries_list = parsed_file.loc[:, "Country"].values

    # Extraer las propiedades numéricas
    properties = parsed_file.iloc[:, 1:8].values

    # Estandarizar las propiedades
    properties_mean = np.mean(properties, axis=0)
    properties_std = np.std(properties, axis=0)
    scaled_properties = (properties - properties_mean) / properties_std

    # Extraer nombres de las características
    feature_names = parsed_file.columns[1:8]

    return countries_list, scaled_properties, feature_names

# Leer y normalizar los datos usando parse_csv
countries, X_normalized, feature_names = parse_csv('europe.csv')

standardized_df = pd.DataFrame(X_normalized, columns=feature_names)

# # Crear el gráfico de boxplots
# plt.figure(figsize=(12, 8))
# standardized_df.boxplot()
# plt.title('Boxplots of the Standardized Economic, Social, and Geographic Variables')
# plt.xticks(rotation=45)
# plt.xlabel('Variables')
# plt.ylabel('Standardized Values')
# plt.grid(True)
# plt.show()




def oja_rule(X, initial_learning_rate=0.5, epochs=200):
    # Número de características
    n_features = X.shape[1]
    
    # Inicializar los pesos aleatoriamente entre 0 y 1
    w = np.random.uniform(0, 1, n_features)
    
    # Iterar a través de las épocas
    for epoch in range(epochs):
        # Ajustar la tasa de aprendizaje
        learning_rate = initial_learning_rate / (epoch + 1)
        
        for x in X:
            # Calcular la salida actual
            O = np.dot(x, w)

            # Actualización de los pesos según la regla de Oja
            w += learning_rate * (O * x - (O ** 2) * w)
            
            # Normalizar los pesos para mantener estabilidad
            norm = np.linalg.norm(w)
            if norm != 0:
                w = w / norm
    
    # Calcular el autovalor como la media del cuadrado de las salidas
    eigenvalue = np.mean([np.dot(x, w)**2 for x in X])
    
    # Normalizar los pesos finales antes de devolver
    final_norm = np.linalg.norm(w)
    
    return w / final_norm if final_norm != 0 else w, learning_rate, epochs, eigenvalue

# Aplicar la regla de Oja y obtener el último learning rate, número de epochs y autovalor
w_oja, last_learning_rate, total_epochs, eigenvalue = oja_rule(X_normalized, initial_learning_rate=0.5)

# Gráfico para visualizar los valores del autovector
def plot_eigenvector_values(w, feature_names, last_learning_rate, total_epochs, eigenvalue):
    # Ordenar los valores del autovector y las características para que los positivos y negativos estén juntos
    sorted_indices = np.argsort(w)
    sorted_w = w[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]
    
    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    
    # Definir colores para los valores positivos y negativos (más agradables)
    colors = ['red' if value < 0 else 'green' for value in sorted_w]
    
    # Crear las barras
    bars = plt.bar(range(len(sorted_w)), sorted_w, color=colors)
    
    # Añadir etiquetas a las barras para indicar el valor
    for index, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=9, color='black')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height - 0.02, f'{height:.4f}', 
                    ha='center', va='top', fontsize=9, color='black')

    
    # Añadir cuadro con información de epochs, último learning rate y autovalor
    info_text = f'Epochs: {total_epochs}\nLast LR: {last_learning_rate:.4f}\nEigenvalue: {eigenvalue:.4f}'
    plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes,
             fontsize=10, color='red', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Configurar el eje x con los nombres de las características
    plt.xticks(range(len(sorted_w)), sorted_features, rotation=45, ha='right')
    
    # Ajustar los márgenes en el eje y
    plt.ylim(min(sorted_w) - 0.1, max(sorted_w) + 0.1)
    
    plt.title("Valores del Autovector (PC1) usando la Regla de Oja")
    plt.xlabel("Características")
    plt.ylabel("Valores del Autovector")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Llamar a la función para mostrar el gráfico
#plot_eigenvector_values(w_oja, feature_names, last_learning_rate, total_epochs, eigenvalue)





# Gráfico para visualizar los valores de PC1 por país
def plot_countries_pc1(countries, X_normalized, w_oja, last_learning_rate, total_epochs):
    # Calcular los valores de PC1 para cada país
    pc1_values = np.dot(X_normalized, w_oja)
    
    # Crear un DataFrame para facilitar la visualización
    df_pc1 = pd.DataFrame({'Country': countries, 'PC1': pc1_values})
    
    # Ordenar el DataFrame para mejorar la visualización
    df_pc1_sorted = df_pc1.sort_values(by='PC1')
    
    # Crear el gráfico de barras horizontales por país
    plt.figure(figsize=(12, 10))
    
    # Definir colores para los valores positivos y negativos
    colors = ['gray' if value < 0 else 'skyblue' for value in df_pc1_sorted['PC1']]
    
    # Crear las barras
    bars = plt.barh(df_pc1_sorted['Country'], df_pc1_sorted['PC1'], color=colors)
    
    for bar in bars:
            width = bar.get_width()
            if width > 0:
                plt.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                        va='center', ha='left', color='black')
            elif width < 0:
                plt.text(width - 0.05, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                        va='center', ha='right', color='black')
    
    # Añadir cuadro con información de epochs y último learning rate
    info_text = f'Epochs: {total_epochs}\nLast LR: {last_learning_rate:.4f}'
    plt.text(0.98, 0.03, info_text, transform=plt.gca().transAxes,
             fontsize=10, color='red', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.xlabel('PC1')
    plt.title('Valores de PC1 para los Países Europeos')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Llamar a la función para mostrar el gráfico
#plot_countries_pc1(countries, X_normalized, w_oja, last_learning_rate, total_epochs)

#--------------------------------------------

import seaborn as sns

# Crear un DataFrame para visualización
pc1_values = np.dot(X_normalized, w_oja)
df_pc1 = pd.DataFrame({"Country": countries, "PC1": pc1_values})
df_pc1_sorted = df_pc1.sort_values(by="PC1", ascending=False)

# Gráfico de Heatmap para la contribución de cada país a `PC1` en horizontal
def plot_pc1_heatmap_horizontal(df_pc1_sorted):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_pc1_sorted.set_index("Country"), cmap="coolwarm", annot=True, cbar_kws={"label": "Contribution to PC1"}, 
                linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of PC1 Contributions by Country (Horizontal)")
    plt.xlabel("Contribution to PC1")
    plt.ylabel("Countries")
    plt.show()

# Llamar a la función para mostrar el Heatmap en horizontal
#plot_pc1_heatmap_horizontal(df_pc1_sorted)


# Gráfico de Círculos Proporcionales para `PC1`
def plot_pc1_bubble_chart(pc1_values, countries):
    plt.figure(figsize=(12, 8))
    
    # Escalar los valores de PC1 para que el tamaño de los círculos sea proporcional
    sizes = np.abs(pc1_values) * 300
    
    plt.scatter(countries, pc1_values, s=sizes, alpha=0.6, c=pc1_values, cmap="coolwarm", edgecolors="w", linewidth=0.5)
    plt.xticks(rotation=90)
    plt.xlabel("Countries")
    plt.ylabel("Contribution to PC1")
    plt.title("Bubble Chart of Contribution to PC1 by Country")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.colorbar(label="PC1 Value")
    plt.show()

# Llamar a la función para mostrar el gráfico de burbujas
#plot_pc1_bubble_chart(pc1_values, countries)


#-----------------------------

# Realizar el análisis PCA usando tu función `perform_pca_analysis` de `pca.py`
pca_results = perform_pca_analysis('europe.csv')

# Obtener los valores de PC1 del PCA manual para comparación
manual_pc1 = pca_results["countries_pc"]["PC1"]



def oja_rule_with_mse(X, manual_pc1, initial_learning_rate=0.5, epochs=200):
    n_features = X.shape[1]
    w = np.random.uniform(0, 1, n_features)
    mse_values = []  # Para almacenar el MSE en cada época

    for epoch in range(epochs):
        learning_rate = initial_learning_rate / (epoch + 1)
        for x in X:
            O = np.dot(x, w)
            w += learning_rate * (O * x - (O ** 2) * w)
            norm = np.linalg.norm(w)
            if norm != 0:
                w = w / norm
        
        # Calcular el PC1 usando los pesos actuales
        oja_pc1 = np.dot(X, w)

        # Calcular el MSE comparando Oja PC1 con PCA PC1
        mse = np.mean((oja_pc1 - manual_pc1) ** 2)
        mse_values.append(mse)

    # Calcular el autovalor como la media del cuadrado de las salidas
    eigenvalue_oja = np.mean([np.dot(x, w) ** 2 for x in X])
    
    final_norm = np.linalg.norm(w)
    return w / final_norm if final_norm != 0 else w, mse_values, learning_rate, eigenvalue_oja

# Aplicar la regla de Oja con el cálculo del MSE
w_oja_mse, mse_values, last_learning_rate, eigenvalue_oja = oja_rule_with_mse(X_normalized, manual_pc1, initial_learning_rate=0.5)

# Realizar PCA usando la librería y obtener el autovalor correspondiente
pca = PCA()
pca.fit(X_normalized)
eigenvalue_pca = pca.explained_variance_[0]

# Función para graficar el MSE a lo largo de las épocas
def plot_mse_across_epochs(mse_values, last_learning_rate, eigenvalue_oja, eigenvalue_pca):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_values) + 1), mse_values, color='blue', marker='o', markersize=2)
    plt.xlabel('Épocas')
    plt.ylabel('MSE (Oja PC1 vs. PCA PC1)')
    plt.title('Evolución de MSE a lo Largo de las Épocas')
    plt.grid(True)

    # Añadir cuadro con último LR, autovalor de Oja y PCA en la parte superior derecha
    info_text = (f'Último LR: {last_learning_rate:.4f}\n'
                 f'Autovalor (Oja): {eigenvalue_oja:.4f}\n'
                 f'Autovalor (PCA): {eigenvalue_pca:.4f}')
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=10, color='red', ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    plt.show()

# Llamar a la función para mostrar el gráfico del MSE
#plot_mse_across_epochs(mse_values, last_learning_rate, eigenvalue_oja, eigenvalue_pca)




def plot_eigenvector_differences(oja_eigenvector, pca_eigenvector, feature_names):
    # Calcular la diferencia entre los autovectores
    differences = oja_eigenvector - pca_eigenvector

    # Ordenar los valores y características de menor a mayor según los valores de los autovectores de Oja
    sorted_indices = np.argsort(oja_eigenvector)
    oja_eigenvector = oja_eigenvector[sorted_indices]
    pca_eigenvector = pca_eigenvector[sorted_indices]
    feature_names = np.array(feature_names)[sorted_indices]

    # Crear el gráfico de comparación de autovectores
    plt.figure(figsize=(10, 8))

    # Dibujar cajas para los autovectores de Oja y PCA
    bar_width = 0.35  # Ancho de las cajas
    x = np.arange(len(feature_names))

    # Dibujar cajas para Oja y PCA sin transparencia
    bars_oja = plt.bar(x - bar_width/2, oja_eigenvector, width=bar_width, color='green', label='Oja Eigenvector', zorder=3)
    bars_pca = plt.bar(x + bar_width/2, pca_eigenvector, width=bar_width, color='orange', label='PCA Eigenvector', zorder=3)

    # Dibujar líneas verticales para conectar los puntos de Oja y PCA y mostrar la diferencia
    for i in range(len(feature_names)):
        # Conectar los puntos de Oja y PCA con una línea vertical negra
        plt.plot([x[i] - bar_width/2, x[i] + bar_width/2], 
                 [oja_eigenvector[i], pca_eigenvector[i]], 
                 color='black', linestyle='--', alpha=0.7, zorder=2)

        # Añadir la diferencia por encima del valor más alto si es positivo, debajo si es negativo
        diff_text = f'{differences[sorted_indices[i]]:.4f}'
        if oja_eigenvector[i] > 0 or pca_eigenvector[i] > 0:
            max_val = max(oja_eigenvector[i], pca_eigenvector[i])
            plt.text(x[i], max_val + 0.02, diff_text, ha='center', va='bottom', color='black', fontsize=9)
        else:
            min_val = min(oja_eigenvector[i], pca_eigenvector[i])
            plt.text(x[i], min_val - 0.02, diff_text, ha='center', va='top', color='black', fontsize=9)

    # Configurar la visualización del gráfico
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.xlabel("Características")
    plt.ylabel("Valor del Autovector")
    plt.title("Comparación de Autovectores (Oja vs. PCA) con Diferencias")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Llamar a la función para mostrar el gráfico de comparación
#plot_eigenvector_differences(w_oja_mse, pca.components_[0], feature_names)





#---------------------

def oja_rule_with_fixed_mse(X, manual_pc1, initial_weights, learning_rate=0.5, epochs=60, variable_lr=False):
    n_features = X.shape[1]
    w = np.array(initial_weights)  # Usar los pesos iniciales proporcionados
    mse_values = []  # Para almacenar el MSE en cada época

    for epoch in range(epochs):
        # Asegurarse de que el primer paso se realice con el `learning_rate` inicial y luego se ajuste
        if variable_lr and epoch == 0:
            lr = learning_rate  # Usar el `learning_rate` inicial en la primera época
        elif variable_lr:
            lr = learning_rate / (epoch + 1)  # Learning rate variable a partir de la segunda época
        else:
            lr = learning_rate  # Learning rate fijo
        
        for x in X:
            O = np.dot(x, w)
            w += lr * (O * x - (O ** 2) * w)
            norm = np.linalg.norm(w)
            if norm != 0:
                w = w / norm
        
        # Calcular el PC1 usando los pesos actuales
        oja_pc1 = np.dot(X, w)

        # Calcular el MSE comparando Oja PC1 con PCA PC1
        mse = np.mean((oja_pc1 - manual_pc1) ** 2)
        mse_values.append(mse)

    return mse_values

# Establecer una semilla para la reproducibilidad
np.random.seed(40)
initial_weights = np.random.uniform(0, 1, X_normalized.shape[1])

# Aplicar la regla de Oja con tres configuraciones de learning rate usando los mismos pesos iniciales
mse_variable_lr = oja_rule_with_fixed_mse(X_normalized, manual_pc1, initial_weights, learning_rate=0.5, variable_lr=True)
mse_fixed_lr_0_02 = oja_rule_with_fixed_mse(X_normalized, manual_pc1, initial_weights, learning_rate=0.02, variable_lr=False)
mse_fixed_lr_0_1 = oja_rule_with_fixed_mse(X_normalized, manual_pc1, initial_weights, learning_rate=0.3, variable_lr=False)
mse_fixed_lr_0_001 = oja_rule_with_fixed_mse(X_normalized, manual_pc1, initial_weights, learning_rate=0.001, variable_lr=False)

# Función para graficar el MSE para los tres tipos de learning rate
def plot_mse_across_epochs_comparison(mse_variable, mse_lr_0_02, mse_lr_0_1, mse_lr_0_001):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(mse_variable) + 1)
    
    # Graficar los MSE de las tres configuraciones
    plt.plot(epochs, mse_variable, color='blue', marker='o', markersize=2, label='LR Variable (0.5 / epoch)')
    plt.plot(epochs, mse_lr_0_02, color='green', marker='s', markersize=2, label='LR Fijo (0.02)')
    plt.plot(epochs, mse_lr_0_1, color='red', marker='^', markersize=2, label='LR Fijo (0.3)')
    plt.plot(epochs, mse_lr_0_001, color='orange', marker='p', markersize=2, label='LR Fijo (0.001)')
    
    plt.xlabel('Épocas')
    plt.ylabel('MSE (Oja PC1 vs. PCA PC1)')
    plt.title('Evolución del MSE con Diferentes Learning Rates')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Llamar a la función para mostrar el gráfico del MSE comparativo
plot_mse_across_epochs_comparison(mse_variable_lr, mse_fixed_lr_0_02, mse_fixed_lr_0_1,mse_fixed_lr_0_001)

