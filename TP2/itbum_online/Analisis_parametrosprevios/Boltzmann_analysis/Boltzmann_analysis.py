import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Lista de archivos en el formato B_Tmin_Tmax.csv
files = ['BU_0.1_1.0.csv', 'BU_0.1_3.0.csv', 'BU_0.1_5.0.csv', 'BU_0.5_1.0.csv', 'BU_0.5_3.0.csv', 'BU_0.5_5.0.csv', 'BU_0.9_1.0.csv', 'BU_0.9_3.0.csv', 'BU_0.9_5.0.csv']

# Crear una lista para almacenar los resultados
results = []

# Leer los datos de cada archivo y extraer el Best Fitness
for file in files:
    # Extraer Tmin y Tmax del nombre del archivo
    parts = file.replace('BU_', '').replace('.csv', '').split('_')
    Tmin = float(parts[0])
    Tmax = float(parts[1])
    
    # Leer el archivo CSV
    df = pd.read_csv(file)
    
    # Obtener el Best Fitness de la última generación
    best_fitness = df[df['Generation'] == df['Generation'].max()]['Best Fitness'].max()
    
    # Almacenar los resultados
    results.append({'Tmin': Tmin, 'Tmax': Tmax, 'Best Fitness': best_fitness})

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Convertir el DataFrame en una tabla pivote
heatmap_data = results_df.pivot('Tmin', 'Tmax', 'Best Fitness')

# Crear el mapa de calor
plt.figure(figsize=(10, 8))  # Ajustar el tamaño del gráfico
sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Best Fitness'}, annot_kws={"size": 12})  # Aumentar el tamaño de las anotaciones
plt.title('Best Fitness for Different Tmin and Tmax Values Combined with Universal', fontsize=16)  # Aumentar el tamaño del título
plt.xlabel('Tmax', fontsize=14)  # Aumentar el tamaño de la etiqueta del eje X
plt.ylabel('Tmin', fontsize=14)  # Aumentar el tamaño de la etiqueta del eje Y
plt.xticks(fontsize=12)  # Aumentar el tamaño de las etiquetas de los ticks del eje X
plt.yticks(fontsize=12)  # Aumentar el tamaño de las etiquetas de los ticks del eje Y
plt.show()
