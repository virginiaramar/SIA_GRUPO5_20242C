import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear la carpeta GRAPHS si no existe
output_dir = 'GRAPHS'
os.makedirs(output_dir, exist_ok=True)

# Leer el archivo CSV con los resultados de estadísticas
df = pd.read_csv('statistics_summary.csv')

# Función para agregar etiquetas a las barras con métodos y heurísticas
def label_bar(row):
    if pd.notna(row['HEURISTICA']):
        return f"{row['METODO']} ({row['HEURISTICA']})"
    else:
        return row['METODO']

# Función para generar el título personalizado
def generar_titulo(nivel, metric, metodo, heuristica):
    metric_title_map = {
        'TIEMPO_PROM': 'Tiempo',
        'PASOS_PROM': 'Pasos',
        'N_EXTENDIDOS_PROM': 'Nodos Extendidos',
        'N_FRONTERA_PROM': 'Nodos en Frontera'
    }
    base_title = f"{metric_title_map[metric]} y desviación estándar promedio de 20 iteraciones para el nivel {nivel}"
    
    if pd.notna(heuristica):
        return f"{base_title} con el método {metodo} con {heuristica}"
    else:
        return f"{base_title} con el método {metodo}"

# Crear un diccionario para almacenar las métricas y sus etiquetas
metrics = {
    'TIEMPO_PROM': 'Tiempo Promedio',
    'PASOS_PROM': 'Pasos Promedio',
    'N_EXTENDIDOS_PROM': 'Nodos Extendidos Promedio',
    'N_FRONTERA_PROM': 'Nodos en Frontera Promedio'
}

# Crear gráficas para cada nivel
niveles = df['NIVEL'].unique()

for nivel in niveles:
    # Filtrar los datos por nivel
    df_nivel = df[df['NIVEL'] == nivel]
    
    # Añadir etiquetas para barras
    df_nivel['LABEL'] = df_nivel.apply(label_bar, axis=1)
    
    # Iterar sobre las métricas
    for metric, label in metrics.items():
        std_metric = metric.replace('PROM', 'STD')
        
        # Crear gráfica de barras
        plt.figure(figsize=(10, 6))
        plt.bar(df_nivel['LABEL'], df_nivel[metric], yerr=df_nivel[std_metric], capsize=5, color='skyblue')
        
        # Generar el título personalizado
        for i, row in df_nivel.iterrows():
            title = generar_titulo(nivel, metric, row['METODO'], row['HEURISTICA'])
        
        # Añadir título y etiquetas
        plt.title(title)
        plt.xlabel('Método (y Heurística si aplica)')
        plt.ylabel(label)
        plt.xticks(rotation=45, ha="right")
        
        # Guardar la gráfica como imagen en la carpeta GRAPHS
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{nivel}_{metric}.png')
        plt.savefig(output_path)
        plt.show()
