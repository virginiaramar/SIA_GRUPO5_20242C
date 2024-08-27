import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def process_data(input_file, output_file):
    # Leer el archivo CSV sin encabezados
    df = pd.read_csv(input_file, header=None, names=[
        'algorithm', 'level', 'heuristic', 'execution_time', 'moves', 'nodes_expanded', 'frontier_size', 'depth'
    ])

    # Extraer el nombre del nivel del path
    df['level'] = df['level'].str.extract(r'level(\d+)')

    # Combinar algoritmo y heurística
    df['method'] = df.apply(lambda row: f"{row['algorithm']}_{row['heuristic']}" if row['heuristic'] != 'N/A' else row['algorithm'], axis=1)

    # Calcular estadísticas
    stats = df.groupby(['level', 'method']).agg({
        'moves': ['mean', 'std'],
        'execution_time': ['mean', 'std'],
        'nodes_expanded': ['mean', 'std'],
        'frontier_size': ['mean', 'std']
    }).reset_index()

    # Aplanar los nombres de las columnas
    stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stats.columns.values]

    # Renombrar columnas para claridad
    stats = stats.rename(columns={
        'moves_mean': 'moves_avg',
        'execution_time_mean': 'time_avg',
        'nodes_expanded_mean': 'nodes_expanded_avg',
        'frontier_size_mean': 'frontier_size_avg'
    })

    # Imprimir información de depuración
    print("Columnas en el DataFrame de estadísticas:")
    print(stats.columns)
    print("\nPrimeras filas del DataFrame de estadísticas:")
    print(stats.head())

    # Guardar estadísticas en un nuevo CSV
    stats.to_csv(output_file, index=False)
    print(f"Estadísticas guardadas en {output_file}")

    return stats

def create_graphs(stats):
    # Configurar el estilo de seaborn
    sns.set(style="whitegrid")

    # Verificar que 'level' está en las columnas
    if 'level' not in stats.columns:
        print("Error: La columna 'level' no está presente en el DataFrame.")
        return

    # Lista de métricas para graficar
    metrics = [
        ('moves_avg', 'Cantidad de Pasos Promedio'),
        ('time_avg', 'Tiempo de Ejecución Promedio (s)'),
        ('nodes_expanded_avg', 'Cantidad de Nodos Expandidos Promedio'),
        ('frontier_size_avg', 'Tamaño de Frontera Promedio')
    ]

    # Crear un gráfico para cada nivel y métrica
    for level in stats['level'].unique():
        level_data = stats[stats['level'] == level]

        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Métricas para el Nivel {level}', fontsize=16)

        for (i, (metric, title)) in enumerate(metrics):
            ax = axs[i // 2, i % 2]
            
            # Crear el gráfico de barras
            sns.barplot(x='method', y=metric, data=level_data, ax=ax)
            
            # Configurar el título y etiquetas
            ax.set_title(title)
            ax.set_xlabel('Método')
            ax.set_ylabel('Valor')
            
            # Rotar las etiquetas del eje x para mejor legibilidad
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Añadir barras de error (desviación estándar)
            error_metric = f"{metric.replace('avg', 'std')}"
            if error_metric in level_data.columns:
                ax.errorbar(x=range(len(level_data)), y=level_data[metric], 
                            yerr=level_data[error_metric], fmt='none', c='red', capsize=5)

            # Añadir etiquetas de valor en las barras
            for j, v in enumerate(level_data[metric]):
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'graficos_nivel_{level}.png')
        plt.close()

    print("Gráficos generados y guardados.")

if __name__ == "__main__":
    input_file = "results.csv"
    output_file = "estadisticas.csv"
    stats = process_data(input_file, output_file)
    create_graphs(stats)