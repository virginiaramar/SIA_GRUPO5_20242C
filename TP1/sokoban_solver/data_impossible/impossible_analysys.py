import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_data(input_file):
    df = pd.read_csv(input_file, header=None, names=[
        'algorithm', 'level', 'heuristic', 'execution_time', 'moves', 'nodes_expanded', 'frontier_size', 'depth'
    ])

    df['method'] = df.apply(lambda row: f"{row['algorithm']}_{row['heuristic']}" if pd.notna(row['heuristic']) else row['algorithm'], axis=1)
    
    # Convertir tiempo a milisegundos
    df['execution_time'] *= 1000

    # Calcular estadísticas de tiempo
    stats = df.groupby('method')['execution_time'].agg(['mean', 'std']).reset_index()
    stats.columns = ['method', 'time_avg', 'time_std']
    
    return stats, df

def create_styled_histogram(stats):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    order = ['bfs', 'dfs', 'iddfs', 
             'astar_h1_heuristic', 'astar_h2_heuristic', 'astar_h3_heuristic', 'astar_h4_heuristic',
             'greedy_h1_heuristic', 'greedy_h2_heuristic', 'greedy_h3_heuristic', 'greedy_h4_heuristic']

    method_labels = {
        'bfs': 'BFS', 'dfs': 'DFS', 'iddfs': 'IDDFS',
        'astar_h1_heuristic': 'A* H1', 'astar_h2_heuristic': 'A* H2',
        'astar_h3_heuristic': 'A* H3', 'astar_h4_heuristic': 'A* H4',
        'greedy_h1_heuristic': 'Greedy H1', 'greedy_h2_heuristic': 'Greedy H2',
        'greedy_h3_heuristic': 'Greedy H3', 'greedy_h4_heuristic': 'Greedy H4'
    }

    # Asegurar que stats contiene todos los métodos en el orden correcto
    stats = stats.set_index('method').reindex(order).reset_index()
    stats = stats.fillna(0)  # Rellenar con 0 si algún método no tiene datos

    # Crear el gráfico de barras
    ax = sns.barplot(x='method', y='time_avg', data=stats, 
                     order=order, color='lightblue', edgecolor='black')

    # Configurar el título y etiquetas
    plt.title('Tiempo de Ejecución Promedio para Nivel IMPOSSIBLE con cada Método de Búsqueda', fontsize=16)
    plt.xlabel('Método de Búsqueda', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (ms)', fontsize=12)
    
    # Rotar las etiquetas del eje x y usar las etiquetas personalizadas
    ax.set_xticklabels([method_labels[m] for m in order], rotation=45, ha='right')

    # Añadir barras de error (desviación estándar)
    ax.errorbar(x=range(len(stats)), y=stats['time_avg'], 
                yerr=stats['time_std'], fmt='none', c='red', capsize=5)

    # Añadir etiquetas de valor en las barras
    for i, v in enumerate(stats['time_avg']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

    # Ajustar los límites del eje y para que las barras de error no se corten
    plt.ylim(0, max(stats['time_avg'] + stats['time_std']) * 1.1)

    plt.tight_layout()
    plt.savefig('histograma_tiempo_ejecucion_impossible.png')
    plt.close()

    print("Histograma de tiempo de ejecución para nivel IMPOSSIBLE generado y guardado.")

if __name__ == "__main__":
    input_file = "impossible.csv"
    stats, df = process_data(input_file)
    create_styled_histogram(stats)