import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_data(input_file):
    df = pd.read_csv(input_file)
    print("Columnas originales:")
    print(df.columns)
    
    df.columns = df.columns.str.rstrip('_')
    
    column_mapping = {
        'level': 'level',
        'method': 'method',
        'moves_avg': 'moves_avg',
        'moves_std': 'moves_std',
        'time_avg': 'time_avg',
        'execution_time_std': 'time_std',
        'nodes_expanded_avg': 'nodes_expanded_avg',
        'nodes_expanded_std': 'nodes_expanded_std',
        'frontier_size_avg': 'frontier_size_avg',
        'frontier_size_std': 'frontier_size_std'
    }
    df = df.rename(columns=column_mapping)
    
    # Eliminar sufijos '_nan' en los métodos
    df['method'] = df['method'].str.replace('_nan', '')
    
    # Convertir tiempo a milisegundos para el nivel 1
    df.loc[df['level'] == 1, 'time_avg'] *= 1000
    df.loc[df['level'] == 1, 'time_std'] *= 1000
    
    print("\nColumnas después del procesamiento:")
    print(df.columns)
    
    return df

def create_refined_graphs(df):
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    metrics = [
        ('time_avg', 'time_std', 'Tiempo de Resolución'),
        ('moves_avg', 'moves_std', 'Cantidad de Pasos'),
        ('nodes_expanded_avg', 'nodes_expanded_std', 'Cantidad de Nodos Expandidos'),
        ('frontier_size_avg', 'frontier_size_std', 'Tamaño de Frontera')
    ]

    order = ['bfs', 'dfs', 'iddfs', 
             'astar_h1_heuristic', 'astar_h2_heuristic', 'astar_h3_heuristic', 'astar_h4_heuristic',
             'greedy_h1_heuristic', 'greedy_h2_heuristic', 'greedy_h3_heuristic', 'greedy_h4_heuristic']

    method_labels = {
        'bfs': 'BFS', 'dfs': 'DFS', 
        'astar_h1_heuristic': 'A* H1', 'astar_h2_heuristic': 'A* H2',
        'astar_h3_heuristic': 'A* H3', 'astar_h4_heuristic': 'A* H4',
        'greedy_h1_heuristic': 'Greedy H1', 'greedy_h2_heuristic': 'Greedy H2',
        'greedy_h3_heuristic': 'Greedy H3', 'greedy_h4_heuristic': 'Greedy H4'
    }

    for level in df['level'].unique():
        level_data = df[df['level'] == level].copy()
        level_data['method'] = pd.Categorical(level_data['method'], categories=order, ordered=True)
        level_data = level_data.sort_values('method')
        
        for mean_col, std_col, title in metrics:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            
            if mean_col not in level_data.columns:
                print(f"Advertencia: La columna {mean_col} no está presente. Saltando este gráfico.")
                continue
            
            if std_col in level_data.columns:
                yerr = level_data[std_col]
            else:
                print(f"Advertencia: La columna {std_col} no está presente. No se mostrarán barras de error para {mean_col}.")
                yerr = None
            
            bars = ax.bar(range(len(level_data)), level_data[mean_col], 
                          yerr=yerr, capsize=5, color='lightblue', edgecolor='black', 
                          error_kw={'ecolor': 'red', 'capthick': 2, 'elinewidth': 2})

            # Ajustar etiquetas según nivel y unidad
            if level == 1 and mean_col == 'time_avg':
                title_unit = '(ms)'
            elif mean_col == 'time_avg':
                title_unit = '(s)'
            else:
                title_unit = ''
            
            ax.set_ylabel(f'{title} {title_unit}', fontsize=14)
            ax.set_title(f'{title} {title_unit} para Nivel {level} con cada Método de Búsqueda', fontsize=16)
            
            ax.set_xlabel('Método de Búsqueda', fontsize=14)
            
            ax.set_xticks(range(len(level_data)))
            ax.set_xticklabels([method_labels.get(m, m) for m in level_data['method']], rotation=45, ha='right', fontsize=12)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if level == 1 and mean_col == 'time_avg':
                    label = f'{height:.2f} ms'
                elif mean_col == 'time_avg':
                    label = f'{height:.2f} s'
                else:
                    label = f'{height:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(f'grafico_nivel_{level}_{mean_col}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Gráficos refinados generados y guardados.")

if __name__ == "__main__":
    input_file = "estadisticas.csv"
    df = process_data(input_file)
    create_refined_graphs(df)
