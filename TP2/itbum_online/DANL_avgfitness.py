import pandas as pd
import matplotlib.pyplot as plt
import json

# Leer la configuración desde el archivo JSON
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Extraer y formatear la configuración
config_text = (
    "$Population Size:$ {}\n"
    "$Offspring Count:$ {}\n\n"
    "$Selection:$\n  $Method1:$ {} ({:.1f}%)\n  $Method2:$ {} ({:.1f}%)\n"
    "$Crossover:$\n  $Type:$ {}\n  Rate: {}\n"
    "$Mutation:$\n  $Type:$ {}\n  Rate: {}\n  Method: {}\n"
    "$Replacement:$\n  $Method1:$ {} ({:.1f}%)\n  $Method2:$ {} ({:.1f}%)\n"
    "$Replacement Method:$ {}\n\n"
    "$Max Generations:$ {}\n"
    "$Content:$ {}\n"
    "$Optimal Fitness:$ {}\n"
    "$Character Class:$ {}\n"
    "$Total Points:$ {}\n"
    "$Time Limit:$ {}"
).format(
    config['genetic_algorithm']['population_size'],
    config['genetic_algorithm']['offspring_count'],
    config['genetic_algorithm']['selection']['parents']['method1'],
    config['genetic_algorithm']['selection']['parents']['method1_proportion'] * 100,
    config['genetic_algorithm']['selection']['parents']['method2'],
    (1 - config['genetic_algorithm']['selection']['parents']['method1_proportion']) * 100,
    config['genetic_algorithm']['crossover']['type'],
    config['genetic_algorithm']['crossover']['rate'],
    config['genetic_algorithm']['mutation']['type'],
    config['genetic_algorithm']['mutation']['rate'],
    config['genetic_algorithm']['mutation']['uniform'],
    config['genetic_algorithm']['selection']['replacement']['method1'],
    config['genetic_algorithm']['selection']['replacement']['method1_proportion'] * 100,
    config['genetic_algorithm']['selection']['replacement']['method2'],
    (1 - config['genetic_algorithm']['selection']['replacement']['method1_proportion']) * 100,
    config['genetic_algorithm']['replacement_method'],
    config['genetic_algorithm']['stop_criteria']['max_generations'],
    config['genetic_algorithm']['stop_criteria']['content'],
    config['genetic_algorithm']['stop_criteria']['optimal_fitness'],
    config['genetic_algorithm']['character_class'],
    config['genetic_algorithm']['total_points'],
    config['genetic_algorithm']['time_limit']
)

# Lista de nombres de archivos CSV y nombres de simulaciones correspondientes
files = ['HIBRIDO1.csv']
simulation_names = ['Warrior']

# Crear gráficos para cada archivo CSV
for file, name in zip(files, simulation_names):
    # Leer los datos del archivo CSV
    df = pd.read_csv(file)
    
    # Calcular el promedio y la desviación estándar del 'Average Fitness' para cada simulación
    simulation_stats = df.groupby('Simulation')['Average Fitness'].agg(['mean', 'std']).reset_index()
    simulation_stats.columns = ['Simulation', 'Average Fitness Mean', 'Average Fitness Std']
    
    # Calcular el promedio total del 'Average Fitness' entre todas las simulaciones
    overall_avg_fitness = simulation_stats['Average Fitness Mean'].mean()
    
    # Crear la figura y los subgráficos, reduciendo el tamaño del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))  # Tamaño reducido para el gráfico
    
    # Graficar el promedio del 'Average Fitness' para cada simulación con puntos en azul oscuro y más grandes
    ax.plot(simulation_stats['Simulation'], simulation_stats['Average Fitness Mean'], 'o-', 
            color='darkblue', markersize=8, linestyle='-', linewidth=0.5, label='Average Fitness per Simulation')
    
    # Añadir la sombra de desviación estándar alrededor de los puntos en azul clarito
    ax.fill_between(simulation_stats['Simulation'], 
                    simulation_stats['Average Fitness Mean'] - simulation_stats['Average Fitness Std'],
                    simulation_stats['Average Fitness Mean'] + simulation_stats['Average Fitness Std'],
                    color='lightblue', alpha=0.5, label='Standard Deviation')
    
    # Añadir una línea horizontal que representa el promedio total
    ax.axhline(y=overall_avg_fitness, color='r', linestyle='--', label='Overall Average Fitness')
    
    # Añadir el número de la línea horizontal en la gráfica
    ax.text(simulation_stats['Simulation'].iloc[-1], overall_avg_fitness + 0.02,
            f'Overall Avg: {overall_avg_fitness:.2f}', color='r', fontsize=12,
            verticalalignment='bottom', horizontalalignment='right')
    
    # Configurar etiquetas y leyenda
    ax.set_xlabel('Simulation', fontsize=14)
    ax.set_ylabel('Average Fitness', fontsize=14)
    ax.set_title(f'Average Fitness of All Generations per Simulation of {name} with Standard Deviation', fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    
    # Ajustar el espacio para la leyenda y el texto de configuración
    plt.subplots_adjust(right=0.65)  # Ajustar el margen derecho para dejar espacio
    
    # Colocar la leyenda fuera del gráfico a la derecha
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Agregar la configuración al gráfico en un área de texto separada
    fig.text(0.75, 0.5, config_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Configurar el fondo de la gráfica (el área de la gráfica) en blanco
    ax.set_facecolor('white')
    
    # Ajustar el layout para que no se corten las etiquetas
    plt.tight_layout()
    
    # Guardar la gráfica como archivo PNG
    plt.savefig(f'average_fitness_{name.lower()}.png')  
    
    # Mostrar gráfica
    plt.show()

    # Mostrar el promedio total de todas las simulaciones para cada archivo
    print(f'Overall Average Fitness for {name}: {overall_avg_fitness:.2f}')



