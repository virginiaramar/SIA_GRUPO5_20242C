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

# Lista de nombres de archivos CSV
files = ['simulations_warrior.csv', 'simulations_archer.csv', 'simulations_guardian.csv', 'simulations_mage.csv']

# Lista de nombres de simulaciones correspondientes
simulation_names = ['Warrior', 'Archer', 'Guardian', 'Mage']

# Crear gráficos para cada archivo CSV
for file, name in zip(files, simulation_names):
    # Leer los datos del archivo CSV
    df = pd.read_csv(file)
    
    # Calcular el máximo y la desviación estándar del 'Fitness' para cada simulación
    simulation_stats = df.groupby('Simulation')['Average Fitness'].agg(['max', 'std']).reset_index()
    simulation_stats.columns = ['Simulation', 'Max Fitness', 'Fitness Std']
    
    # Calcular el máximo total del 'Fitness' entre todas las simulaciones
    overall_max_fitness = simulation_stats['Max Fitness'].mean()
    
    # Crear la figura y los subgráficos, reduciendo el tamaño del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))  # Tamaño reducido para el gráfico
    
    # Graficar el máximo del 'Fitness' para cada simulación con puntos en azul oscuro y más grandes
    ax.plot(simulation_stats['Simulation'], simulation_stats['Max Fitness'], 'o-', 
            color='darkblue', markersize=8, linestyle='-', linewidth=0.5, label='Max Fitness per Simulation')
    
    # Añadir la sombra de desviación estándar alrededor de los puntos en azul clarito
    ax.fill_between(simulation_stats['Simulation'], 
                    simulation_stats['Max Fitness'] - simulation_stats['Fitness Std'],
                    simulation_stats['Max Fitness'] + simulation_stats['Fitness Std'],
                    color='lightblue', alpha=0.5, label='Standard Deviation')
    
    # Añadir una línea horizontal que representa el máximo total
    ax.axhline(y=overall_max_fitness, color='r', linestyle='--', label='Overall Max Fitness')
    
    # Añadir el número de la línea horizontal en la gráfica
    ax.text(simulation_stats['Simulation'].iloc[-1], overall_max_fitness + 0.02,
            f'Overall Max: {overall_max_fitness:.2f}', color='r', fontsize=12,
            verticalalignment='bottom', horizontalalignment='right')
    
    # Configurar etiquetas y leyenda
    ax.set_xlabel('Simulation', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)
    ax.set_title(f'Maximum Fitness of All Generations per Simulation of {name} with Standard Deviation', fontsize=16)
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
    plt.savefig(f'max_fitness_{name.lower()}.png')  
    
    # Mostrar gráfica
    plt.show()

    # Mostrar el máximo total de todas las simulaciones para cada archivo
    print(f'Overall Max Fitness for {name}: {overall_max_fitness:.2f}')





