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

# Lista de nombres de archivos CSV y sus tamaños de población
files = ['TP_100.csv', 'TP_150.csv', 'TP_200.csv']
simulation_names = ['Total Points of 100', 'Total Points of 150', 'Total Points of 200']
colors = ['blue', 'green', 'purple']  # Colores para cada tamaño de población

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Procesar cada archivo CSV
for i, (file, name, color) in enumerate(zip(files, simulation_names, colors)):
    # Leer los datos del archivo CSV
    df = pd.read_csv(file)
    
    # Obtener el "Best Fitness" y "Worst Fitness" de la última generación para cada simulación
    last_generation_fitness = df.groupby('Simulation').apply(
        lambda x: x[x['Generation'] == x['Generation'].max()].agg({'Best Fitness': 'max', 'Average Fitness': 'min'})
    ).reset_index()

    last_generation_fitness.columns = ['Simulation', 'Best Fitness', 'Worst Fitness']
    
    # Calcular el máximo total del 'Best Fitness' entre todas las simulaciones
    overall_max_fitness = last_generation_fitness['Best Fitness'].mean()
    
    # Graficar el "Best Fitness" de la última generación para cada simulación
    ax.plot(last_generation_fitness['Simulation'], last_generation_fitness['Best Fitness'], 'o-', 
            color=color, markersize=8, linestyle='-', linewidth=0.5, label=f'{name} Best Fitness of Last Generation')
    
    # Añadir la sombra desde el "Best Fitness" hasta el "Worst Fitness" para visualizar el rango
    ax.fill_between(last_generation_fitness['Simulation'], 
                    last_generation_fitness['Worst Fitness'],
                    last_generation_fitness['Best Fitness'],
                    color=color, alpha=0.15, label=f'{name} Fitness Range (Worst to Best)')  # alpha reducido a 0.15 para un tono más claro
    
    # Añadir una línea horizontal que representa el máximo total
    ax.axhline(y=overall_max_fitness, color=color, linestyle='--', label=f'{name} Overall Average Max Fitness')
    
# Añadir el número del "Overall Max" a la derecha del gráfico
    ax.text(len(last_generation_fitness['Simulation']) + 0.5,  # Reducir desplazamiento horizontal
            overall_max_fitness + 0.3 * (i - 1),  # Ajustar el valor multiplicador para que los textos se separen claramente
            f'Overall Max: {overall_max_fitness:.2f}', color=color, fontsize=12,
            verticalalignment='center', horizontalalignment='left')


# Configurar etiquetas y leyenda
ax.set_xlabel('Simulation', fontsize=14)
ax.set_ylabel('Fitness', fontsize=14)
ax.set_title('Fitness Range of Last Generation per Simulation for Different Total Points', fontsize=16)
ax.tick_params(axis='x', rotation=45)

# Ajustar el espacio para la leyenda y el texto de configuración
plt.subplots_adjust(right=0.65)  # Ajustar el margen derecho para dejar espacio

# Colocar la leyenda fuera del gráfico a la derecha y abajo
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Mover la leyenda más abajo, ajustando 'bbox_to_anchor'

# Agregar la configuración al gráfico en un área de texto separada
#fig.text(0.75, 0.4, config_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Configurar el fondo de la gráfica (el área de la gráfica) en blanco
ax.set_facecolor('white')

# Ajustar el layout para que no se corten las etiquetas
plt.tight_layout()

# Guardar la gráfica como archivo PNG
plt.savefig('max_fitness_comparison.png')  

# Mostrar gráfica
plt.show()
