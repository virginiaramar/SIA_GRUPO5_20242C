import pandas as pd
import matplotlib.pyplot as plt
import json

# Leer el archivo CSV
df = pd.read_csv('hibrido_mage.csv')

simulation_name = 'Mage'
# Obtener el "Best Fitness" de la última generación para cada simulación
last_gen_best_fitness = df[df['Generation'] == df['Generation'].max()].groupby('Simulation')['Best Fitness'].max()

# Calcular el promedio del "Best Fitness" de la última generación para todas las simulaciones
global_average_fitness = last_gen_best_fitness.mean()

# Encontrar la simulación con el "Best Fitness" más cercano al promedio global
closest_simulation = last_gen_best_fitness.sub(global_average_fitness).abs().idxmin()

# Filtrar los datos para la simulación seleccionada
df_selected_simulation = df[df['Simulation'] == closest_simulation]

# Calcular la varianza de cada atributo por generación para la simulación seleccionada
# Aquí usamos directamente las columnas de varianza (v_*), ajusta según tus datos
diversity = df_selected_simulation.groupby('Generation').agg({
    'v_strength': 'mean',
    'v_agility': 'mean',
    'v_expertise': 'mean',
    'v_endurance': 'mean',
    'v_health': 'mean',
    'v_height': 'mean'
})

print(f"Simulación seleccionada: {closest_simulation}")
print(diversity)

# Configuración del algoritmo genético
with open('config/config.json', 'r') as f:
    config = json.load(f)

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

# Graficar la varianza de cada atributo por generación
plt.figure(figsize=(16, 8))
for column in diversity.columns:
    plt.plot(diversity.index, diversity[column], marker='o', label=column)

plt.title(f'Diversity of Best Fitness by Generation for the Best Combination of {simulation_name} for Simulation {closest_simulation}')
plt.xlabel('Generation')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)

# Ajustar el espacio para el texto de configuración
plt.subplots_adjust(right=0.75)

# Agregar la configuración al gráfico
plt.gcf().text(0.79, 0.5, config_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.show()
