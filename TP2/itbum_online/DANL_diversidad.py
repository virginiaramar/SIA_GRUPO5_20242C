import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV
df = pd.read_csv('simulations_warrior.csv')

# Función para calcular el índice de diversidad de Shannon
def shannon_index(values):
    if len(values) == 0:  # Evitar problemas con datos vacíos
        return 0
    value_counts = pd.Series(values).value_counts()
    prob = value_counts / len(values)
    # Agregar un pequeño valor para evitar log(0) y calcular el índice de Shannon
    return -np.sum(prob * np.log(prob + 1e-9))

# Calcular el promedio de la aptitud (fitness) global
global_fitness = df['Average Fitness'].mean()

# Agrupar por simulación y calcular el promedio de fitness para cada simulación
simulation_fitness = df.groupby('Simulation').agg({
    'Average Fitness': 'mean'
}).reset_index()

# Encontrar la simulación cuyo promedio de fitness esté más cerca del fitness global
simulation_fitness['Fitness Difference'] = abs(simulation_fitness['Average Fitness'] - global_fitness)
closest_simulation = simulation_fitness.loc[simulation_fitness['Fitness Difference'].idxmin()]

# Filtrar datos para la simulación más cercana
simulation_to_plot = closest_simulation['Simulation']
filtered_data = df[df['Simulation'] == simulation_to_plot]

# Agrupar por generación y calcular el índice de Shannon para cada atributo en esa simulación
shannon_diversity_simulation = filtered_data.groupby('Generation').agg({
    'Strength': lambda x: shannon_index(x),
    'Agility': lambda x: shannon_index(x),
    'Expertise': lambda x: shannon_index(x),
    'Endurance': lambda x: shannon_index(x),
    'Health': lambda x: shannon_index(x),
}).reset_index()

# Verificar valores calculados para depuración
print("Shannon Index values for each generation:")
print(shannon_diversity_simulation)

# Crear la gráfica
plt.figure(figsize=(12, 8))
generations = shannon_diversity_simulation['Generation']

for attribute in ['Strength', 'Agility', 'Expertise', 'Endurance', 'Health']:
    plt.plot(
        generations,
        shannon_diversity_simulation[attribute],
        marker='o',
        label=f'{attribute} Shannon Index'
    )

plt.xlabel('Generation', fontsize=14)
plt.ylabel('Shannon Index', fontsize=14)
plt.title(f'Shannon Index for Attributes in Simulation {simulation_to_plot}', fontsize=16)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.gca().set_facecolor('white')
plt.tight_layout()
plt.show()

print(f"The simulation closest to the global fitness is {simulation_to_plot}")



