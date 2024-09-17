import pandas as pd
import matplotlib.pyplot as plt

# Archivo CSV y nombre del personaje
file = 'ejerc2_mage.csv'
character_name = 'Mage'
color = 'purple'  # Color para la gráfica

# Leer el archivo CSV
df = pd.read_csv(file)

# Obtener el "Best Fitness" de la última generación para cada simulación
last_gen_best_fitness = df[df['Generation'] == df['Generation'].max()].groupby('Simulation')['Best Fitness'].max()

# Calcular el promedio del "Best Fitness" de la última generación para todas las simulaciones
average_last_gen_best_fitness = last_gen_best_fitness.mean()

# Encontrar la simulación con el "Best Fitness" más cercano al promedio
closest_simulation = last_gen_best_fitness.sub(average_last_gen_best_fitness).abs().idxmin()

# Filtrar los datos para la simulación seleccionada
df_selected_simulation = df[df['Simulation'] == closest_simulation]

# Obtener los valores de "Best Fitness" por generación
best_fitness_per_generation = df_selected_simulation.groupby('Generation')['Best Fitness'].max()

# Obtener el tiempo total y la razón de parada
total_time = df_selected_simulation['Time Taken'].iloc[-1]
stop_reason = df_selected_simulation['Stop Reason'].iloc[-1]

print(f"Selected Simulation for {character_name}: {closest_simulation}")
print(f"Total Time Taken: {total_time} seconds")
print(f"Stop Reason: {stop_reason}")

# Graficar el "Best Fitness" por generación
plt.figure(figsize=(16, 8))
plt.plot(best_fitness_per_generation.index, best_fitness_per_generation.values, marker='o', color=color, label='Best Fitness')

plt.title(f'Best Fitness by Generation for {character_name} with a Maximum Time Limit of 30 Minutes', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Best Fitness', fontsize=14)
#plt.legend(fontsize=12)
plt.grid(True)

# Añadir cuadro de texto con la información de tiempo y razón de parada
textstr = f'Total Time: {total_time:.2f} seconds\nStop Reason: {stop_reason}'
props = dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.15, edgecolor='black')
ax = plt.gca()
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Guardar la gráfica como archivo PNG
plt.savefig(f'best_fitness_{character_name.lower()}.png')

# Mostrar la gráfica
plt.show()

# Crear la segunda gráfica con límite en el eje X
plt.figure(figsize=(16, 8))
plt.plot(best_fitness_per_generation.index, best_fitness_per_generation.values, marker='o', color=color, label='Best Fitness')

plt.title(f'Best Fitness by Generation for {character_name} with Limited X-Axis', fontsize=16)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Best Fitness', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Añadir límite en el eje X
plt.xlim(0, 200)  # Ajusta el valor del límite según tu preferencia

# Añadir cuadro de texto con la información de tiempo y razón de parada
ax = plt.gca()
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Guardar la segunda gráfica como archivo PNG
plt.savefig(f'best_fitness_{character_name.lower()}_limited_x.png')

# Mostrar la segunda gráfica
plt.show()
