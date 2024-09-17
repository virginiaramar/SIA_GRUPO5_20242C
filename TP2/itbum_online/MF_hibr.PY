import pandas as pd
import matplotlib.pyplot as plt

# Archivo CSV y su configuración
file = 'hibrido_mage.csv'
simulation_name = 'Best Combination with Mage'
color = 'orange'
output_filename = 'max_hybrid_mage.png'
character_name = 'Mage'

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

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
        color=color, markersize=8, linestyle='-', linewidth=0.5, label=f'{simulation_name} Best Fitness of Last Generation')

# Añadir la sombra desde el "Best Fitness" hasta el "Worst Fitness" para visualizar el rango
ax.fill_between(last_generation_fitness['Simulation'], 
                last_generation_fitness['Worst Fitness'],
                last_generation_fitness['Best Fitness'],
                color=color, alpha=0.15, label=f'{simulation_name} Fitness Range (Worst to Best)')

# Añadir una línea horizontal que representa el máximo total
ax.axhline(y=overall_max_fitness, color='red', linestyle='--', label=f'{simulation_name} Overall Average Max Fitness')

# Añadir el número del "Overall Max" a la derecha del gráfico
ax.text(len(last_generation_fitness['Simulation']) + 1.7,  # Reducir desplazamiento horizontal
        overall_max_fitness,  # Ajustar la posición
        f'Overall Max: {overall_max_fitness:.2f}', color=color, fontsize=12,
        verticalalignment='center', horizontalalignment='left')

# Configurar etiquetas y leyenda
ax.set_xlabel('Simulation', fontsize=14)
ax.set_ylabel('Fitness', fontsize=14)
ax.set_title(f'Fitness Range of Last Generation for {simulation_name}', fontsize=16)
ax.tick_params(axis='x', rotation=45)

# Ajustar el espacio para la leyenda y el texto de configuración
plt.subplots_adjust(right=0.65)  # Ajustar el margen derecho para dejar espacio

# Colocar la leyenda en la parte superior derecha del gráfico
ax.legend(loc='lower right')

# Configurar el fondo de la gráfica (el área de la gráfica) en blanco
ax.set_facecolor('white')

# Ajustar el layout para que no se corten las etiquetas
plt.tight_layout()

# Guardar la gráfica como archivo PNG
plt.savefig(output_filename)

# Mostrar gráfica
plt.show()
