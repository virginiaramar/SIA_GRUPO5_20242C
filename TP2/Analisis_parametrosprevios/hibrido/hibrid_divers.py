import pandas as pd
import matplotlib.pyplot as plt

# Archivo CSV y su configuración
file = 'hibrido_archer.csv'
simulation_name = 'Hybrid with Archer'
color = 'green'
output_filename = 'diversity_hybrid_archer.png'
character_name = 'Archer'

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Leer los datos del archivo CSV
df = pd.read_csv(file)

# Calcular la diversidad como la desviación estándar del "Best Fitness" por generación
diversity_per_generation = df.groupby('Generation')['Best Fitness'].std()

# Graficar la diversidad por generación
ax.plot(diversity_per_generation.index, diversity_per_generation.values, marker='o', color=color, label=f'{simulation_name} Diversity (Std Dev of Best Fitness)')

# Añadir etiquetas y leyenda
ax.set_xlabel('Generation', fontsize=14)
ax.set_ylabel('Diversity (Std Dev)', fontsize=14)
ax.set_title(f'Diversity of Best Fitness by Generation for {simulation_name}', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True)

# Ajustar el layout para que no se corten las etiquetas
plt.tight_layout()

# Guardar la gráfica como archivo PNG
plt.savefig(output_filename)

# Mostrar gráfica
plt.show()
