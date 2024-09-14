import pandas as pd
import matplotlib.pyplot as plt

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
    
    # Crear la gráfica
    plt.figure(figsize=(12, 8))
    
    # Graficar el máximo del 'Fitness' para cada simulación con puntos en azul oscuro y más grandes
    plt.plot(simulation_stats['Simulation'], simulation_stats['Max Fitness'], 'o-', 
             color='darkblue', markersize=8, linestyle='-', linewidth=0.5, label='Max Fitness per Simulation')
    
    # Añadir la sombra de desviación estándar alrededor de los puntos en azul clarito
    plt.fill_between(simulation_stats['Simulation'], 
                     simulation_stats['Max Fitness'] - simulation_stats['Fitness Std'],
                     simulation_stats['Max Fitness'] + simulation_stats['Fitness Std'],
                     color='lightblue', alpha=0.5, label='Standard Deviation')
    
    # Añadir una línea horizontal que representa el máximo total
    plt.axhline(y=overall_max_fitness, color='r', linestyle='--', label='Overall Max Fitness')
    
    # Añadir el número de la línea horizontal en la gráfica
    plt.text(simulation_stats['Simulation'].iloc[-1], overall_max_fitness + 0.02,
             f'Overall Max: {overall_max_fitness:.2f}', color='r', fontsize=12,
             verticalalignment='bottom', horizontalalignment='right')
    
    # Configurar etiquetas y leyenda
    plt.xlabel('Simulation', fontsize=14)
    plt.ylabel('Max Fitness', fontsize=14)
    plt.title(f'Maximum Fitness of All Generations per Simulation of {name} with Standard Deviation', fontsize=16)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Configurar el fondo de la gráfica (el área de la gráfica) en blanco
    plt.gca().set_facecolor('white')
    
    # Mostrar gráfica
    plt.tight_layout()  # Ajustar el layout para que no se corten las etiquetas
    plt.savefig(f'max_fitness_{name.lower()}.png')  # Guardar la gráfica como archivo PNG
    plt.show()

    # Mostrar el máximo total de todas las simulaciones para cada archivo
    print(f'Overall Max Fitness for {name}: {overall_max_fitness:.2f}')
