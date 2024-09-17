import pandas as pd
import matplotlib.pyplot as plt

# Lista de archivos CSV y nombres de personajes
elit_files = ['elit_warrior.csv', 'elit_archer.csv', 'elit_guardian.csv', 'elit_mage.csv']
rand_files = ['rand_warrior.csv', 'rand_archer.csv', 'rand_guardian.csv', 'rand_mage.csv']
character_names = ['Warrior', 'Archer', 'Guardian', 'Mage']
colors = ['blue', 'green', 'purple', 'orange']  # Colores para las gráficas elitistas

# Procesar cada par de archivos CSV (elitista y aleatoria) y generar la gráfica
for elit_file, rand_file, character_name, color in zip(elit_files, rand_files, character_names, colors):
    # Leer los archivos CSV
    df_elit = pd.read_csv(elit_file)
    df_rand = pd.read_csv(rand_file)

    # Obtener el "Best Fitness" de la última generación para cada simulación (elitista)
    last_gen_best_fitness_elit = df_elit[df_elit['Generation'] == df_elit['Generation'].max()].groupby('Simulation')['Best Fitness'].max()

    # Calcular el promedio del "Best Fitness" de la última generación para todas las simulaciones (elitista)
    average_last_gen_best_fitness_elit = last_gen_best_fitness_elit.mean()

    # Encontrar la simulación con el "Best Fitness" más cercano al promedio (elitista)
    closest_simulation_elit = last_gen_best_fitness_elit.sub(average_last_gen_best_fitness_elit).abs().idxmin()

    # Filtrar los datos para la simulación seleccionada (elitista)
    df_selected_simulation_elit = df_elit[df_elit['Simulation'] == closest_simulation_elit]

    # Obtener los valores de "Best Fitness" por generación (elitista)
    best_fitness_per_generation_elit = df_selected_simulation_elit.groupby('Generation')['Best Fitness'].max()

    # Obtener el tiempo total y la razón de parada (elitista)
    total_time_elit = df_selected_simulation_elit['Time Taken'].iloc[-1]
    stop_reason_elit = df_selected_simulation_elit['Stop Reason'].iloc[-1]

    # Repetir el mismo proceso para la configuración aleatoria
    last_gen_best_fitness_rand = df_rand[df_rand['Generation'] == df_rand['Generation'].max()].groupby('Simulation')['Best Fitness'].max()
    average_last_gen_best_fitness_rand = last_gen_best_fitness_rand.mean()
    closest_simulation_rand = last_gen_best_fitness_rand.sub(average_last_gen_best_fitness_rand).abs().idxmin()
    df_selected_simulation_rand = df_rand[df_rand['Simulation'] == closest_simulation_rand]
    best_fitness_per_generation_rand = df_selected_simulation_rand.groupby('Generation')['Best Fitness'].max()
    total_time_rand = df_selected_simulation_rand['Time Taken'].iloc[-1]
    stop_reason_rand = df_selected_simulation_rand['Stop Reason'].iloc[-1]

    print(f"Selected Elitist Simulation for {character_name}: {closest_simulation_elit}")
    print(f"Total Time Taken (Elitist): {total_time_elit} seconds")
    print(f"Stop Reason (Elitist): {stop_reason_elit}")
    print(f"Selected Random Simulation for {character_name}: {closest_simulation_rand}")
    print(f"Total Time Taken (Random): {total_time_rand} seconds")
    print(f"Stop Reason (Random): {stop_reason_rand}")

    # Graficar el "Best Fitness" por generación para ambas configuraciones
    plt.figure(figsize=(16, 8))
    plt.plot(best_fitness_per_generation_elit.index, best_fitness_per_generation_elit.values, marker='o', color=color, label='Elitist Best Fitness')
    plt.plot(best_fitness_per_generation_rand.index, best_fitness_per_generation_rand.values, marker='x', linestyle='--', color='red', label='Random Best Fitness')

    plt.title(f'Best Fitness by Generation for {character_name} in Elitist and Exploratory Configurations', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Best Fitness', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Añadir cuadros de texto con la información de tiempo y razón de parada
    ax = plt.gca()
    textstr_elit = f'Elitist - Total Time: {total_time_elit:.2f} seconds\nStop Reason: {stop_reason_elit}'
    textstr_rand = f'Random - Total Time: {total_time_rand:.2f} seconds\nStop Reason: {stop_reason_rand}'
    
    # Configuración del color de fondo para las cajas de texto
    elit_props = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.15, edgecolor='black')  # Alpha reducido para mayor opacidad
    rand_props = dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.15, edgecolor='black')  # Alpha reducido para mayor opacidad
    
    # Agregar los textos dentro de la gráfica, ajustando la posición vertical
    ax.text(0.95, 0.15, textstr_elit, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=elit_props)
    ax.text(0.95, 0.05, textstr_rand, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=rand_props)


    # Guardar la gráfica como archivo PNG
    plt.savefig(f'best_fitness_combined_{character_name.lower()}.png')

    # Mostrar la gráfica
    #plt.show()
