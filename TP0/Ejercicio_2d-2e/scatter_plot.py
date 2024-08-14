import csv
import sys
import matplotlib.pyplot as plt

def read_csv_and_plot(file_path):
    levels = []
    successes = []
    capture_rates = []

    # Leer el archivo CSV
    with open(file_path, 'r') as csvfile:
        pokemon_name = next(csvfile).strip()  # Leer el nombre del Pokémon
        csv_reader = csv.DictReader(csvfile)
        
        for row in csv_reader:
            levels.append(int(row['Level']))
            successes.append(row['Success'] == 'True')
            capture_rates.append(float(row['Capture Rate']))

    # Crear el gráfico de barras para los éxitos de captura
    plt.figure(figsize=(12, 6))
    plt.bar(levels, successes, color=['green' if s else 'red' for s in successes])
    plt.xlabel('Nivel del Pokémon')
    plt.ylabel('Captura Exitosa')
    plt.title(f'Éxitos de Captura por Nivel para {pokemon_name}')
    plt.xticks(range(min(levels), max(levels)+1, 5))
    plt.yticks([0, 1], ['Fallido', 'Exitoso'])

    # Añadir un scatter plot para la tasa de captura
    ax2 = plt.twinx()
    ax2.scatter(levels, capture_rates, color='blue', label='Tasa de Captura')
    ax2.set_ylabel('Tasa de Captura', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.tight_layout()
    plt.savefig(f'{pokemon_name}_capture_analysis_noisy.png')
    print(f"Gráfico guardado como {pokemon_name}_capture_analysis_noisy.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <ruta_del_archivo_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    read_csv_and_plot(file_path)
