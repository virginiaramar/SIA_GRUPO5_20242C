import csv
import sys
import matplotlib.pyplot as plt

def read_csv_and_plot(file_path):
    iterations = []
    successes = []
    capture_rates = []

    # Leer el archivo CSV
    with open(file_path, 'r') as csvfile:
        pokemon_name = next(csvfile).strip()  # Leer el nombre del Pokémon
        csv_reader = csv.DictReader(csvfile)
        
        for i, row in enumerate(csv_reader, start=1):
            iterations.append(i)  # Usar el índice de la fila como iteración
            successes.append(row['Success'] == 'True')
            capture_rates.append(float(row['Capture Rate']))

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.bar(iterations, successes, color=['green' if s else 'red' for s in successes])
    plt.xlabel('Iteración')
    plt.ylabel('Captura Exitosa')
    plt.title(f'Éxitos de Captura por Iteración para {pokemon_name}')
    plt.xticks(range(min(iterations), max(iterations)+1, 5))
    plt.yticks([0, 1], ['Fallido', 'Exitoso'])

    # Añadir un scatter plot para la tasa de captura
    ax2 = plt.twinx()
    ax2.scatter(iterations, capture_rates, color='blue', label='Tasa de Captura')
    ax2.set_ylabel('Tasa de Captura', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.tight_layout()
    plt.savefig(f'{pokemon_name}_capture_analysis.png')
    print(f"Gráfico guardado como {pokemon_name}_capture_analysis.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <ruta_del_archivo_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    read_csv_and_plot(file_path)
