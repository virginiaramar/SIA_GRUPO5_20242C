# Guía para Generación y Análisis de Configuraciones

### 1. Cambiar al directorio del proyecto
```bash
cd itbum_online
```

### 2. Ejecutar el generador de selección
```bash
python generator_selection.py
```

> Otros generadores disponibles son `generator_crossover.py`, `generator_mutation.py`, `generator_replacement.py`. Cada generador creará las configuraciones en la carpeta especificada en el código, por ejemplo:
> 
```python
os.makedirs('config/hibrido/crossover', exist_ok=True)
```

### 3. Ejecutar el análisis principal con simulaciones
```bash
python main_analysis.py --history --simulations 20
```
> **Nota:** Este comando ejecuta el análisis principal tomando en cuenta el historial de ejecuciones y realiza 20 simulaciones.

### 4. Configurar la carpeta a analizar
> Si cambias el generador, asegúrate de cambiar la carpeta que se va a analizar en el script, por ejemplo:
> 
```python
folder_to_analyze = 'hibrido/replacement'  # Línea 83
```

### 5. Comparar los resultados
```bash
python compare_results.py
```

> Este comando te dará una salida con el resumen de los resultados obtenidos. 

### 6. Ajustar la selección de los mejores y peores resultados
> Si hay menos de 20 resultados, es necesario cambiar el número de archivos seleccionados para los 10 mejores y 10 peores. Esto implica modificar las líneas en el script:
> 
```python
# Encontrar los archivos con el mejor promedio de aptitud
best_files = unique_results_df.nlargest(10, 'Average Best Fitness')

# Encontrar los archivos con el peor promedio de aptitud
worst_files = unique_results_df.nsmallest(10, 'Average Best Fitness')

print("Top 10 files with the best average fitness:")
print(best_files)

print("\\nTop 10 files with the worst average fitness:")
print(worst_files)
```

> Ajusta el número 10 a un valor menor si hay menos de 20 resultados en total, tal como se hace en el caso de 'crossover'.

### 7. Cambiar el directorio en el comparador si cambias el generador
> Si cambias de generador, también debes cambiar la ruta del archivo en el comparador:
> 
```python
directory = 'output/hibrido/selection'  # Línea 50
```

> Asegúrate de actualizar esta línea para que apunte al directorio correcto donde se encuentran los archivos CSV generados por el nuevo generador.

### Ejemplo de los resultados
```
Top 10 files with the best average fitness:
                                                 File  Average Best Fitness
15      C_0_P_boltzmann_boltzmann_R_ranking_elite.csv             52.718630
1     C_0_P_boltzmann_boltzmann_R_boltzmann_elite.csv             52.306290
5   C_0_P_boltzmann_boltzmann_R_boltzmann_tourname...             52.278120
4   C_0_P_boltzmann_boltzmann_R_boltzmann_tourname...             52.073090
13    C_0_P_boltzmann_boltzmann_R_elite_universal.csv             51.942825
21  C_0_P_boltzmann_boltzmann_R_roulette_boltzmann...             51.913170
6   C_0_P_boltzmann_boltzmann_R_boltzmann_universa...             51.882075
17   C_0_P_boltzmann_boltzmann_R_ranking_roulette.csv             51.850540
18  C_0_P_boltzmann_boltzmann_R_ranking_tournament...             51.803720
0   C_0_P_boltzmann_boltzmann_R_boltzmann_boltzman...             51.798725

Top 10 files with the worst average fitness:
                                                 File  Average Best Fitness
2   C_0_P_boltzmann_boltzmann_R_boltzmann_ranking.csv             50.083515
24  C_0_P_boltzmann_boltzmann_R_roulette_roulette.csv             50.419665
7     C_0_P_boltzmann_boltzmann_R_elite_boltzmann.csv             50.680635
20  C_0_P_boltzmann_boltzmann_R_ranking_universal.csv             50.752930
9       C_0_P_boltzmann_boltzmann_R_elite_ranking.csv             50.825515
3   C_0_P_boltzmann_boltzmann_R_boltzmann_roulette...             50.880455
11  C_0_P_boltzmann_boltzmann_R_elite_tournament_T...             50.912285
19  C_0_P_boltzmann_boltzmann_R_ranking_tournament...             51.170055
8         C_0_P_boltzmann_boltzmann_R_elite_elite.csv             51.184615
16    C_0_P_boltzmann_boltzmann_R_ranking_ranking.csv             51.205865

