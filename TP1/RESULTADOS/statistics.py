import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# En ese code se utiliza data.csv para entrar a todos los datos y poder hacer análisis con ellos
# data.csv contiene 7 columnas con los siguientes nombres
# | NIVEL | METODO | FINAL | HEURISTICA | TIEMPO | PASOS | N_EXTENDIDOS | N_FRONTERA |
# Donde:
# NIVEL (easy, medium, difficult, impossible), METODO (dfs, bfs, iddfs, a_star, greedy), FINAL (EXITO o FRACASO),
# HEURISTICA (H1, H2, H3, H4), TIEMPO (tiempo de ejecución z), PASOS (número de paso hasta solucións),
# N_EXTENDIDOS (número de nodos extendidos), N_FRONTERA (número de nodos en frontera).

#----------------

# Primero vamos a leer el csv
df = pd.read_csv("data.csv")

# Crear un DataFrame para almacenar los resultados de estadísticas
result_df = pd.DataFrame(columns=['NIVEL', 'METODO', 'HEURISTICA', 'TIEMPO_PROM', 'TIEMPO_STD', 'PASOS_PROM', 'PASOS_STD', 'N_EXTENDIDOS_PROM', 'N_EXTENDIDOS_STD', 'N_FRONTERA_PROM', 'N_FRONTERA_STD'])

# Definir los métodos que utilizan heurísticas
heuristic_methods = ['a_star', 'greedy']

# Iterar sobre las combinaciones de NIVEL y METODO
for nivel in df['NIVEL'].unique():
    for metodo in df['METODO'].unique():
        if metodo in heuristic_methods:
            # Si el método usa heurística, iterar también sobre HEURISTICA
            for heuristica in df[df['METODO'] == metodo]['HEURISTICA'].unique():
                subset = df[(df['NIVEL'] == nivel) & 
                            (df['METODO'] == metodo) & 
                            (df['HEURISTICA'] == heuristica)]

                if not subset.empty:
                    # Calcular estadísticas
                    tiempo_prom = subset['TIEMPO'].mean()
                    tiempo_std = subset['TIEMPO'].std()
                    pasos_prom = subset['PASOS'].mean()
                    pasos_std = subset['PASOS'].std()
                    n_extendidos_prom = subset['N_EXTENDIDOS'].mean()
                    n_extendidos_std = subset['N_EXTENDIDOS'].std()
                    n_frontera_prom = subset['N_FRONTERA'].mean()
                    n_frontera_std = subset['N_FRONTERA'].std()

                    # Añadir resultados al DataFrame
                    result_df = result_df.append({
                        'NIVEL': nivel,
                        'METODO': metodo,
                        'HEURISTICA': heuristica,
                        'TIEMPO_PROM': tiempo_prom,
                        'TIEMPO_STD': tiempo_std,
                        'PASOS_PROM': pasos_prom,
                        'PASOS_STD': pasos_std,
                        'N_EXTENDIDOS_PROM': n_extendidos_prom,
                        'N_EXTENDIDOS_STD': n_extendidos_std,
                        'N_FRONTERA_PROM': n_frontera_prom,
                        'N_FRONTERA_STD': n_frontera_std
                    }, ignore_index=True)
        else:
            # Si el método no usa heurística, no se considera la columna HEURISTICA
            subset = df[(df['NIVEL'] == nivel) & (df['METODO'] == metodo)]

            if not subset.empty:
                # Calcular estadísticas
                tiempo_prom = subset['TIEMPO'].mean()
                tiempo_std = subset['TIEMPO'].std()
                pasos_prom = subset['PASOS'].mean()
                pasos_std = subset['PASOS'].std()
                n_extendidos_prom = subset['N_EXTENDIDOS'].mean()
                n_extendidos_std = subset['N_EXTENDIDOS'].std()
                n_frontera_prom = subset['N_FRONTERA'].mean()
                n_frontera_std = subset['N_FRONTERA'].std()

                # Añadir resultados al DataFrame
                result_df = result_df.append({
                    'NIVEL': nivel,
                    'METODO': metodo,
                    'HEURISTICA': None,  # No se usa heurística
                    'TIEMPO_PROM': tiempo_prom,
                    'TIEMPO_STD': tiempo_std,
                    'PASOS_PROM': pasos_prom,
                    'PASOS_STD': pasos_std,
                    'N_EXTENDIDOS_PROM': n_extendidos_prom,
                    'N_EXTENDIDOS_STD': n_extendidos_std,
                    'N_FRONTERA_PROM': n_frontera_prom,
                    'N_FRONTERA_STD': n_frontera_std
                }, ignore_index=True)

# Guardar los resultados en un archivo CSV
result_df.to_csv('statistics_summary.csv', index=False)



