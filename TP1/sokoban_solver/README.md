# TP SIA - GRUPO 5

## Integrantes
- Madero Torres, Eduardo Federico - 59494  
- Ramos Marca, María Virginia - 67200  
- Pluss, Ramiro - 66254  
- Kuchukhidze, Giorgi - 67262  

## Descripción del Trabajo

**Trabajo Práctico Número 1 - Sistemas de Inteligencia Artificial - ITBA**

En el presente trabajo se desarrolla un motor de búsquedas de soluciones para el juego **SOKOBAN**. Además, se propone un visualizador para guiar al usuario con la solución encontrada. 

### Métodos de Búsqueda Implementados

Hemos implementado tanto métodos de búsqueda informados como desinformados:

- **Métodos desinformados**:  
  - BFS (Breadth-First Search)  
  - DFS (Depth-First Search)  

- **Métodos informados**:  
  - Greedy  
  - A*  

### Heurísticas Utilizadas

Se han seleccionado tres heurísticas admisibles:

- **h1**: Distancia mínima del jugador a la caja más cercana.  
- **h2**: Suma de las distancias mínimas de cada caja a cualquier objetivo.  
- **h3**: Combinación de h1 y h2.  

Además, hemos implementado una heurística no admisible (**h4**) que incluye una penalización basada en la proximidad a las paredes y se calcula de la siguiente manera:

- **h4**: Suma de la distancia mínima del jugador a la caja más cercana, la suma de las distancias mínimas de cada caja a cualquier objetivo, y una penalización adicional por la cercanía de las cajas a las paredes.

### Loggin de Algoritmos

Se proporciona logging para los algoritmos Greedy y A* con el fin de monitorear los nodos explorados y el tamaño de la frontera durante la ejecución.

## Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas y archivos:

- **SRC**:  
  - **game.py**: Define el estado del juego y sus métodos asociados como `get_successors`, `move`, `is_goal`, entre otros.  
  - **heuristics.py**: Describe las heurísticas utilizadas en los algoritmos de búsqueda.  
  - **search.py**: Implementa los algoritmos de búsqueda utilizados.  
  - **visualizer.py**: Utiliza la librería Pygame para ofrecer una visualización de las soluciones encontradas.  

- **levels**:  
  Contiene cuatro niveles de juego:  
  - nivel 1 (easy)  
  - nivel 2 (medium)  
  - nivel 3 (difficult)  
  - un nivel sin solución (impossible)  

- **images**:  
  Contiene las imágenes utilizadas en la implementación de los niveles.

## Instrucciones para Correr el Motor de Búsquedas

1. Debes asegurarte de estar en la carpeta raíz del proyecto (`sokoban_solver`).
2. Configura el archivo `config.json` para seleccionar el nivel y el algoritmo de búsqueda que deseas utilizar. Ejemplo de configuración:

```json
{
  "level_file": "levels\\level1.txt",
  "algorithm": "greedy",
  "heuristic": "h4_heuristic"
}
```

3. Puedes elegir entre los siguientes algoritmos:
   - `bfs` (Breadth-First Search)
   - `dfs` (Depth-First Search)
   - `iddfs` (Iterative Deepening Depth-First Search)
   - `greedy`
   - `astar` (A*)

4. Solo para los algoritmos `greedy` y `astar`, puedes usar las siguientes heurísticas:
   - `h1_heuristic`
   - `h2_heuristic`
   - `h3_heuristic`
   - `h4_heuristic`

5. Para correr el motor de búsqueda, ejecuta el siguiente comando en tu terminal:

```bash
python main.py
```

Esto desplegará la solución encontrada para el nivel seleccionado.

## Procesamiento de Datos

Se ha escrito un script para correr todas las combinaciones posibles entre tablero y nivel, y heurística si corresponde. Para ello, ejecutamos el archivo `multiple_run.py` de la siguiente manera:

```bash
python multiple_run.py
```

Este comando generará un archivo llamado `results.csv` que contiene los resultados de la ejecución, incluyendo el tiempo de ejecución, los pasos, la profundidad, los nodos frontera y los nodos expandidos.

Después de generar el archivo `results.csv`, se realizará una limpieza de los datos mediante el uso de dataframes ejecutando el script `sokoban_analysis.py`. Este script depositará los resultados limpios (calculando el promedio y la desviación estándar) en el archivo `estadisticas.csv`.

Para graficar los resultados, se ha creado un tercer script que utiliza el archivo `estadisticas.csv` para comparar cada nivel con la métrica desarrollada.

```bash
python sokoban_histogram_analysys.py
```

De esta manera se generarán 12 imágenes representativas del análisis, teniendo en cuenta las siguientes métricas:

   - `Cantidad de pasos por nivel para cada método de búsqueda`
   - `Cantidad de nodos explorados/expandidos por nivel para cada método de búsqueda`
   - `Cantidad de nodos frontera por nivel para cada método de búsqueda`
   - `Tiempo de resolución de de nodos explorados/expandidos por nivel para cada método de búsqueda`


## Correcciones y notas


Nota: 6 (seis)

**Correcciones**
Se debieron haber elegido otras heurísticas. No fue una selección acorde a lo que presentamos.
Se debió haber hecho un análisis más detallado de los cambios.
El hecho de elegir el algoritmo que menos tiempo tarde en darse cuenta que no había una solución, sirve, pero debe ser justificado con que según el HARDWARE que tengamos puede ser que el cálculo de los métodos que tienen heurísticas tarde más, sugirieron mostrar más comparaciones.



Preguntas:
- Bajo que condiciones BFS es ÓPTIMO 
- ¿Qué se entiende con inminencia de heurísticas?
- ¿Por qué es importante considerar los estados repetidos?
- ¿Qué ventajas tiene BFS sobre DFS?
  
