## TP SIA - GRUPO 5 

## Integrantes
 Madero Torres, Eduardo Federico - 59494
 Ramos Marca, María Virginia - 67200
 Pluss, Ramiro - 66254
 Kuchukhidze, Giorgi - 67262

## Descripción del Trabajo

**Trabajo Práctico Número 2 - Sistemas de Inteligencia Artificial - ITBA**

Este proyecto implementa un sistema de creación de personajes para el juego ITBUM ONLINE utilizando algoritmos genéticos. El objetivo es generar personajes optimizados basados en diferentes atributos y clases, evaluados por un Entrenador Virtual Experto (EVE).

## Guía de Configuración para ITBUM ONLINE 

**Estructura del archivo config.json**

```json
{
  "genetic_algorithm": {
    "population_size": ,
    "offspring_count": ,
    "crossover": {
      "type": ,
      "rate": 
    },
    "mutation": {
      "type": ,
      "uniform": ,
      "rate": 
    },
    "selection": {
      "parents": {
        "method1": ,
        "method2": ,
        "method1_proportion": 
      },
      "replacement": {
        "method1": ,
        "method2": ,
        "method1_proportion": 
      },
      "boltzmann": {
        "Tmin": ,
        "Tmax": ,
        "k": 
      }
    },
    "replacement_method": ,
    "stop_criteria": {
      "max_generations": ,
      "structure": ,
      "content": ,
      "optimal_fitness": 
    },
    "character_class": ,
    "total_points": 
  },
  "time_limit": 
}
```

## Opciones de Configuración

### Algoritmo Genético

- `population_size`: Tamaño de la población (entero positivo)
- `offspring_count`: Cantidad de hijos a generar por generación (entero positivo)

### Cruce (Crossover)

- `type`: Tipo de cruce
  - Opciones: "one_point", "two_point", "uniform", "arithmetic"
- `rate`: Tasa de cruce (float entre 0 y 1)

### Mutación

- `type`: Tipo de mutación
  - Opciones: "gen", "multigen"
- `uniform`: Si la mutación es uniforme o no (boolean)
- `rate`: Tasa de mutación (float entre 0 y 1)

### Selección

Para `parents` y `replacement`:
- `method1` y `method2`: Métodos de selección
  - Opciones: "tournament", "roulette", "universal", "boltzmann", "ranking", "elite"
- `method1_proportion`: Proporción del primer método (float entre 0 y 1)

#### Método Boltzmann y función de temperatura 

- `Tmin`: Temperatura mínima (float positivo)
- `Tmax`: Temperatura máxima (float positivo)
- `k`: Constante de enfriamiento (float positivo)

### Método de Reemplazo

- `replacement_method`: Método de reemplazo generacional
  - Opciones: 
    - "traditional": Método Tradicional (Fill-All)
    - "young_bias": Método de Sesgo Joven (Fill-Parent)

### Criterios de Parada

- `max_generations`: Número máximo de generaciones (entero positivo)
- `structure`: Umbral de convergencia estructural (float entre 0 y 1)
- `content`: Umbral de convergencia de contenido (float entre 0 y 1)
- `optimal_fitness`: Fitness óptimo objetivo (float positivo)

### Configuración de Personaje

- `character_class`: Clase fija del personaje (opcional)
  - Opciones: 0 (Warrior), 1 (Archer), 2 (Guardian), 3 (Mage), o null para aleatorio
- `total_points`: Total de puntos a distribuir (opcional, entero positivo o null para aleatorio)

### Límite de Tiempo

- `time_limit`: Tiempo límite en segundos (entero positivo)

## Ejemplo de Configuración

```json
{
  "genetic_algorithm": {
    "population_size": 100,
    "offspring_count": 30,
    "crossover": {
      "type": "one_point",
      "rate": 0.8
    },
    "mutation": {
      "type": "multigen",
      "uniform": true,
      "rate": 0.01
    },
    "selection": {
      "parents": {
        "method1": "universal",
        "method2": "roulette",
        "method1_proportion": 0.7
      },
      "replacement": {
        "method1": "universal",
        "method2": "ranking",
        "method1_proportion": 0.2
      },
      "boltzmann": {
        "Tmin": 0.5,
        "Tmax": 2.0,
        "k": 0.1
      }
    },
    "replacement_method": "young_bias",
    "stop_criteria": {
      "max_generations": 20,
      "structure": 0.01,
      "content": 0.01,
      "optimal_fitness": 100.0
    },
    "character_class": null,
    "total_points": null
  },
  "time_limit": 10
}
```

Este ejemplo configura un algoritmo genético con una población de 100 individuos, generando 30 hijos por generación, usando cruce de un punto, mutación multigen uniforme, y una combinación de métodos de selección para padres y reemplazo. El método de reemplazo es de sesgo joven (Fill-Parent). La selección de Boltzmann está configurada con temperaturas mínima y máxima, y una constante de enfriamiento. La clase del personaje y el total de puntos se generarán aleatoriamente.

##  Ejecución del Programa

Para ejecutar el programa:

1. Asegúrate de tener Python instalado.
2. Navega hasta el directorio del proyecto en la terminal.
3. Ejecuta uno de los siguientes comandos:

### Ejecutar el programa sin mostrar el historial:

El programa generará el mejor personaje posible dentro del límite de tiempo especificado en la configuración.

```
python main.py
```

### Ejecutar el programa mostrando el historial de generaciones y un gráfico:

```
python main.py --history
```

## Posibles Extensiones y Mejoras

- Implementación de interfaz gráfica para visualizar la evolución de los personajes.
- Paralelización del algoritmo para mejorar el rendimiento.
- Implementación de más operadores genéticos y métodos de selección.
- Ajuste dinámico de parámetros durante la ejecución del algoritmo.