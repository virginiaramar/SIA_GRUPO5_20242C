import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from typing import Literal, Optional

class KohonenNetwork:
    def __init__(self, input_dim: int, grid_size: int, input_data: np.ndarray,
                 learning_rate: float = 0.1, radius: Optional[float] = None,
                 weights_init: Literal['random', 'sample'] = 'random',
                 distance_metric: Literal['euclidean', 'exponential'] = 'euclidean',
                 n_iterations: Optional[int] = None,
                 constant_radius: bool = False):  # Nuevo parámetro
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.input_data = input_data
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Radio inicial: si no se especifica, usar el tamaño total de la grilla
        self.initial_radius = radius if radius is not None else grid_size
        self.radius = self.initial_radius
        
        # Número de iteraciones
        self.n_iterations = n_iterations if n_iterations is not None else 500 * input_dim
        
        # Método de distancia
        self.distance_metric = distance_metric
        
        # Nuevo atributo para controlar si el radio se mantiene constante
        self.constant_radius = constant_radius
        
        # Inicializar la grilla de pesos
        self.weights = self._initialize_weights(weights_init)
        
        # Para almacenar las distancias entre neuronas vecinas
        self.neighbor_distances = None
    
    def _initialize_weights(self, method: str) -> np.ndarray:
        """Inicializar pesos según el método especificado"""
        if method == 'random':
            return np.random.rand(self.grid_size, self.grid_size, self.input_dim)
        elif method == 'sample':
            weights = np.zeros((self.grid_size, self.grid_size, self.input_dim))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    weights[i, j] = self.get_random_sample()
            return weights
        else:
            raise ValueError("Método de inicialización no soportado")
    
    def get_random_sample(self) -> np.ndarray:
        """Obtener una muestra aleatoria de los datos de entrada"""
        return np.array(self.input_data[np.random.randint(len(self.input_data))])
    
    def euclidean_distance(self, current_input: np.ndarray) -> np.ndarray:
        """Calcular distancia euclidiana"""
        return np.sqrt(np.sum((self.weights - current_input) ** 2, axis=2))
    
    def exponential_distance(self, current_input: np.ndarray) -> np.ndarray:
        """Calcular distancia exponencial"""
        euclidean = self.euclidean_distance(current_input)
        return np.exp(-(euclidean ** 2))
    
    def _find_winner(self, x: np.ndarray) -> tuple:
        """Encontrar la neurona ganadora según la métrica de distancia especificada"""
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(x)
            return np.unravel_index(np.argmin(distances), distances.shape)
        else:  # exponential
            distances = self.exponential_distance(x)
            return np.unravel_index(np.argmax(distances), distances.shape)
    
    def _get_neighborhood(self, winner: tuple, current_radius: float) -> np.ndarray:
        """Calcular el vecindario de la neurona ganadora"""
        y, x = np.ogrid[0:self.grid_size, 0:self.grid_size]
        distances = np.sqrt((x - winner[1]) ** 2 + (y - winner[0]) ** 2)
        
        # Si el radio actual es 1, solo actualizar vecinos inmediatos
        if current_radius <= 1:
            return distances <= 1
        
        return np.exp(-(distances ** 2) / (2 * (current_radius ** 2)))
    
    def train(self, data: np.ndarray):
        """Entrenar la red de Kohonen"""
        print("\nIniciando entrenamiento...")
        for i in range(self.n_iterations):
            if i % (self.n_iterations // 10) == 0:  # Mostrar progreso cada 10%
                print(f"Progreso: {i/self.n_iterations*100:.1f}%")
                
            # Seleccionar un dato aleatorio
            sample = data[np.random.randint(len(data))]
            
            # Encontrar la neurona ganadora
            winner = self._find_winner(sample)
            
            # Calcular radio actual según la configuración
            if self.constant_radius:
                current_radius = self.initial_radius
            else:
                current_radius = max(1.0, self.initial_radius * (1 - i/self.n_iterations))
            
            # Calcular tasa de aprendizaje actual
            current_lr = self.initial_learning_rate * np.exp(-i / self.n_iterations)
            
            # Obtener vecindario
            neighborhood = self._get_neighborhood(winner, current_radius)
            
            # Actualizar pesos
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    self.weights[x, y] += current_lr * neighborhood[x, y] * (sample - self.weights[x, y])
        print("Entrenamiento completado (100%)")
    
    def get_winner_mapping(self, data: np.ndarray) -> dict:
        """Obtener mapeo de datos a neuronas ganadoras"""
        mapping = {}
        for i, x in enumerate(data):
            winner = self._find_winner(x)
            if winner not in mapping:
                mapping[winner] = []
            mapping[winner].append(i)
        return mapping
    
    def calculate_neighbor_distances(self) -> np.ndarray:
        """Calcular distancias entre neuronas vecinas"""
        distances = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = []
                if i > 0:  # Vecino superior
                    neighbors.append(self.weights[i-1, j])
                if i < self.grid_size-1:  # Vecino inferior
                    neighbors.append(self.weights[i+1, j])
                if j > 0:  # Vecino izquierdo
                    neighbors.append(self.weights[i, j-1])
                if j < self.grid_size-1:  # Vecino derecho
                    neighbors.append(self.weights[i, j+1])
                
                if neighbors:
                    distances[i, j] = np.mean([np.linalg.norm(self.weights[i, j] - neighbor) 
                                             for neighbor in neighbors])
        
        self.neighbor_distances = distances
        return distances

def load_and_preprocess_data(file_path: str) -> tuple:
    """Cargar y preprocesar los datos"""
    data = pd.read_csv(file_path)
    
    # Separar el nombre del país
    countries = data['Country'].values
    
    # Seleccionar solo las características numéricas
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 
                'Pop.growth', 'Unemployment']
    X = data[features].values
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, countries, features

def plot_results(som, data, countries, features, output_dir, plot_feature_heatmaps=True):
    """Generar visualizaciones de los resultados y mostrar agrupaciones"""
    # Crear directorio de salida si no existe
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Mapa de países
    mapping = som.get_winner_mapping(data)
    plt.figure(figsize=(15, 12))
    
    # Crear matrices para el mapa de calor y las etiquetas
    country_map = np.zeros((som.grid_size, som.grid_size))
    country_labels = np.empty((som.grid_size, som.grid_size), dtype=object)
    
    # Llenar las matrices
    for pos, indices in mapping.items():
        country_map[pos] = len(indices)
        countries_list = [countries[i] for i in indices]
        country_labels[pos] = f'({len(countries_list)} países)\n' + '\n'.join(countries_list)
    
    # Crear el mapa de calor con anotaciones usando una paleta pastel
    ax = sns.heatmap(country_map, 
                     cmap='RdPu',
                     annot=country_labels, 
                     fmt='',
                     cbar_kws={'label': 'Cantidad de Países'},
                     square=True)
    
    # Ajustar el tamaño de las anotaciones y su formato
    for t in ax.texts:
        t.set_size(8)
        t.set_verticalalignment('center')
    
    plt.title('Distribución de Países en el Mapa de Kohonen' + 
             f'\nRadio {"Constante" if som.constant_radius else "Decremental"}')
    plt.tight_layout()
    plt.savefig(output_dir / 'country_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distancias entre neuronas vecinas
    distances = som.calculate_neighbor_distances()
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, cmap='Pastel1')
    plt.title('Distancias Promedio entre Neuronas Vecinas' + 
             f'\nRadio {"Constante" if som.constant_radius else "Decremental"}')
    plt.savefig(output_dir / 'neighbor_distances.png')
    plt.close()
    
    # 3. Heatmaps por variable
    if plot_feature_heatmaps:
        # Crear directorio para heatmaps de características
        feature_dir = output_dir / 'feature_heatmaps'
        feature_dir.mkdir(exist_ok=True)
        
        # Calcular valores promedio por neurona para cada característica
        feature_maps = {}
        for i, feature in enumerate(features):
            feature_map = np.zeros((som.grid_size, som.grid_size))
            feature_values = np.zeros((som.grid_size, som.grid_size))
            count_map = np.zeros((som.grid_size, som.grid_size))
            
            # Calcular suma y contador para promedios
            for pos, indices in mapping.items():
                if indices:  # Si hay países en esta posición
                    values = data[indices, i]
                    feature_values[pos] = np.sum(values)
                    count_map[pos] = len(indices)
            
            # Calcular promedios
            mask = count_map > 0
            feature_map[mask] = feature_values[mask] / count_map[mask]
            feature_maps[feature] = feature_map
            
            # Crear visualización
            plt.figure(figsize=(10, 8))
            
            # Preparar etiquetas con valores y países
            value_labels = np.empty((som.grid_size, som.grid_size), dtype=object)
            for pos, indices in mapping.items():
                if indices:
                    countries_here = [countries[i] for i in indices]
                    avg_value = feature_map[pos]
                    value_labels[pos] = f'Valor: {avg_value:.2f}\n({len(countries_here)} países)'
                else:
                    value_labels[pos] = ''
            
            # Crear heatmap
            sns.heatmap(feature_map, 
                       cmap='RdPu',
                       annot=value_labels,
                       fmt='',
                       cbar_kws={'label': feature},
                       square=True)
            
            plt.title(f'Distribución de {feature}\nRadio {"Constante" if som.constant_radius else "Decremental"}')
            plt.tight_layout()
            plt.savefig(feature_dir / f'heatmap_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Crear archivo de asignaciones y mostrar países por neurona
    assignments = {}
    print("\nAgrupaciones encontradas:")
    print("------------------------")
    for pos, indices in mapping.items():
        countries_in_neuron = [countries[i] for i in indices]
        neuron_info = {
            "countries": countries_in_neuron,
            "count": len(countries_in_neuron),
            "average_values": {}
        }
        
        # Agregar valores promedio por característica
        if indices:
            group_data = data[indices]
            group_means = np.mean(group_data, axis=0)
            for feature, mean_value in zip(features, group_means):
                neuron_info["average_values"][feature] = float(mean_value)
        
        assignments[f"Neurona ({pos[0]}, {pos[1]})"] = neuron_info
        
        # Imprimir información
        print(f"\nGrupo en neurona ({pos[0]}, {pos[1]}) - {len(countries_in_neuron)} países:")
        print("  Países:")
        for country in countries_in_neuron:
            print(f"    - {country}")
        
        if indices:
            print("  Características promedio:")
            for feature, mean_value in zip(features, group_means):
                print(f"    - {feature}: {mean_value:.2f}")
    
    # Guardar asignaciones en archivo JSON
    with open(output_dir / 'country_assignments.json', 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

# También necesitamos actualizar el archivo de configuración para incluir la nueva opción
def main():
    """Función principal"""
    # Cargar configuración
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("Cargando datos...")
    X_scaled, countries, features = load_and_preprocess_data(config['data_path'])
    
    # Obtener el valor de constant_radius de la configuración, por defecto False
    constant_radius = config.get('constant_radius', False)
    
    print("\nConfiguración de la red de Kohonen:")
    print("--------------------------------")
    print(f"- Tamaño de grilla: {config['grid_size']}x{config['grid_size']}")
    print(f"- Tasa de aprendizaje: {config['learning_rate']}")
    print(f"- Radio inicial: {config['initial_radius']}")
    print(f"- Radio constante: {constant_radius}")
    print(f"- Número de iteraciones: {config['n_iterations']}")
    print(f"- Método de inicialización: {config['weights_init']}")
    print(f"- Métrica de distancia: {config['distance_metric']}")
    
    # Crear y entrenar la red
    som = KohonenNetwork(
        input_dim=len(features),
        grid_size=config['grid_size'],
        input_data=X_scaled,
        learning_rate=config['learning_rate'],
        radius=config['initial_radius'],
        weights_init=config['weights_init'],
        distance_metric=config['distance_metric'],
        n_iterations=config['n_iterations'],
        constant_radius=constant_radius  # Nueva opción
    )
    
    som.train(X_scaled)
    
    print("\nGenerando visualizaciones y análisis...")
    plot_results(som, X_scaled, countries, features, config['output_dir'])
    print("\nResultados guardados en el directorio 'output'.")

if __name__ == "__main__":
    main()# También necesitamos actualizar el archivo de configuración para incluir la nueva opción
def main():
    """Función principal"""
    # Cargar configuración
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("Cargando datos...")
    X_scaled, countries, features = load_and_preprocess_data(config['data_path'])
    
    # Obtener el valor de constant_radius de la configuración, por defecto False
    constant_radius = config.get('constant_radius', False)
    
    print("\nConfiguración de la red de Kohonen:")
    print("--------------------------------")
    print(f"- Tamaño de grilla: {config['grid_size']}x{config['grid_size']}")
    print(f"- Tasa de aprendizaje: {config['learning_rate']}")
    print(f"- Radio inicial: {config['initial_radius']}")
    print(f"- Radio constante: {constant_radius}")
    print(f"- Número de iteraciones: {config['n_iterations']}")
    print(f"- Método de inicialización: {config['weights_init']}")
    print(f"- Métrica de distancia: {config['distance_metric']}")
    
    # Crear y entrenar la red
    som = KohonenNetwork(
        input_dim=len(features),
        grid_size=config['grid_size'],
        input_data=X_scaled,
        learning_rate=config['learning_rate'],
        radius=config['initial_radius'],
        weights_init=config['weights_init'],
        distance_metric=config['distance_metric'],
        n_iterations=config['n_iterations'],
        constant_radius=constant_radius 
    )
    
    som.train(X_scaled)
    
    print("\nGenerando visualizaciones y análisis...")
    plot_results(som, X_scaled, countries, features, config['output_dir'])
    print("\nResultados guardados en el directorio 'output'.")

if __name__ == "__main__":
    main()