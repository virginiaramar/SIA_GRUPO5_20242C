import numpy as np
import matplotlib.pyplot as plt
from utils import (
    relu, relu_derivative, sigmoid, sigmoid_derivative,
    tanh, tanh_derivative, mse_loss, mse_loss_derivative,
    cross_entropy_loss, cross_entropy_loss_derivative
)

class Autoencoder:
    def __init__(self, layers, activation_fn_name, loss_fn_name, optimizer, learning_rate, initial_lr=None, decay_rate=None, variable_lr=None, seed=None, init_method="uniform"):
        self.layers = layers
        self.activation_fn_name = activation_fn_name
        self.activation_fn = globals()[activation_fn_name]
        self.activation_derivative_fn = globals()[activation_fn_name + '_derivative']
        self.loss_fn_name = loss_fn_name
        self.loss_fn = globals()[loss_fn_name]
        self.loss_derivative_fn = globals()[loss_fn_name + '_derivative']
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.initial_lr = initial_lr if initial_lr else learning_rate
        self.decay_rate = decay_rate
        self.variable_lr = variable_lr
        self.seed = seed  
        self.init_method = init_method  
        self.weights, self.biases = self.initialize_weights()
        self.optimizer_params = {}
        self.latent_layer_index = len(layers) // 2

    def initialize_weights(self):

        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(None)  # Semilla aleatoria

        weights = {}
        biases = {}
        for i in range(len(self.layers) - 1):
            input_dim = self.layers[i]
            output_dim = self.layers[i + 1]

            if self.init_method == "uniform":
                # Método uniforme aleatorio (ya implementado por ti)
                limit = np.sqrt(6 / (input_dim + output_dim))
                weights[f"W{i+1}"] = np.random.uniform(-limit, limit, (input_dim, output_dim))
            elif self.init_method == "xavier":
                std = np.sqrt(2 / (input_dim + output_dim))
                weights[f"W{i+1}"] = np.random.normal(0, std, (input_dim, output_dim))
            elif self.init_method == "he":
                # Método He
                std = np.sqrt(2 / input_dim)
                weights[f"W{i+1}"] = np.random.normal(0, std, (input_dim, output_dim))
            else:
                raise ValueError(f"Método de inicialización desconocido: {self.init_method}")
            
            biases[f"b{i+1}"] = np.zeros((1, output_dim))
            
        return weights, biases

    def forward_propagation(self, X):
        activations = {"A0": X}
        Z_values = {}
        num_layers = len(self.weights)
        for i in range(1, num_layers + 1):
            W = self.weights[f"W{i}"]
            b = self.biases[f"b{i}"]
            A_prev = activations[f"A{i-1}"]
            Z = np.dot(A_prev, W) + b
            if i == num_layers:
                # Última capa - usar sigmoid
                A = sigmoid(Z)
            elif i == self.latent_layer_index:
                # Capa latente - función de activación lineal
                A = Z
            else:
                A = self.activation_fn(Z)
            Z_values[f"Z{i}"] = Z
            activations[f"A{i}"] = A
        return activations, Z_values

    def backward_propagation(self, X, activations, Z_values):
        grads = {}
        num_layers = len(self.weights)
        m = X.shape[0]

        # Inicializar dA para la capa de salida
        A_final = activations[f"A{num_layers}"]
        dA = self.loss_derivative_fn(X, A_final)

        for i in reversed(range(1, num_layers + 1)):
            A_prev = activations[f"A{i-1}"]
            Z = Z_values[f"Z{i}"]
            if i == num_layers:
                dZ = dA * sigmoid_derivative(Z)
            else:
                dZ = dA * self.activation_derivative_fn(Z)
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            grads[f"dW{i}"] = dW
            grads[f"db{i}"] = db
            if i > 1:
                W = self.weights[f"W{i}"]
                dA = np.dot(dZ, W.T)
        return grads

    def update_parameters(self, grads, epoch):
        num_layers = len(self.weights)
        lr = self.learning_rate
        if self.variable_lr and self.decay_rate:
            lr = self.initial_lr * (self.decay_rate ** epoch)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        if self.optimizer == 'adam':
            # Inicializar momentos si no existen
            if not self.optimizer_params.get('m'):
                self.optimizer_params['m'] = {}
                self.optimizer_params['v'] = {}
                for i in range(1, num_layers + 1):
                    self.optimizer_params['m'][f"dW{i}"] = np.zeros_like(grads[f"dW{i}"])
                    self.optimizer_params['m'][f"db{i}"] = np.zeros_like(grads[f"db{i}"])
                    self.optimizer_params['v'][f"dW{i}"] = np.zeros_like(grads[f"dW{i}"])
                    self.optimizer_params['v'][f"db{i}"] = np.zeros_like(grads[f"db{i}"])
                self.optimizer_params['t'] = 0

            self.optimizer_params['t'] += 1
            t = self.optimizer_params['t']
            for i in range(1, num_layers + 1):
                # Actualizar momentos
                self.optimizer_params['m'][f"dW{i}"] = beta1 * self.optimizer_params['m'][f"dW{i}"] + (1 - beta1) * grads[f"dW{i}"]
                self.optimizer_params['m'][f"db{i}"] = beta1 * self.optimizer_params['m'][f"db{i}"] + (1 - beta1) * grads[f"db{i}"]
                self.optimizer_params['v'][f"dW{i}"] = beta2 * self.optimizer_params['v'][f"dW{i}"] + (1 - beta2) * (grads[f"dW{i}"] ** 2)
                self.optimizer_params['v'][f"db{i}"] = beta2 * self.optimizer_params['v'][f"db{i}"] + (1 - beta2) * (grads[f"db{i}"] ** 2)

                # Corregir sesgo
                m_hat_dw = self.optimizer_params['m'][f"dW{i}"] / (1 - beta1 ** t)
                m_hat_db = self.optimizer_params['m'][f"db{i}"] / (1 - beta1 ** t)
                v_hat_dw = self.optimizer_params['v'][f"dW{i}"] / (1 - beta2 ** t)
                v_hat_db = self.optimizer_params['v'][f"db{i}"] / (1 - beta2 ** t)

                # Actualizar parámetros
                self.weights[f"W{i}"] -= lr * m_hat_dw / (np.sqrt(v_hat_dw) + epsilon)
                self.biases[f"b{i}"] -= lr * m_hat_db / (np.sqrt(v_hat_db) + epsilon)
        else:
            # Descenso de gradiente estándar
            for i in range(1, num_layers + 1):
                self.weights[f"W{i}"] -= lr * grads[f"dW{i}"]
                self.biases[f"b{i}"] -= lr * grads[f"db{i}"]

    def train(self, noisy_X, original_X, epochs, iteration=1, save_prefix="training"):
        """
        Entrena el modelo para minimizar la diferencia entre los datos reconstruidos y los originales.
        """
        losses = []
        for epoch in range(epochs):
            # Propagación hacia adelante
            activations, Z_values = self.forward_propagation(noisy_X)

            # Calcular la pérdida con los datos originales
            loss = self.loss_fn(original_X, activations[f"A{len(self.weights)}"])
            losses.append(loss)

            # Retropropagación
            grads = self.backward_propagation(original_X, activations, Z_values)

            # Actualizar parámetros
            self.update_parameters(grads, epoch)

            # Mostrar progreso
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Repetición {iteration}, Época {epoch + 1}/{epochs}, Pérdida: {loss:.6f}")

        # Guardar curva de pérdida
        self.plot_loss_curve(losses, iteration, save_prefix)
        return losses
    
    def plot_loss_curve(self, losses, iteration, save_prefix):
        """
        Genera y guarda una gráfica de la curva de pérdida.
        
        Args:
            losses (list): Lista de pérdidas durante el entrenamiento.
            iteration (int): Número de iteración/repetición actual.
            save_prefix (str): Prefijo base para los nombres de los archivos generados.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(losses) + 1), losses, label='Pérdida de entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title(f'Curva de Pérdida durante el Entrenamiento (Iteración {iteration})')
        plt.legend()
        
        # Guardar el gráfico
        save_path = f"{save_prefix}_iteration_{iteration}_loss.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Liberar memoria
        print(f"Gráfico de pérdida guardado en: {save_path}")



    def reconstruct(self, X):
        activations, _ = self.forward_propagation(X)
        reconstructed = activations[f"A{len(self.weights)}"]
        return reconstructed

    def get_latent_space(self, X):
        activations, _ = self.forward_propagation(X)
        # Suponiendo que el espacio latente está en la capa central
        latent_layer_index = len(self.layers) // 2
        latent_space = activations[f"A{latent_layer_index}"]
        return latent_space

    def save_weights(self, filepath):
        """
        Guarda los pesos y sesgos del modelo en un archivo.

        Args:
            filepath (str): Ruta donde guardar los parámetros del modelo.
        """
        import pickle

        # Crear un diccionario con los pesos y sesgos
        model_parameters = {
            'weights': self.weights,
            'biases': self.biases
        }

        # Guardar en un archivo utilizando pickle
        with open(filepath, 'wb') as file:
            pickle.dump(model_parameters, file)
        print(f"Pesos y sesgos guardados en: {filepath}")

    def decode_latent_points(self, latent_points):
        """
        Decodifica puntos del espacio latente para generar reconstrucciones.

        Args:
            latent_points (np.ndarray): Puntos en el espacio latente (de dimensión igual a la capa latente).

        Returns:
            np.ndarray: Reconstrucciones generadas por el decodificador.
        """
        # Comenzar desde la capa latente
        activations = {f"A{self.latent_layer_index}": latent_points}

        # Pasar los puntos por el decodificador
        for i in range(self.latent_layer_index + 1, len(self.layers)):
            W = self.weights[f"W{i}"]
            b = self.biases[f"b{i}"]
            Z = np.dot(activations[f"A{i-1}"], W) + b

            if i == len(self.layers) - 1:
                # Última capa usa sigmoid
                A = sigmoid(Z)
            else:
                # Capas ocultas usan la función de activación
                A = self.activation_fn(Z)

            activations[f"A{i}"] = A

        # Salida final del decodificador
        return activations[f"A{len(self.layers) - 1}"]
    
    def plot_latent_space_with_generated(self, training_latent_points, character_labels, num_generated_points, save_path=None):
        """
        Grafica el espacio latente con puntos de entrenamiento y puntos generados aleatoriamente
        dentro del rango del conjunto de entrenamiento.

        Args:
            training_latent_points (np.ndarray): Puntos del espacio latente del conjunto de entrenamiento.
            character_labels (list): Etiquetas de los caracteres correspondientes a los puntos de entrenamiento.
            num_generated_points (int): Número de nuevos puntos generados aleatoriamente.
            save_path (str, optional): Ruta para guardar el gráfico. Si no se proporciona, muestra el gráfico.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Obtener rangos del conjunto de entrenamiento
        dim1_min, dim1_max = training_latent_points[:, 0].min(), training_latent_points[:, 0].max()
        dim2_min, dim2_max = training_latent_points[:, 1].min(), training_latent_points[:, 1].max()

        # Generar nuevos puntos aleatorios dentro del rango observado
        random_latent_points = np.random.uniform(
            low=[dim1_min, dim2_min],
            high=[dim1_max, dim2_max],
            size=(num_generated_points, 2)
        )

        # Configuración del gráfico
        plt.figure(figsize=(10, 8))

        # Puntos del conjunto de entrenamiento
        plt.scatter(
            training_latent_points[:, 0], training_latent_points[:, 1],
            label="Letras de Entrenamiento", alpha=0.7, s=50
        )
        for i, label in enumerate(character_labels):
            plt.text(training_latent_points[i, 0] + 2, training_latent_points[i, 1] + 2, label, fontsize=9)

        # Nuevos puntos generados
        plt.scatter(
            random_latent_points[:, 0], random_latent_points[:, 1],
            color='red', marker='x', label="Nuevas Letras Generadas", s=70
        )
        for i, (x, y) in enumerate(random_latent_points):
            plt.text(x + 2, y + 2, f"Gen{i+1}", fontsize=9, color="blue")

        # Configuración del gráfico
        plt.title("Espacio Latente: Letras de Entrenamiento y Generadas")
        plt.xlabel("Dimensión 1")
        plt.ylabel("Dimensión 2")
        plt.legend()
        plt.grid(True)

        # Guardar o mostrar el gráfico
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()

    def generate_latent_points_in_grid(self, num_points, dim1_range, dim2_range):
        """
        Genera puntos aleatorios en una grilla bidimensional dentro de un rango definido.

        Args:
            num_points (int): Número de puntos a generar.
            dim1_range (tuple): Rango de valores para la primera dimensión (min, max).
            dim2_range (tuple): Rango de valores para la segunda dimensión (min, max).

        Returns:
            np.ndarray: Puntos generados aleatoriamente.
        """
        dim1_points = np.random.uniform(dim1_range[0], dim1_range[1], num_points)
        dim2_points = np.random.uniform(dim2_range[0], dim2_range[1], num_points)
        return np.column_stack((dim1_points, dim2_points))
    
    def generate_concept_vector(self, latent_space, character_labels, char_start, char_end, num_steps=10):
        """
        Genera un vector de concepto entre dos letras en el espacio latente y reconstruye los puntos intermedios.

        Args:
            latent_space (np.ndarray): Puntos en el espacio latente del conjunto de entrenamiento.
            character_labels (list): Etiquetas asociadas a los puntos del espacio latente.
            char_start (str): Letra inicial ("o").
            char_end (str): Letra final ("x").
            num_steps (int): Número de puntos intermedios a generar.

        Returns:
            np.ndarray: Reconstrucciones de los puntos intermedios.
        """
        # Obtener índices de las letras inicial y final en el espacio latente
        idx_start = character_labels.index(char_start)
        idx_end = character_labels.index(char_end)
        
        # Obtener coordenadas en el espacio latente
        latent_start = latent_space[idx_start]
        latent_end = latent_space[idx_end]

        # Generar puntos intermedios usando interpolación lineal
        interpolated_points = np.linspace(latent_start, latent_end, num_steps)

        # Decodificar los puntos intermedios
        reconstructed_letters = self.decode_latent_points(interpolated_points)

        return reconstructed_letters
    
    def decode_latent_points(self, latent_points):
        """
        Decodifica puntos en el espacio latente para reconstruir datos.

        Args:
            latent_points (np.ndarray): Puntos en el espacio latente a decodificar.

        Returns:
            np.ndarray: Reconstrucciones decodificadas.
        """
        activations = {}  # Diccionario para almacenar activaciones de cada capa

        # Inicializar el espacio latente como la primera activación de decodificación
        activations[f"A{self.latent_layer_index}"] = latent_points

        # Decodificar desde la capa latente hasta la capa de salida
        for i in range(self.latent_layer_index + 1, len(self.layers)):
            W = self.weights[f"W{i}"]  # Pesos de la capa i
            b = self.biases[f"b{i}"]  # Sesgos de la capa i
            Z = np.dot(activations[f"A{i-1}"], W) + b

            # Aplicar función de activación
            if i == len(self.layers) - 1:
                # Última capa: usar sigmoid
                A = sigmoid(Z)
            else:
                # Capas intermedias: usar la función de activación del modelo
                A = self.activation_fn(Z)

            # Guardar activación de la capa actual
            activations[f"A{i}"] = A

        # Retornar la activación de la última capa (capa de salida)
        return activations[f"A{len(self.layers) - 1}"]
    
    def plot_concept_vector(self, reconstructed_letters, char_start, char_end, save_path=None):
        """
        Muestra la transición generada por el concepto vector en un gráfico.

        Args:
            reconstructed_letters (np.ndarray): Reconstrucciones de los puntos intermedios.
            char_start (str): Letra inicial ("o").
            char_end (str): Letra final ("x").
            save_path (str, optional): Ruta para guardar el gráfico. Si no se proporciona, muestra el gráfico.
        """
        num_steps = len(reconstructed_letters)
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(1 - reconstructed_letters[i].reshape(7, 5), cmap="gray", vmin=0, vmax=1)  # Invertir colores
            ax.axis("off")
            if i == 0:
                ax.set_title(f"{char_start}", fontsize=12)
            elif i == num_steps - 1:
                ax.set_title(f"{char_end}", fontsize=12)
            else:
                ax.set_title(f"Paso {i+1}", fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_latent_space_with_concept_vector(self, latent_space, character_labels, char_start, char_end, interpolated_points, save_path=None):
        """
        Grafica el espacio latente con las letras de entrenamiento, las letras inicial y final, y
        los puntos intermedios generados por el concept vector.

        Args:
            latent_space (np.ndarray): Puntos en el espacio latente del conjunto de entrenamiento.
            character_labels (list): Etiquetas de los caracteres correspondientes a los puntos del espacio latente.
            char_start (str): Letra inicial ("o").
            char_end (str): Letra final ("x").
            interpolated_points (np.ndarray): Puntos intermedios generados por el concept vector.
            save_path (str, optional): Ruta para guardar el gráfico. Si no se proporciona, muestra el gráfico.
        """
        import matplotlib.pyplot as plt

        # Configuración del gráfico
        plt.figure(figsize=(10, 8))

        # Puntos del conjunto de entrenamiento
        plt.scatter(latent_space[:, 0], latent_space[:, 1], alpha=0.7, label="Letras de Entrenamiento", s=50)

        # Anotar etiquetas de las letras de entrenamiento
        for i, label in enumerate(character_labels):
            plt.text(latent_space[i, 0] + 1, latent_space[i, 1] + 1, label, fontsize=9)

        # Obtener los puntos específicos de las letras inicial y final
        idx_start = character_labels.index(char_start)
        idx_end = character_labels.index(char_end)

        start_point = latent_space[idx_start]
        end_point = latent_space[idx_end]

        # Puntos inicial y final
        plt.scatter(start_point[0], start_point[1], color='red', label=f"Letra '{char_start}'", s=100)
        plt.scatter(end_point[0], end_point[1], color='green', label=f"Letra '{char_end}'", s=100)

        # Puntos intermedios generados
        plt.scatter(interpolated_points[:, 0], interpolated_points[:, 1], color='blue', label="Puntos Intermedios", s=70)

        # Conectar los puntos con una línea
        for i, (x, y) in enumerate(interpolated_points):
            plt.text(x + 1, y + 1, f"P{i+1}", fontsize=8, color="blue")
        plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], linestyle="--", color="blue", alpha=0.7)

        # Configuración final del gráfico
        plt.title("Espacio Latente: Vector de Concepto entre Letras")
        plt.xlabel("Dimensión 1")
        plt.ylabel("Dimensión 2")
        plt.legend()
        plt.grid(True)

        # Guardar o mostrar el gráfico
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()
        plt.close()