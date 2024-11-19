import numpy as np
import matplotlib.pyplot as plt
from utils import (
    relu, relu_derivative, sigmoid, sigmoid_derivative,
    tanh, tanh_derivative, mse_loss, mse_loss_derivative,
    cross_entropy_loss, cross_entropy_loss_derivative
)

class Autoencoder:
    def __init__(self, layers, activation_fn_name, loss_fn_name, optimizer='gd', learning_rate=0.001, initial_lr=None, decay_rate=None, variable_lr=False):
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
        self.weights, self.biases = self.initialize_weights()
        self.optimizer_params = {}
        self.latent_layer_index = len(layers) // 2

    

    def initialize_weights(self):
        np.random.seed(42)
        weights = {}
        biases = {}
        for i in range(len(self.layers) - 1):
            input_dim = self.layers[i]
            output_dim = self.layers[i + 1]
            limit = np.sqrt(6 / (input_dim + output_dim))
            weights[f"W{i+1}"] = np.random.uniform(-limit, limit, (input_dim, output_dim))
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

    def train(self, X, epochs):
        losses = []
        for epoch in range(epochs):
            # Propagación hacia adelante
            activations, Z_values = self.forward_propagation(X)

            # Calcular la pérdida
            loss = self.loss_fn(X, activations[f"A{len(self.weights)}"])
            losses.append(loss)

            # Retropropagación
            grads = self.backward_propagation(X, activations, Z_values)

            # Actualizar parámetros
            self.update_parameters(grads, epoch)

            # Mostrar progreso
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Época {epoch + 1}/{epochs}, Pérdida: {loss:.6f}")

        # Después del entrenamiento, graficar la pérdida
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), losses, label='Pérdida de entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Curva de Pérdida durante el Entrenamiento')
        plt.legend()
        plt.show()

        return losses

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

