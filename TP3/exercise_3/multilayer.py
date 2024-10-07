import copy
import math
import sys
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import random

import json
from exercise_3.activation_functions import *




def update_delta_w(delta_w, delta_w_matrix):
    if delta_w is None:
        delta_w = delta_w_matrix
    else:
        delta_w += delta_w_matrix
    return delta_w


def convert_data(data_input, data_output):
    new_input = []
    new_output = []

    for i, o in zip(data_input, data_output):
        new_input.append(np.array(i))
        new_output.append(np.array(o))

    return np.array(new_input), np.array(new_output)


class MultiPerceptron:
    def __init__(self, config_path):
       
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        
        # Architecture
        self.entry_layer_amount = config["architecture"]["entry_layer_amount"]
        self.hidden_layer_amount = config["architecture"]["hidden_layer_neurons"]
        self.output_layer_amount = config["architecture"]["output_layer_amount"]
        
        # general parameters
        self.seed = config["seed"]
        self.learning_constant = config["learning_constant"]
        self.epoch = config["epoch"]
        self.epsilon = config["epsilon"]
        
        
        # activation functions
        self.hidden_function_name = config["activation_function"]["hidden_function"]
        self.output_function_name = config["activation_function"]["output_function"]
        self.beta = config["activation_function"]["beta"]

        # Weight initialization method
        self.initialization_w_method = config["initialization_weights"]["type"]

        # Optimizers
        self.optimization_method = config["optimization_method"]["type"]
        if self.optimization_method == "momentum":
            self.momentum_alpha = config["optimization_method"]["momentum"]["alpha"]
        elif self.optimization_method == "adam":
            self.adam_beta1 = config["optimization_method"]["adam"]["beta1"]
            self.adam_beta2 = config["optimization_method"]["adam"]["beta2"]
            self.adam_epsilon = config["optimization_method"]["adam"]["epsilon"]
            self.adam_alpha = config["optimization_method"]["adam"]["alpha"]
            
        self.batch_size = config["mode"]["batch_size"]
        
        # Graphs
        self.generate_error_graph = config["generate_error_graph"]
        self.print_final_values = config["print_final_values"]
        
        # Get activation functions to use in hidden layers
        self.hidden_activation_function, self.hidden_activation_derivative = get_activation_function(self.hidden_function_name, self.beta)
        
        # Get activation functions to use in output layer
        self.output_activation_function, self.output_activation_derivative = get_activation_function(self.output_function_name, self.beta)



        # an empty list to store the layers of the neural network
        self.layers: [NeuronLayer] = []

        # Vectorize the activation function to apply element-wise to arrays and fix the 'beta' parameter using 'partial'
        self.hidden_activation_function = np.vectorize(partial(self.hidden_activation_function, self.beta))
        self.output_activation_function = np.vectorize(partial(self.output_activation_function, self.beta))
        self.hidden_activation_derivative = np.vectorize(partial(self.hidden_activation_derivative, self.beta))
        self.output_activation_derivative = np.vectorize(partial(self.output_activation_derivative, self.beta))

        ### LAYERS ###

        # INPUT LAYER
        self.layers.append(
            NeuronLayer(self.entry_layer_amount, self.entry_layer_amount, self.hidden_activation_function, self.initialization_w_method))

        # Creamos la primera capa interna
        self.layers.append(
            NeuronLayer(self.entry_layer_amount, self.hidden_layer_amount[0], self.hidden_activation_function, self.initialization_w_method))

        # Creamos el resto de las capas interna
        for i in range(1, len(self.hidden_layer_amount)):
            self.layers.append(
                NeuronLayer(self.hidden_layer_amount[i-1], self.hidden_layer_amount[i], self.hidden_activation_function, self.initialization_w_method))

        # Creamos la ultima capa
        self.layers.append(
            NeuronLayer(self.hidden_layer_amount[-1], self.output_layer_amount, self.output_activation_function, self.initialization_w_method))
        
        self.input = None
        # Variables usadas en compute_error_parallel
        self.error_calc_items = None


    ### ALGORITM FLOW ###

    ## FORWARD PROPAGATION ##
    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current
    
    def update_all_weights(self, delta_w):  
        if self.optimization_method == "momentum":
            for idx, layer in enumerate(self.layers):
                layer.update_weights_momentum(delta_w[idx], self.momentum_alpha)
        elif self.optimization_method == "adam":
            for idx, layer in enumerate(self.layers):
                layer.update_weights_adam(delta_w[idx], self.adam_beta1, self.adam_beta2, self.adam_epsilon, self.adam_alpha)
        else:  # Gradient Descent
            for idx, layer in enumerate(self.layers):
                layer.update_weights_gradient_descent(delta_w[idx])


    """ Inicializa una lista vacía error_vector para almacenar los errores de cada muestra.
    Itera sobre los pares de datos de entrada (data_input) y salidas esperadas (expected_outputs):
    Para cada par (i, o):
    Realiza la propagación hacia adelante (forward_propagation) con la entrada i.
    Calcula el error cuadrático (o - output_result)^2 y lo añade a error_vector.
    Suma todos los errores cuadráticos:
    Inicializa total a 0.
    Para cada elemento en error_vector, suma todos sus componentes a total.
    Retorna 0.5 * total, que es la fórmula estándar para el MSE.
    """
    def compute_error(self, data_input, expected_outputs):

            error_vector = []

            for i, o in zip(data_input, expected_outputs):
                output_result = self.forward_propagation(i)
                error_vector.append(np.power(o - output_result, 2))

            total = 0
            for elem in error_vector:
                total += sum(elem)

            return 0.5 * total
    
    def back_propagation(self, expected_output, generated_output) -> list:
        delta_w = []

        # Calculamos el delta W de la capa de salida
        prev_delta = (expected_output - generated_output) * self.output_activation_derivative(
            self.layers[-1].excitement)
        delta_w.append(
            self.learning_constant * prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.hidden_activation_derivative(
                self.layers[idx].excitement)
            delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(
                self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        # Calculamos el delta W de la capa inicial
        delta = np.dot(prev_delta, self.layers[1].weights) * self.hidden_activation_derivative(
            self.layers[0].excitement)
        delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w
    



    def train(self, input_data, expected_output, collect_metrics):
        batch_size = self.batch_size
        size = len(input_data)
        if size < batch_size:
            raise ValueError("Batch size is greater than size of input.")

        i = 0
        error = None
        w_min = None
        min_error = float("inf")
        metrics = {}
        self.initialize_metrics(metrics)

        # Convertimos los datos de entrada a Numpy Array (asi no lo tenemos que hacer mientras procesamos)
        converted_input, converted_output = convert_data(input_data, expected_output)

        while min_error > self.epsilon and i < self.epoch:

            delta_w = None 

            # usamos todos los datos
            if batch_size == size:
                for i, o in zip(converted_input, converted_output):

                    result = self.forward_propagation(i)
                    delta_w_matrix = self.back_propagation(o, result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # usamos un subconjunto
            else:
                for _ in range(batch_size):
                    number = random.randint(0, size - 1)

                    result = self.forward_propagation(converted_input[number])
                    delta_w_matrix = self.back_propagation(converted_output[number], result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # Actualizamos los pesos
            self.update_all_weights(delta_w)

            error = self.compute_error(converted_input, converted_output)

            if error < min_error:
                min_error = error
                w_min = self.get_weights()
            i += 1
            collect_metrics(metrics, error, i)

        return error, w_min, metrics
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(copy.deepcopy(layer.weights))
        return weights

    
    def test(self, input_test_data, expected_output):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for input_data, outputs in zip(input_test_data, expected_output):
            results = self.forward_propagation(input_data)
            for result, expected_output in zip(results, outputs):
                if expected_output == 1:
                    if math.fabs(expected_output - result) < self.epsilon:
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if math.fabs(expected_output - result) < self.epsilon:
                        true_negative += 1
                    else:
                        false_positive += 1

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive+false_negative)
        f1_score = None
        if precision + recall != 0:
            f1_score = (2 * precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1_score

    @staticmethod
    def initialize_metrics(metrics):
        metrics["error"] = []
        metrics["iteration"] = 0   

class NeuronLayer:
    def __init__(self, previous_layer_neuron_amount, current_layer_neurons_amount, activation_function, weight_initialization):
        
        self.excitement = None
        self.output = None
        self.activation_function = activation_function
        self.prev_delta = 0
        


        # Initialize the weights
        self.weights = []
        for i in range(current_layer_neurons_amount):
            self.weights.append([])
            for j in range(previous_layer_neuron_amount):
                self.weights[i].append(self.initialize_weight(previous_layer_neuron_amount, current_layer_neurons_amount, weight_initialization))
        self.weights = np.array(self.weights)

        self.m = np.zeros_like(self.weights)  # Para Adam
        self.v = np.zeros_like(self.weights)  # Para Adam
        self.t = 0  # Contador de pasos para Adam

    def initialize_weight(self, input_size, output_size, method):
        
        # Random between -0.01 y 0.01
        if method == 'random':
            return 2 * np.random.rand() - 1

        # All zero initialization
        elif method == 'zero':
            return 0

        # Normla distribution
        elif method == 'normal':
            return np.random.normal(0, 1)

        # xavier
        elif method == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            return np.random.uniform(-limit, limit)

        # he
        elif method == 'he':
            limit = np.sqrt(2 / input_size)
            return np.random.normal(0, limit)

        else:
            raise ValueError(f"Unknown weight initialization method: {method}")


    def compute_activation(self, prev_input):
        self.excitement = np.dot(self.weights, prev_input)
        # Activation function application
        print(self.excitement)
        self.output = self.activation_function(self.excitement)
        return self.output  

    def update_weights_momentum(self, delta_w, alpha):
       new_delta = delta_w + alpha * self.prev_delta
       self.weights += new_delta
       self.prev_delta = new_delta
    
    def update_weights_adams(self, delta_w, beta1, beta2,epsilon,alpha):
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * delta_w
        self.v = beta2 * self.v + (1 - beta2) * (delta_w ** 2)
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        self.weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    def update_weights_gradient_descent(self, delta_w):
        self.weights -= delta_w

    