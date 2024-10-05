import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

class multilayer_perceptron:
    def __init__(self, config_file='config.json'):
        # Config data
        with open(config_file) as config_file:
            config = json.load(config_file)

      # Get the data from the config and put in variables
        input_source = config['data']['input']
        if isinstance(input_source, str) and input_source.endswith('.txt') and os.path.isfile(input_source):
            # If it is a txt, read it
            self.X = np.genfromtxt(input_source, delimiter=' ')
        else:
            self.X = np.array(input_source)

        # output
        output_source = config['data']['output']
        
        if isinstance(output_source, str) and output_source.endswith('.txt') and os.path.isfile(output_source):
            self.y = np.genfromtxt(output_source).astype(float)
            
        else:
            self.y = np.array(output_source)

        self.y = self.y.reshape(-1, 1)


            

        
        self.architecture = config['initial_parameters']['architecture']  # layers
        self.learning_rate = config['initial_parameters']['learning_rate']  # LR
        self.epochs = config['initial_parameters']['epochs']  # Epochs
        self.mode = config['initial_parameters']['mode']  # Mode
        self.batch_size = config['initial_parameters']['minibatch_size']  # batch size
        self.error_threshold = config['error']['threshold']  # stop criteria: error threshold
        self.weight_initialization = config['weights']['initialization']  # weight initialization method
        self.activation_function = config['activation_function']['function']  # activation function
        self.output_function = config['activation_function']['output_function']  # use of softmax or sigmoid in the last layer
        self.beta = config['activation_function']['beta']
        
        self.adaptive_learning_rate = config['optimizer']['adaptive_learning_rate']
        self.lr_adjustment_value = config['optimizer']['lr_adjustment_value']
        self.optimizer = config['optimizer']['type']



        # Activation function for hidden layers
        if self.activation_function == 'sigmoid':
            self.hidden_activation_function = lambda x: 1 / (1 + np.exp(-self.beta * x))
            self.hidden_activation_derivative = lambda x: self.beta * x * (1 - x)
        elif self.activation_function == 'tanh':
            self.hidden_activation_function = lambda x: np.tanh(self.beta * x)
            self.hidden_activation_derivative = lambda x: self.beta * (1 - x ** 2)
        elif self.activation_function == 'relu':
            self.hidden_activation_function = lambda x: np.maximum(0, x)
            self.hidden_activation_derivative = lambda x: np.where(x > 0, 1, 0)
        else:
            raise ValueError("Invalid activation function. Choose 'sigmoid', 'tanh', or 'relu'.")

        # Activation function for output layer
        if self.output_function == 'sigmoid':
            self.output_activation_function = lambda x: 1 / (1 + np.exp(-self.beta * x))
            self.output_activation_derivative = lambda x: self.beta * x * (1 - x)
            # Sigmoid for binary
        elif self.output_function == 'softmax':
            self.output_activation_function = lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
            # Softmax for multiclass
        else:
            raise ValueError("Invalid output activation function. Choose 'sigmoid' or 'softmax'.")

        # If we use momentum
        if self.optimizer == 'momentum':
            self.alpha = config['optimizer']['momentum']['alpha']
        
        # If we use Adam
        elif self.optimizer == 'adam':
            self.beta1 = config['optimizer']['adam']['beta1']
            self.beta2 = config['optimizer']['adam']['beta2']
            self.epsilon = config['optimizer']['adam']['epsilon']
            self.alpha = config['optimizer']['adam']['alpha']  
            

        # weights matrix initialize
        self.weights = []  
        self.biases = []

        # In all the layers in the first epoch
        self._initialize_weights()


        # Optimizer initiators
        self.momentum_velocity = [np.zeros_like(w) for w in self.weights]
        self.adam_m = [np.zeros_like(w) for w in self.weights]
        self.adam_v = [np.zeros_like(w) for w in self.weights]
        self.timestep = 1
        


    
    ##### INITIALIZE WEIGHTS #####   

    def _initialize_weights(self):
        np.random.seed(1)  # Fijar semilla para obtener números aleatorios reproducibles

        self.weights = []  
        self.biases = []

        for i in range(len(self.architecture) - 1):
            input_size = self.architecture[i] 
            output_size = self.architecture[i + 1]

            if self.weight_initialization == 'random':
                # Random weights between -0.01 and 0.01
                weight_matrix = 2 * np.random.rand(input_size, output_size) - 1

            elif self.weight_initialization == 'zero':
                # Initialize all weights to zero
                weight_matrix = np.zeros((input_size, output_size))

            elif self.weight_initialization == 'normal':
                # Normal distribution initialization
                weight_matrix = np.random.normal(0, 1, (input_size, output_size))

            elif self.weight_initialization == 'xavier':
                # Xavier initialization (also known as Glorot initialization)
                limit = np.sqrt(6 / (input_size + output_size))
                weight_matrix = np.random.uniform(-limit, limit, (input_size, output_size))

            elif self.weight_initialization == 'he':
                # He initialization 
                limit = np.sqrt(2 / input_size)
                weight_matrix = np.random.normal(0, limit, (input_size, output_size))


            else:
                raise ValueError("Invalid weight initialization method. Choose 'random', 'zero', 'normal', 'xavier', or 'he'.")

            self.weights.append(weight_matrix)
            self.biases.append(np.ones((1, output_size)))




    ##### MULTILAYER ALGORITHM ##### 
    
    def multilayer_algorithm(self):
        for epoch in range(self.epochs):
            total_error = 0  # Error in that epoch

            # Mode Batch: actualization after calculating deltaw for all elements in the data
            if self.mode == "batch":
                output = self._forward_prop(self.X)
                
                error =  self.y - output

                delta_vec = self._back_prop(error)

                # Actualización de pesos después de procesar todo el conjunto de datos
                self._update_weights(delta_vec)  # Actualiza los pesos con los delta w acumulados
                total_error = 0.5 * np.sum(error ** 2)

            # Mode Mini-batch: small ranges of data
            elif self.mode == "mini-batch":
                total_error = 0
                total_samples = 0
                for start in range(0, len(self.X), self.batch_size):
                    end = min(start + self.batch_size, len(self.X))
                    batch_X = self.X[start:end]
                    batch_y = self.y[start:end]
                    
                    output = self._forward_prop(batch_X)

                    error = batch_y - output
                    delta_vec = self._back_prop(error)

                    self._update_weights(delta_vec)
                    total_error += 0.5 * np.sum(error ** 2)
                    total_samples += len(batch_X)
                total_error = total_error / total_samples
                
                

            # Modo Online: after each element
            elif self.mode == "online":
                for x, y in zip(self.X, self.y):
                    output = self._forward_prop(x.reshape(1, -1))
                    error = y - output
                    delta_vec = self._back_prop(error)
                    self._update_weights(delta_vec)
                    total_error += 0.5 * np.sum(error ** 2)

            else:
                raise ValueError("Invalid mode. Choose 'batch', 'mini-batch', 'online'.")

            # Convergence criteria
            if total_error < self.error_threshold:
                print(f"Convergence reached in epoch {epoch + 1}")
                break





    ##### ACTIVATION FUNCTIONS AND DERIVATIVES ##### 

    def _get_activation_function(self, layer_index):
        if layer_index == len(self.weights) - 1:
            # output layer
            return self.output_activation_function
        else:
            # Hidden layers
            return self.hidden_activation_function

    def _get_activation_derivative(self, layer_index):
        if layer_index == len(self.weights) - 1:
            # output layer
            return self.output_activation_derivative
        else:
            # Hidden layers
            return self.hidden_activation_derivative






    ##### FORWARD PROPAGATION ##### 

    def _forward_prop(self,input):
        
        self.activations = [input]  # The first one are the inputs
        

        for i in range(len(self.weights)): # Iterations for all the layers of the multilayer
            
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]  # Sumatoria and got product between x and w for every neuron of th next layer
            
            activation_function = self._get_activation_function(i)
            output = activation_function(z)

            self.activations.append(output)  # Save the output of this layer with the previous ones
        return self.activations[-1] 
    



    ##### BACKWARD PROPAGATION ##### 
    
    def _back_prop(self, error):
        delta_vec = []
        for i in reversed(range(len(self.weights))):
            activation_derivative = self._get_activation_derivative(i)
            derivative = activation_derivative(self.activations[i + 1])
            
            delta = error * derivative
            delta_vec.insert(0, delta)
            if i != 0:
                error = np.dot(delta, self.weights[i].T)
                
        return delta_vec





    ##### UPDATE WEIGHTS ##### 

    def _update_weights(self, delta_vec):
        for i in range(len(self.weights)):
            
            layer = self.activations[i]

            # Delta of the current layer calculated in backprop
            delta = delta_vec[i]
            #print(f"Layer {i}: {layer.shape}, Delta {i}: {delta.shape}, Weights {i}: {self.weights[i].shape}")
        
            adjustment = self.learning_rate * np.dot(layer.T, delta)
            #print(f"Adjustment {i}: {adjustment.shape}, Weights {i}: {self.weights[i].shape}")
        
            if adjustment.shape != self.weights[i].shape:
                raise ValueError(f"Shapes of weights {self.weights[i].shape} and adjustment {adjustment.shape} are incompatible.")


            # Optimizers
               
                # Basic methods, moves the weights in direction of negative gradient
            if self.optimizer == 'gradient_descent':
                self.weights[i] += adjustment
                if self.adaptive_learning_rate:  # If LR iterative true
                    self._adjust_learning_rate(delta)
                
                
                # Allows the same direction for several iteractions
            elif self.optimizer == 'momentum':
                self.momentum_velocity[i] = self.alpha * self.momentum_velocity[i] + self.learning_rate * np.dot(layer.T, delta)
                if self.adaptive_learning_rate:
                    self._adjust_learning_rate(delta)
                self.weights[i] += self.momentum_velocity[i]
                
                # Use promedio and variance of gradient to adjust adaptive the weights
            elif self.optimizer == 'adam':
                self.adam_m[i] = self.beta1 * self.adam_m[i] + (1 - self.beta1) * np.dot(layer.T, delta)
                self.adam_v[i] = self.beta2 * self.adam_v[i] + (1 - self.beta2) * (np.dot(layer.T, delta) ** 2)
                m_hat = self.adam_m[i] / (1 - self.beta1 ** self.timestep)
                v_hat = self.adam_v[i] / (1 - self.beta2 ** self.timestep)
                self.weights[i] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


        if self.optimizer == 'adam':
            self.timestep += 1  # increment timestep adam

    def _adjust_learning_rate(self, delta):
        if np.mean(delta) < 0:
            self.learning_rate += self.lr_adjustment_value
        elif np.mean(delta) > 0:
            self.learning_rate -= self.lr_adjustment_value * self.learning_rate


    #### EVALUATE THE OUTPUT ####

    def evaluate(self):
        correct_predictions = 0
        for x, y_true in zip(self.X, self.y):
            x = np.array(x).reshape(1, -1)  
            output = self._forward_prop(x)
            prediction = np.round(output)
            if prediction == y_true:
                correct_predictions += 1
        print(f"Correct predictions: {correct_predictions} out of {len(self.X)}")




 