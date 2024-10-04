import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

class multilayer_perceptron:
    def __init__(self, config_file='config.json'):
        # Config data
        with open(config_file) as config_file:
            config = json.load(config_file)

      # Get the data from the config and put in variables
        self.X = np.array(config['data']['input'])  # input
        self.y = np.array(config['data']['output'])  # output
        self.architecture = config['initial_parameters']['architecture']  # layers
        self.learning_rate = config['initial_parameters']['learning_rate']  # LR
        self.epochs = config['initial_parameters']['epochs']  # Epochs
        self.mode = config['initial_parameters']['mode']  # Mode
        self.batch_size = config['initial_parameters']['minibatch_size']  # batch size
        self.error_threshold = config['error']['threshold']  # stop criteria: error threshold
        self.weight_initialization = config['weights']['initialization']  # weight initialization method
        self.activation_function = config['activation_function']['function']  # activation function
        self.use_softmax = config['activation_function']['use_softmax']  # use of softmax in the last layer
        self.beta = config['activation_function']['beta']
        
        self.adaptive_learning_rate = config['optimizer']['adaptive_learning_rate']
        self.lr_adjustment_value = config['optimizer']['lr_adjustment_value']
        self.optimizer = config['optimizer']['type']
        # If we use momentum
        if self.optimizer == 'momentum':
            self.alpha = config['optimizer']['momentum']['alpha']
        
        # If we use Adam
        elif self.optimizer == 'adam':
            self.beta1 = config['optimizer']['adam']['beta1']
            self.beta2 = config['optimizer']['adam']['beta2']
            self.epsilon = config['optimizer']['adam']['epsilon']
            self.alpha = config['optimizer']['adam']['alpha']  
            


        # Optimizer initiators
        self.momentum_velocity = [np.zeros_like(w) for w in self.weights]
        self.adam_m = [np.zeros_like(w) for w in self.weights]
        self.adam_v = [np.zeros_like(w) for w in self.weights]
        self.timestep = 1
        

        # weights matrix initialize
        self.weights = []  

        # In all the layers in the first epoch
        self._initialize_weights()



    
    ##### INITIALIZE WEIGHTS #####   

    def _initialize_weights(self):
        np.random.seed(1)  # Fijar semilla para obtener números aleatorios reproducibles

        for i in range(len(self.architecture) - 1):
            if self.weight_initialization == 'random':
                # Random weights between -0.01 and 0.01
                weight_matrix = 2 * np.random.rand(self.architecture[i], self.architecture[i + 1]) - 1

            elif self.weight_initialization == 'zero':
                # Initialize all weights to zero
                weight_matrix = np.zeros((self.architecture[i], self.architecture[i + 1]))

            elif self.weight_initialization == 'normal':
                # Normal distribution initialization
                weight_matrix = np.random.normal(0, 1, (self.architecture[i], self.architecture[i + 1]))

            elif self.weight_initialization == 'xavier':
                # Xavier initialization (also known as Glorot initialization)
                limit = np.sqrt(6 / (self.architecture[i] + self.architecture[i + 1]))
                weight_matrix = np.random.uniform(-limit, limit, (self.architecture[i], self.architecture[i + 1]))

            elif self.weight_initialization == 'he':
                # He initialization
                limit = np.sqrt(2 / self.architecture[i])  # He initialization formula
                weight_matrix = np.random.normal(0, limit, (self.architecture[i], self.architecture[i + 1]))


            else:
                raise ValueError("Invalid weight initialization method. Choose 'random', 'zero', 'normal', 'xavier', or 'he'.")

            self.weights.append(weight_matrix)
            self.biases.append(np.ones((1, self.architecture[i + 1])))




    ##### MULTILAYER ALGORITHM ##### 
    
    def multilayer_algorithm(self):
        for epoch in range(self.epochs):
            total_error = 0  # Error in that epoch

            # Mode Batch: actualization after calculating deltaw for all elements in the data
            if self.mode == "batch":
                output = np.array([self._forward_prop(x) for x in self.X])
                error = self.y - output
                delta_vec = self._back_prop(error)
                self._update_weights(delta_vec)
                total_error = np.mean(np.abs(error))

            # Mode Mini-batch: small ranges of data
            elif self.mode == "mini-batch":
                for start in range(0, len(self.X), self.batch_size):
                    end = start + self.batch_size
                    batch_X = self.X[start:end]
                    batch_y = self.y[start:end]
                    output = np.array([self._forward_prop(x) for x in batch_X])
                    error = batch_y - output
                    delta_vec = self._back_prop(error)
                    self._update_weights(delta_vec)
                    total_error += np.mean(np.abs(error))

            # Modo Online: after each element
            elif self.mode == "online":
                for x, y in zip(self.X, self.y):
                    output = self._forward_prop(x)
                    error = y - output
                    delta_vec = self._back_prop(error)
                    self._update_weights(delta_vec)
                    total_error += np.mean(np.abs(error))

            # Convergence criteria
            if total_error < self.error_threshold:
                print(f"Convergence reached in epoch {epoch + 1}")
                break





    ##### ACTIVATION FUNCTIONS AND DERIVATIVES ##### 

    def activation(self, x, layer_index):
        if layer_index == len(self.weights) - 1 and self.use_softmax:  
            exp_x = np.exp(x - np.max(x))  
            return exp_x / exp_x.sum(axis=0, keepdims=True)
        
        
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-self.beta * x))  
        elif self.activation_function == 'tanh':
            return np.tanh(self.beta * x)  
        elif self.activation_function == 'relu':
            return np.maximum(0, self.beta * x)  
        else:
            raise ValueError("Invalid activation function. Choose 'sigmoid', 'tanh', or 'relu'.")
        
    def _activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.beta * x * (1 - x)
        elif self.activation_function == 'tanh':
            return self.beta * (1 - x ** 2)
        elif self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError("Invalid activation function for derivative. Choose 'sigmoid', 'tanh', or 'relu'.")





    ##### FORWARD PROPAGATION ##### 

    def _forward_prop(self,input):
        input = np.append(input, 1) # Add bias to input
        self.activations = [input]  # The first one are the inputs
        for i in range(len(self.weights)): # Iterations for all the layers of the multilayer
            z = np.dot(self.activations[i], self.weights[i])  # Sumatoria and got product between x and w for every neuron of th next layer
            
            # This process every neuron of the layer in parallel, makes the code faster
            with ThreadPoolExecutor(max_workers=self.weights[i].shape[1]) as executor:
                output = list(executor.map(lambda x: self.activation(x, i), z.T))
                output = np.array(output).T  # Convert again to array

            # Add an extra columna with value 1, if all the values are 0, the neuron will not be 0, more flexible
            if i < len(self.weights) - 1:
                output = np.append(output, 1)

            self.activations.append(output)  # Save the output of this layer with the previous ones
        return self.activations[-1] 
    



    ##### BACKWARD PROPAGATION ##### 
    
    def _back_prop(self, error):
        # Keeps the delta w for the output, then for all the layers
        delta_vec = [error * self._activation_derivative(self.activations[-1])]
        
        # Hidden layers backpropagation
        for i in range(len(self.weights) - 2, -1, -1):
            delta = delta_vec[-1].dot(self.weights[i + 1][1:].T) * self._activation_derivative(self.activations[i + 1][1:])
            delta_vec.append(delta)
        delta_vec.reverse()  # Invert the error for starting in the first layers

        return delta_vec




    ##### UPDATE WEIGHTS ##### 

    def _update_weights(self, delta_vec):
        for i in range(len(self.weights)):
            # sesgo, adds 1 for flexibility, 2D conversion for posterior multiplication
            layer = np.insert(self.activations[i], 0, 1, axis=0).reshape(1, -1)
            # Delta of the current layer calculated in backprop
            delta = delta_vec[i].reshape(1, -1)

            # Optimizers
               
                # Basic methods, moves the weights in direction of negative gradient
            if self.optimizer == 'gradient_descent':
                adjustment = self.learning_rate * layer.T.dot(delta)
                if self.adaptive_learning_rate:  # If LR iterative true
                    self._adjust_learning_rate(delta)
                self.weights[i] += adjustment
                
                # Allows the same direction for several iteractions
            elif self.optimizer == 'momentum':
                self.momentum_velocity[i] = self.alpha * self.momentum_velocity[i] + self.learning_rate * layer.T.dot(delta)
                if self.adaptive_learning_rate:
                    self._adjust_learning_rate(delta)
                self.weights[i] += self.momentum_velocity[i]
                
                # Use promedio and variance of gradient to adjust adaptive the weights
            elif self.optimizer == 'adam':
                self.adam_m[i] = self.beta1 * self.adam_m[i] + (1 - self.beta1) * layer.T.dot(delta)
                self.adam_v[i] = self.beta2 * self.adam_v[i] + (1 - self.beta2) * (layer.T.dot(delta) ** 2)
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