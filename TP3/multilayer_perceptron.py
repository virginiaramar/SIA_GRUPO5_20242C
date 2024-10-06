import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import seaborn as sns

class multilayer_perceptron:
    def __init__(self, config_file='config.json'):
        # Config data
        with open(config_file) as config_file:
            config = json.load(config_file)

        # Get the data from the config and put in variables
        self.problem_type = config['data']['problem_type']

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

        if self.problem_type == 'binary':
            self.y = self.y.reshape(-1, 1)
        elif self.problem_type == 'multiclass':
            if self.y.shape[1] != 10:
                num_classes = int(np.max(self.y)) + 1
                y_one_hot = np.zeros((self.y.size, num_classes))
                y_one_hot[np.arange(self.y.size), self.y.astype(int)] = 1
                self.y = y_one_hot
        else:
            raise ValueError("Invalid problem type. Choose 'binary' or 'multiclass'.")

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

        # Cross-validation parameters
        self.use_cross_validation = config['cross_validation']['use_cross_validation']
        self.k_folds = config['cross_validation']['k_folds']
        self.shuffle = config['cross_validation']['shuffle']
        self.random_seed = config['cross_validation']['random_seed']

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
        elif self.output_function == 'softmax':
            self.output_activation_function = lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
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

        # Add new attributes for results visualization
        self.error_history = []

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

    ##### ERROR FUNCTIONS AND DERIVATIVES #####
    def compute_loss(self, y_true, output):
        if self.problem_type == 'multiclass':
            return -np.mean(np.sum(y_true * np.log(output + 1e-8), axis=1))
        else:
            return 0.5 * np.mean((y_true - output) ** 2)

    def compute_loss_derivative(self, y_true, output):
        if self.problem_type == 'multiclass':
            return output - y_true
        else:
            return output - y_true

    ##### FOLD FOR CROSS VALIDATION #####
    def _create_folds(self):
        np.random.seed(self.random_seed)
        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)
        fold_sizes = np.full(self.k_folds, len(self.X) // self.k_folds)
        fold_sizes[:len(self.X) % self.k_folds] += 1
        current = 0
        self.folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            self.folds.append(indices[start:stop])
            current = stop

    ##### MULTILAYER ALGORITHM ##### 
    def multilayer_algorithm(self, X=None, y=None):
        if X is not None and y is not None:
            self.X = X
            self.y = y

        if self.use_cross_validation:
            return self._cross_validation_training()
        else:
            return self._normal_training()

    def _cross_validation_training(self):
        self._create_folds()
        accuracies = []
        for fold_index in range(self.k_folds):
            print(f"Fold {fold_index + 1}/{self.k_folds}")
            validation_indices = self.folds[fold_index]
            training_indices = np.hstack([self.folds[i] for i in range(self.k_folds) if i != fold_index])
            
            X_train, y_train = self.X[training_indices], self.y[training_indices]
            X_val, y_val = self.X[validation_indices], self.y[validation_indices]
            
            self._initialize_weights()
            
            for epoch in range(self.epochs):
                total_error = self._train_epoch(X_train, y_train)
                
                if total_error < self.error_threshold:
                    print(f"Convergence reached in epoch {epoch + 1}")
                    break
            
            accuracy = self._evaluate_fold(X_val, y_val)
            accuracies.append(accuracy)
        
        average_accuracy = np.mean(accuracies)
        print(f"Average Accuracy over {self.k_folds} folds: {average_accuracy * 100:.2f}%")
        return average_accuracy

    def _normal_training(self):
        for epoch in range(self.epochs):
            total_error = self._train_epoch(self.X, self.y)
            self.error_history.append(total_error)
            
            if total_error < self.error_threshold:
                print(f"Convergence reached in epoch {epoch + 1}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {total_error}")
        
        return self.error_history

    def _train_epoch(self, X, y):
        total_error = 0
        if self.mode == "batch":
            output = self._forward_prop(X)
            error = y - output
            delta_vec = self._back_prop(error)
            self._update_weights(delta_vec)
            total_error = self.compute_loss(y, output)
        elif self.mode == "mini-batch":
            total_error = 0
            total_samples = 0
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                batch_X = X[start:end]
                batch_y = y[start:end]
                
                output = self._forward_prop(batch_X)
                error = batch_y - output
                delta_vec = self._back_prop(error)
                self._update_weights(delta_vec)
                total_error += self.compute_loss(batch_y, output) * len(batch_X)
                total_samples += len(batch_X)
            total_error = total_error / total_samples
        elif self.mode == "online":
            for x, y_true in zip(X, y):
                x = x.reshape(1, -1)
                y_true = y_true.reshape(1, -1)
                output = self._forward_prop(x)
                error = y_true - output
                delta_vec = self._back_prop(error)
                self._update_weights(delta_vec)
                total_error += self.compute_loss(y_true, output)
            total_error /= len(X)
        else:
            raise ValueError("Invalid mode. Choose 'batch', 'mini-batch', 'online'.")
        
        return total_error

    def _evaluate_fold(self, X_val, y_val):
        correct_predictions = 0
        for x, y_true in zip(X_val, y_val):
            x = x.reshape(1, -1)
            output = self._forward_prop(x)
            if self.problem_type == 'binary':
                prediction = (output >= 0.5).astype(int)
                y_true = int(y_true)
                if prediction == y_true:
                    correct_predictions += 1
            elif self.problem_type == 'multiclass':
                prediction = np.argmax(output, axis=1)
                y_true_class = np.argmax(y_true)
                if prediction == y_true_class:
                    correct_predictions += 1
        accuracy = correct_predictions / len(X_val)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

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
    def _forward_prop(self, input):
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
            if i == len(self.weights) - 1 and self.output_function == 'softmax':
                # Usamos la derivada especial para softmax
                delta = self.compute_output_delta(self.activations[i + 1], error)
            else:
                activation_derivative = self._get_activation_derivative(i)
                derivative = activation_derivative(self.activations[i + 1])
                delta = error * derivative
            delta_vec.insert(0, delta)
            if i != 0:
                error = np.dot(delta, self.weights[i].T)
        return delta_vec

    def compute_output_delta(self, output, error):
        # Para softmax, el error ya viene calculado
        return error

    # def compute_output_delta(self, output, y_true):
    #     error = output - y_true
    #     softmax_deriv = self.softmax_derivative(output)
    #     delta = np.einsum('nij,nj->ni', softmax_deriv, error)
    #     return delta

    def softmax_derivative(self, y):
        n_samples, n_classes = y.shape
        derivative = np.zeros((n_samples, n_classes, n_classes))
        for n in range(n_samples):
            y_n = y[n].reshape(-1, 1)
            derivative[n] = np.diagflat(y_n) - np.dot(y_n, y_n.T)
        return derivative

    ##### UPDATE WEIGHTS ##### 
    def _update_weights(self, delta_vec):
        for i in range(len(self.weights)):
            layer = self.activations[i]
            delta = delta_vec[i]
            adjustment = self.learning_rate * np.dot(layer.T, delta)
            
            if adjustment.shape != self.weights[i].shape:
                raise ValueError(f"Shapes of weights {self.weights[i].shape} and adjustment {adjustment.shape} are incompatible.")

            # Optimizers
            if self.optimizer == 'gradient_descent':
                self.weights[i] += adjustment
                if self.adaptive_learning_rate:  # If LR iterative true
                    self._adjust_learning_rate(delta)
            elif self.optimizer == 'momentum':
                self.momentum_velocity[i] = self.alpha * self.momentum_velocity[i] + self.learning_rate * np.dot(layer.T, delta)
                if self.adaptive_learning_rate:
                    self._adjust_learning_rate(delta)
                self.weights[i] += self.momentum_velocity[i]
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
    def evaluate(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        correct_predictions = 0
        predictions = []
        
        for x, y_true in zip(X, y):
            x = np.array(x).reshape(1, -1)  # Asegurarse de que los datos estén en la forma correcta
            output = self._forward_prop(x)
            
            if self.problem_type == 'multiclass':
                # Para multiclase, obtener la clase con mayor probabilidad
                prediction = np.argmax(output, axis=1)  # La clase predicha por la red
                y_true_class = np.argmax(y_true)  # La clase verdadera
            else:
                # Para problemas binarios
                prediction = (output >= 0.5).astype(int)
                y_true_class = y_true
            
            predictions.append(prediction)
            if prediction == y_true_class:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(X)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy, predictions

    def predict(self, X):
        return self._forward_prop(X)

    #### VISUALIZATION METHODS ####
    def plot_error_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.error_history) + 1), self.error_history)
        plt.title('Error durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Error')
        plt.show()

    def plot_decision_boundary(self, X, y):
        # Only for 2D input
        if X.shape[1] != 2:
            print("Esta función solo funciona para datos de entrada bidimensionales.")
            return

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1) if self.problem_type == 'multiclass' else Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Frontera de decisión')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        if self.problem_type != 'multiclass':
            print("Esta función solo es aplicable para problemas multiclase.")
            return

        cm = np.zeros((self.architecture[-1], self.architecture[-1]), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()

    def visualize_weights(self, layer=0):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.weights[layer], annot=False, cmap='viridis')
        plt.title(f'Mapa de calor de los pesos - Capa {layer}')
        plt.xlabel('Neuronas de salida')
        plt.ylabel('Neuronas de entrada')
        plt.show()

# Fin de la clase multilayer_perceptron