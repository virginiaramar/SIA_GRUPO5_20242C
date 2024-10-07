import numpy as np
import math

# Avoid the overflow and excessive calculations
exp_overflow = math.floor(math.log(np.finfo(np.float64).max) / -2) + 2  
tanh_overflow = math.floor(math.log(np.finfo(np.float64).max) / 1) 
range_values_sigmoid = math.log(1 / 0.999 - 1) 
range_values_tanh = np.arctanh(0.999)

def sigmoid(x, beta):
    if x < 0 and x * beta < exp_overflow:
        return 0  
    
    if x > range_values_sigmoid / (-2*beta):
        return 0.999
    elif x < range_values_sigmoid / (2*beta):
        return 0.001

    return 1 / (1 + np.exp(-beta * x))

def sigmoid_derivative(x, beta):
    sigmoid_result = sigmoid(x, beta)
    return 2 * beta * sigmoid_result * (1 - sigmoid_result)

def tanh(x, beta):
    if x < 0 and x * beta < tanh_overflow:
        return -1  
    elif x > tanh_overflow:
        return 1  
    
    if x > range_values_tanh / beta:
        return 0.999
    elif x < -range_values_tanh / beta:
        return -0.999

    return np.tanh(beta * x)

def tanh_derivative(x, beta):
    tanh_result = tanh(x, beta)
    return beta * (1 - tanh_result ** 2)

def relu(x, beta):
    return np.maximum(0, x)

def relu_derivative(x, beta):
    return np.where(x > 0, 1, 0)

def softmax(x, beta):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def softmax_derivative(x, beta):
    s = softmax(x, beta)  
    
    jacobian = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                if j == k:
                    jacobian[i, j, k] = s[i, j] * (1 - s[i, j])  
                else:
                    jacobian[i, j, k] = -s[i, j] * s[i, k]  
    return jacobian

def get_activation_function(name, beta):
    if name == 'sigmoid':
        return lambda x: sigmoid(x, beta), lambda x: sigmoid_derivative(x, beta)
    elif name == 'tanh':
        return lambda x: tanh(x, beta), lambda x: tanh_derivative(x, beta)
    elif name == 'relu':
        return lambda x: relu(x, beta), lambda x: relu_derivative(x, beta)
    elif name == 'softmax':
        return lambda x: softmax(x, beta), lambda x: softmax_derivative(x, beta)
    else:
        raise ValueError(f"Invalid activation function: {name}. Choose 'sigmoid', 'tanh', 'relu' or 'softmax'.")