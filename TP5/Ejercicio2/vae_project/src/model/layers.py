import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, initialization="xavier", bias=1):
        super().__init__()
        
        # Xavier/Glorot initialization
        if initialization == "xavier":
            limit = np.sqrt(2.0 / float(input_size + output_size))
            self.weights = np.random.normal(0.0, limit, (input_size, output_size))
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01
            
        self.bias = np.ones((1, output_size)) * bias
        
        # Momentum parameters
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
        self.m_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        
        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        self.t += 1
        
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Adam optimization
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradient
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * np.square(weights_gradient)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * np.square(bias_gradient)
        
        m_w_hat = self.m_w / (1 - self.beta1**self.t)
        m_b_hat = self.m_b / (1 - self.beta1**self.t)
        v_w_hat = self.v_w / (1 - self.beta2**self.t)
        v_b_hat = self.v_b / (1 - self.beta2**self.t)
        
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return input_gradient

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Cosh(Layer):
    def forward(self, input):
        self.input = input
        return np.cosh(input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * np.sinh(self.input)

class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-np.clip(input, -88.0, 88.0)))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.output * (1 - self.output)

class Sampling(Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = None
        self.log_var = None
        self.std = None
        self.epsilon = None

    def forward(self, input):
        self.mean = input[:, :self.latent_dim]
        self.log_var = np.clip(input[:, self.latent_dim:], -10, 10)
        self.std = np.exp(0.5 * self.log_var)
        self.epsilon = np.random.normal(0, 1, self.mean.shape)
        return self.mean + self.std * self.epsilon

    def backward(self, output_gradient, learning_rate):
        d_mean = output_gradient
        d_std = output_gradient * self.epsilon
        d_log_var = d_std * self.std * 0.5 * np.exp(-0.5 * self.log_var)
        return np.concatenate([d_mean, d_log_var], axis=1)