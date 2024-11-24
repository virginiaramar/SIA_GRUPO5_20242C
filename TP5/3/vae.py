import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

activation_functions = {"relu": relu, "sigmoid": sigmoid}
activation_derivatives = {"relu": relu_derivative, "sigmoid": sigmoid_derivative}

class VAE:
    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers, activation_fn_name, optimizer="adam", learning_rate=0.001, initial_lr=None, decay_rate=None, variable_lr=False):
        self.input_dim = input_dim
        self.encoder_layers = [input_dim] + encoder_layers + [2 * latent_dim]
        self.latent_dim = latent_dim
        self.decoder_layers = [latent_dim] + decoder_layers + [input_dim]
        self.activation_fn = activation_functions[activation_fn_name]
        self.activation_derivative_fn = activation_derivatives[activation_fn_name]
        self.learning_rate = learning_rate
        self.variable_lr = variable_lr
        self.initial_lr = initial_lr if initial_lr else learning_rate
        self.decay_rate = decay_rate
        self.encoder_weights, self.encoder_biases = self.initialize_weights(self.encoder_layers)
        self.decoder_weights, self.decoder_biases = self.initialize_weights(self.decoder_layers)

    def initialize_weights(self, layers):
        weights, biases = {}, {}
        for i in range(len(layers) - 1):
            input_dim, output_dim = layers[i], layers[i + 1]
            limit = np.sqrt(6 / (input_dim + output_dim))
            weights[f"W{i+1}"] = np.random.uniform(-limit, limit, (input_dim, output_dim))
            biases[f"b{i+1}"] = np.zeros((1, output_dim))
        return weights, biases

    def sampling(self, mu, log_var):
        epsilon = np.random.normal(size=mu.shape)
        sigma = np.exp(0.5 * log_var)
        z = mu + sigma * epsilon
        return z

    def forward(self, X):
        # Encoder
        encoder_activations, encoder_Z_values = self.encoder_forward(X)
        mu_log_var = encoder_activations[f"A{len(self.encoder_layers)-1}"]
        mu = mu_log_var[:, :self.latent_dim]
        log_var = mu_log_var[:, self.latent_dim:]
        z = self.sampling(mu, log_var)

        # Decoder
        decoder_activations, decoder_Z_values = self.decoder_forward(z)
        reconstructed_X = decoder_activations[f"A{len(self.decoder_layers)-1}"]
        return mu, log_var, z, reconstructed_X, encoder_activations, encoder_Z_values, decoder_activations, decoder_Z_values

    def encoder_forward(self, X):
        activations, Z_values = {"A0": X}, {}
        for i in range(1, len(self.encoder_layers)):
            W, b = self.encoder_weights[f"W{i}"], self.encoder_biases[f"b{i}"]
            Z = np.dot(activations[f"A{i-1}"], W) + b
            if i == len(self.encoder_layers) - 1:
                activations[f"A{i}"] = Z  # Sin activación en la última capa
            else:
                activations[f"A{i}"] = self.activation_fn(Z)
            Z_values[f"Z{i}"] = Z
        return activations, Z_values

    def decoder_forward(self, z):
        activations, Z_values = {"A0": z}, {}
        for i in range(1, len(self.decoder_layers)):
            W, b = self.decoder_weights[f"W{i}"], self.decoder_biases[f"b{i}"]
            Z = np.dot(activations[f"A{i-1}"], W) + b
            if i == len(self.decoder_layers) - 1:
                activations[f"A{i}"] = sigmoid(Z)  # Función sigmoid en la última capa
            else:
                activations[f"A{i}"] = self.activation_fn(Z)
            Z_values[f"Z{i}"] = Z
        return activations, Z_values

    def compute_loss(self, X, reconstructed_X, mu, log_var):
        reconstruction_loss = np.mean(np.sum((X - reconstructed_X) ** 2, axis=1))
        kl_loss = -0.5 * np.mean(np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1))
        kl_weight = 5.0  # Peso ajustable
        total_loss = reconstruction_loss + kl_weight * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def backward(self, X, reconstructed_X, mu, log_var, z, encoder_activations, encoder_Z_values, decoder_activations, decoder_Z_values):
        batch_size = X.shape[0]
        kl_weight = 5.0  # Como se usa en el cálculo de la pérdida

        # Derivada de la pérdida total con respecto a reconstructed_X
        d_reconstructed_X = (1 / batch_size) * (-2) * (X - reconstructed_X)

        # Retropropagación a través del decodificador
        grads_decoder_W = {}
        grads_decoder_b = {}
        L = len(self.decoder_layers) - 1

        # Capa de salida del decodificador
        dZ = d_reconstructed_X * sigmoid_derivative(decoder_Z_values[f"Z{L}"])
        A_prev = decoder_activations[f"A{L-1}"]
        grads_decoder_W[f"dW{L}"] = np.dot(A_prev.T, dZ)
        grads_decoder_b[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True)

        for l in reversed(range(1, L)):
            activation_fn_derivative = self.activation_derivative_fn
            dA = np.dot(dZ, self.decoder_weights[f"W{l+1}"].T)
            dZ = dA * activation_fn_derivative(decoder_Z_values[f"Z{l}"])
            A_prev = decoder_activations[f"A{l-1}"]
            grads_decoder_W[f"dW{l}"] = np.dot(A_prev.T, dZ)
            grads_decoder_b[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

        d_z = np.dot(dZ, self.decoder_weights["W1"].T)

        # Retropropagación a través del muestreo
        epsilon = (z - mu) / np.exp(0.5 * log_var)
        sigma = np.exp(0.5 * log_var)
        dz_d_mu = 1
        dz_d_log_var = 0.5 * sigma * epsilon

        d_kl_d_mu = (1 / batch_size) * mu * kl_weight
        d_kl_d_log_var = (1 / batch_size) * 0.5 * (np.exp(log_var) - 1) * kl_weight

        dL_d_mu = d_z * dz_d_mu + d_kl_d_mu
        dL_d_log_var = d_z * dz_d_log_var + d_kl_d_log_var

        # Retropropagación a través del codificador
        grads_encoder_W = {}
        grads_encoder_b = {}
        dA = np.concatenate([dL_d_mu, dL_d_log_var], axis=1)
        L_enc = len(self.encoder_layers) - 1

        dZ = dA
        A_prev = encoder_activations[f"A{L_enc-1}"]
        grads_encoder_W[f"dW{L_enc}"] = np.dot(A_prev.T, dZ)
        grads_encoder_b[f"db{L_enc}"] = np.sum(dZ, axis=0, keepdims=True)

        for l in reversed(range(1, L_enc)):
            W_next = self.encoder_weights[f"W{l+1}"]
            dA = np.dot(dZ, W_next.T)
            activation_fn_derivative = self.activation_derivative_fn
            dZ = dA * activation_fn_derivative(encoder_Z_values[f"Z{l}"])
            A_prev = encoder_activations[f"A{l-1}"]
            grads_encoder_W[f"dW{l}"] = np.dot(A_prev.T, dZ)
            grads_encoder_b[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

        # Actualización de pesos y sesgos
        for l in range(1, L_enc+1):
            self.encoder_weights[f"W{l}"] -= self.learning_rate * grads_encoder_W[f"dW{l}"]
            self.encoder_biases[f"b{l}"] -= self.learning_rate * grads_encoder_b[f"db{l}"]

        for l in range(1, L+1):
            self.decoder_weights[f"W{l}"] -= self.learning_rate * grads_decoder_W[f"dW{l}"]
            self.decoder_biases[f"b{l}"] -= self.learning_rate * grads_decoder_b[f"db{l}"]

    def train(self, X, epochs):
        for epoch in range(epochs):
            mu, log_var, z, reconstructed_X, encoder_activations, encoder_Z_values, decoder_activations, decoder_Z_values = self.forward(X)
            loss, reconstruction_loss, kl_loss = self.compute_loss(X, reconstructed_X, mu, log_var)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Recon: {reconstruction_loss:.4f}, KL: {kl_loss:.4f}")
            self.backward(X, reconstructed_X, mu, log_var, z, encoder_activations, encoder_Z_values, decoder_activations, decoder_Z_values)


