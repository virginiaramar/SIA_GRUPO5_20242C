import numpy as np

from .layers import Dense, ReLU, Sigmoid, Sampling


class VAE:
    def __init__(self, input_dim, latent_dim, encoder_layers=None, decoder_layers=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if encoder_layers is None:
            encoder_layers = [
                {"units": 256, "activation": "relu"},
                {"units": 128, "activation": "relu"},
                {"units": 64, "activation": "relu"}
            ]

        if decoder_layers is None:
            decoder_layers = [
                {"units": 64, "activation": "relu"},
                {"units": 128, "activation": "relu"},
                {"units": 256, "activation": "relu"}
            ]

        self.encoder = self._build_encoder(encoder_layers)
        self.decoder = self._build_decoder(decoder_layers)

    def _build_encoder(self, layer_configs):
        layers = []
        input_size = self.input_dim

        for config in layer_configs:
            layers.append(Dense(input_size, config["units"]))
            if config["activation"] == "relu":
                layers.append(ReLU())
            input_size = config["units"]

        # Capa final para mean y log_var
        layers.append(Dense(input_size, self.latent_dim * 2))
        layers.append(Sampling(self.latent_dim))

        return layers

    def _build_decoder(self, layer_configs):
        layers = []
        input_size = self.latent_dim

        for config in layer_configs:
            layers.append(Dense(input_size, config["units"]))
            if config["activation"] == "relu":
                layers.append(ReLU())
            input_size = config["units"]

        # Capa final para reconstrucción
        layers.append(Dense(input_size, self.input_dim))
        layers.append(Sigmoid())

        return layers

    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer.forward(h)
        return h

    def decode(self, z):
        h = z
        for layer in self.decoder:
            h = layer.forward(h)
        return h

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def compute_loss(self, x, x_recon, z_mean, z_log_var):
        # Pérdida de reconstrucción (binary cross entropy)
        epsilon = 1e-7
        recon_loss = -np.sum(
            x * np.log(np.clip(x_recon, epsilon, 1.0)) +
            (1 - x) * np.log(np.clip(1 - x_recon, epsilon, 1.0))
        )

        # Pérdida KL
        kl_loss = -0.5 * np.sum(
            1 + np.clip(z_log_var, -10, 10) -
            np.square(z_mean) -
            np.exp(np.clip(z_log_var, -10, 10))
        )

        return recon_loss + kl_loss

    def train_step(self, x, learning_rate, beta=0):
        # Encode
        z_params = x
        for layer in self.encoder[:-1]:
            z_params = layer.forward(z_params)

        z_mean = z_params[:, :self.latent_dim]
        z_log_var = z_params[:, self.latent_dim:]
        z = self.encoder[-1].forward(z_params)

        # Decode
        x_recon = self.decode(z)

        # Compute loss
        recon_loss = -np.sum(
            x * np.log(x_recon + 1e-10) + (1 - x) * np.log(1 - x_recon + 1e-10)
        )
        kl_loss = -0.5 * np.sum(1 + z_log_var - z_mean ** 2 - np.exp(z_log_var))
        print("kl_loss", kl_loss)
        loss = recon_loss + beta * kl_loss

        # Backpropagation
        grad = (x_recon - x) / (x_recon * (1 - x_recon) + 1e-10)
        for layer in reversed(self.decoder):
            grad = layer.backward(grad, learning_rate)
        grad = self.encoder[-1].backward(grad, learning_rate)
        for layer in reversed(self.encoder[:-1]):
            grad = layer.backward(grad, learning_rate)

        return loss

    def train(self, X, epochs, batch_size=32, learning_rate=0.001):
        n_samples = len(X)
        history = []

        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(X)

            # Manejar el caso de pocos datos
            if n_samples < batch_size:
                batch_size = n_samples

            n_batches = max(n_samples // batch_size, 1)

            for i in range(0, n_samples, batch_size):
                batch = X[i:min(i + batch_size, n_samples)]
                loss = self.train_step(batch, learning_rate)
                epoch_loss += loss

            epoch_loss /= n_batches
            history.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        return history

    def generate(self, n_samples=1):
        z = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.decode(z)