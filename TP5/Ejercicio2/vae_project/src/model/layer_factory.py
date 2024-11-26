from .layers import Dense, ReLU, Sigmoid, Cosh

def create_activation(activation_name):
    """Factory para crear capas de activación"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'cosh': Cosh
    }
    if activation_name.lower() not in activations:
        raise ValueError(f"Activación no soportada: {activation_name}")
    return activations[activation_name.lower()]()

def create_dense_layer(input_size, output_size, config):
    """Factory para crear capas densas con configuración"""
    return Dense(
        input_size=input_size,
        output_size=output_size,
        initialization=config.get('initialization', {}).get('type', 'xavier'),
        bias=config.get('bias', 1)
    )