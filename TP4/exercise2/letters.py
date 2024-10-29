import numpy as np
import matplotlib.pyplot as plt
from HopfieldNetwork import HopfieldNetwork

patterns = {
    'J': np.array([
        [1, 1, 1, 1, 1],
        [-1, -1, -1, 1, -1],
        [-1, -1, -1, 1, -1],
        [1, -1, -1, 1, -1],
        [1, 1, 1, -1, -1]
    ]),
    'A': np.array([
        [-1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1]
    ]),
    'T': np.array([
        [1, 1, 1, 1, 1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1]
    ]),
    'F': np.array([
        [1, 1, 1, 1, 1],
        [1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1]
    ])
}

patterns_flat = [pattern.flatten() for pattern in patterns.values()]

hopfield_net = HopfieldNetwork()
hopfield_net.train(patterns_flat)

def add_noise(pattern, noise_level=0.15):
    """Add noise to a flattened pattern by flipping some bits."""
    noisy_pattern = pattern.copy()
    num_noisy_bits = int(noise_level * len(pattern))
    indices = np.random.choice(len(pattern), num_noisy_bits, replace=False)
    noisy_pattern[indices] = -noisy_pattern[indices]  # Flip bits
    return noisy_pattern

def display_multiple_steps(steps, title="Pattern Steps", square_size=100):
    """Display the different steps in a single drawing using subplots."""
    num_steps = len(steps)
    fig, axs = plt.subplots(1, num_steps, figsize=(num_steps * 3, 3), constrained_layout=True)

    for i, pattern in enumerate(steps):
        colors = np.where(pattern == 1, 'black', 'white')

        for row in range(5):
            for col in range(5):
                square = plt.Rectangle((col * square_size, (4 - row) * square_size),
                                       square_size, square_size,
                                       color=colors[row, col])
                axs[i].add_patch(square)

        axs[i].set_xlim(0, 5 * square_size)
        axs[i].set_ylim(0, 5 * square_size)
        axs[i].set_title(f"Step {i + 1}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.suptitle(title)
    plt.show()

for letter, pattern in patterns.items():
    print(f"\nProcessing letter '{letter}':")

    noisy_pattern = add_noise(pattern.flatten(), noise_level=0.1)

    steps = [noisy_pattern.reshape(5, 5)]

    predicted_pattern = noisy_pattern
    for _ in range(5):
        predicted_pattern = hopfield_net.predict([predicted_pattern], iterations=1, threshold=0, async_update=True)[0]
        steps.append(predicted_pattern.reshape(5, 5))

    display_multiple_steps(steps, title=f"Steps for '{letter}' Pattern")

