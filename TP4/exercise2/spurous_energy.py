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
    'B': np.array([
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1]
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

def add_noise(pattern, noise_level=0.3):
    """Add noise to a flattened pattern by flipping some bits."""
    noisy_pattern = pattern.copy()
    num_noisy_bits = int(noise_level * len(pattern))
    indices = np.random.choice(len(pattern), num_noisy_bits, replace=False)
    noisy_pattern[indices] = -noisy_pattern[indices]
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

random_pattern = np.array([
        [-1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1],
        [1, -1, -1, -1, 1],
        [-1, -1, 1, 1, -1],
        [-1, -1, -1, 1, 1]
    ])
pattern_a = random_pattern.flatten()
noisy_pattern_a = add_noise(pattern_a, noise_level=0)

steps = [noisy_pattern_a.reshape(5, 5)]
energies = []
predicted_pattern = noisy_pattern_a

for _ in range(6):
    energy = hopfield_net._compute_energy(predicted_pattern)
    energies.append(energy)

    indices = np.random.choice(len(predicted_pattern), size=len(predicted_pattern) // 2, replace=False)
    for index in indices:
        predicted_pattern[index] = hopfield_net.predict([predicted_pattern], iterations=1, async_update=True)[0][index]

    steps.append(predicted_pattern.reshape(5, 5))

display_multiple_steps(steps, title="Steps for 'A' Pattern with Noise")

plt.figure(figsize=(8, 5))
plt.plot(energies, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Energy Convergence for Pattern 'A'")
plt.grid(True)
plt.show()