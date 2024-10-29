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

def add_noise(pattern, noise_level=0.15):
    """Add noise to a flattened pattern by flipping some bits."""
    noisy_pattern = pattern.copy()
    num_noisy_bits = int(noise_level * len(pattern))
    indices = np.random.choice(len(pattern), num_noisy_bits, replace=False)
    noisy_pattern[indices] = -noisy_pattern[indices]
    return noisy_pattern

def check_pattern_match(predicted, original_patterns):
    """Check if the predicted pattern matches any of the original patterns."""
    for i, pattern in enumerate(original_patterns):
        if np.array_equal(predicted, pattern):
            return i
    return -1

def get_pattern_index(pattern, original_patterns):
    """Return the index of the pattern in original_patterns if it matches; else -1."""
    for i, original_pattern in enumerate(original_patterns):
        if np.array_equal(pattern, original_pattern):
            return i
    return -1

noise_levels = np.arange(0, 0.6, 0.1)
results = []

for noise_level in noise_levels:
    correct_count = 0
    incorrect_count = 0
    spurious_count = 0
    trials = 100

    for pattern in patterns_flat:
        for _ in range(trials):
            noisy_pattern = add_noise(pattern, noise_level=noise_level)
            predicted_pattern = hopfield_net.predict([noisy_pattern], iterations=10, threshold=0, async_update=True)[0]
            match_index = get_pattern_index(predicted_pattern, patterns_flat)

            if match_index == get_pattern_index(pattern, patterns_flat):
                correct_count += 1
            elif match_index != -1:
                incorrect_count += 1
            else:
                spurious_count += 1

    total_attempts = trials * len(patterns_flat)
    correct_prob = correct_count / total_attempts
    incorrect_prob = incorrect_count / total_attempts
    spurious_prob = spurious_count / total_attempts
    results.append((noise_level, correct_prob, incorrect_prob, spurious_prob))

print("Noise Level | Correct Probability | Incorrect Probability | Spurious Probability")
for noise_level, correct_prob, incorrect_prob, spurious_prob in results:
    print(f"{noise_level:.1f}         | {correct_prob:.3f}             | {incorrect_prob:.3f}            | {spurious_prob:.3f}")