# TP SIA - GRUPO 5

## Team Members
- **Madero Torres, Eduardo Federico** - 59494
- **Ramos Marca, María Virginia** - 67200
- **Pluss, Ramiro** - 66254
- **Kuchukhidze, Giorgi** - 67262

## Overview
This project explores the implementation of a Hopfield Network, which is trained on various letter patterns and tested for pattern recognition accuracy under different conditions, such as increasing levels of noise and repeated trials.

## Instructions

### 1. Training and Displaying Results with Specific Letters
To train the Hopfield Network with specific letters and display the recognition results:
1. Open **`letters.py`**.
2. Initialize the training set with the letters of your choice. Examples of letter patterns can be found in **`letters.txt`**.
3. Configure parameters like the **number of iterations** and the **noise level** to test different recognition scenarios.
4. Run the program to see the results, with visual outputs showing the network’s performance in recognizing noisy patterns.

### 2. Running Multiple Simulations with Varying Noise Levels
To test the network's accuracy across multiple simulations, with noise levels incrementally increasing from 0 to 0.5:
1. Open **`multiple_simulations.py`**.
2. Set the desired **number of simulations** using the `trials` variable.
3. Run the program to conduct the simulations. After completion, it will output the following averages:
   - **Correct Probability**: Likelihood of correctly identifying a noisy pattern.
   - **Incorrect Probability**: Likelihood of incorrect pattern recognition.
   - **Spurious Probability**: Likelihood of the network entering a spurious state (unintended stable state).

### 3. Testing Spurious States and Displaying Energy Graphs
To analyze spurious states and visualize the network's energy landscape:
1. Run **`spurious_energy.py`**.
2. Adjust parameters such as the **number of iterations** and the **noise level** to test different scenarios.
3. This script will output an energy graph, allowing you to observe the Hopfield Network’s stability as it converges to different states.

---

Each script allows for customization of key parameters, enabling detailed exploration of the network's behavior under various training and testing conditions. Modify the variables as needed to tailor the experiments to specific objectives.

---

### Additional Notes
- **Letters Format**: The letter patterns in `letters.txt` are represented as matrices of `1` (black) and `-1` (white), where each entry corresponds to a pixel in a 5x5 grid.
- **Energy Graphs**: The energy function provides insight into the stability of different states in the Hopfield Network, which is useful for understanding how the network avoids or settles into spurious states.

### Requirements
Ensure you have installed any necessary dependencies (e.g., `numpy`, `matplotlib`) before running the scripts.
