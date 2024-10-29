# TP SIA - GRUPO 5

## Team Members
- **Madero Torres, Eduardo Federico** - 59494
- **Ramos Marca, María Virginia** - 67200
- **Pluss, Ramiro** - 66254
- **Kuchukhidze, Giorgi** - 67262

## Exercise 1.1: Self-Organizing Maps (SOM)

This exercise implements Kohonen's Self-Organizing Map to analyze socioeconomic data from European countries, identifying patterns and natural groupings in multidimensional data.

### Features Analyzed
- Area
- GDP (Gross Domestic Product)
- Inflation
- Life Expectancy
- Military Expenditure
- Population Growth
- Unemployment

### SOM Configuration
```json
{
    "grid_size": 4,            # Grid size (4x4)
    "learning_rate": 0.001,    # Learning rate
    "initial_radius": 5,       # Initial neighborhood radius
    "n_iterations": 10000,     # Number of iterations
    "weights_init": "sample",  # Weight initialization method -> sample or random
    "distance_metric": "euclidean", # Distance metric -> euclidean or exponential
    "plot_feature_heatmaps": true, #Generate heatmaps for each variable. -> true or false
    "constant_radius": false #Set decremental or constant radius 
}
```

### Instructions
1. Ensure you have all required dependencies installed:
   ```bash
   pip install numpy pandas matplotlib
   ```

2. Run the SOM analysis:
   ```bash
   python kohonen.py
   ```

3. The script will generate several visualizations in the `output` directory:
   - Country distribution map
   - Neighbor distances heatmap
   - Feature-specific heatmaps
   - Detailed JSON file with cluster assignments

### Data Preprocessing
- Data is loaded from a CSV file containing European country statistics
- Features are standardized using Z-score normalization: (x - μ) / σ
- Missing values are handled appropriately

### Visualization Outputs
1. **Country Distribution Map**: Shows how countries are clustered across the SOM grid
2. **Neighbor Distances**: Visualizes the topological relationships between neurons
3. **Feature Heatmaps**: Individual heatmaps for each analyzed feature
4. **Cluster Assignments**: Detailed JSON file with cluster statistics and country groupings

## Exercise 1.2
This exercise applies Oja's rule to analyze socio-economic data from European countries, identifying the principal component (PC1) and comparing it to standard PCA. The analysis observes feature contributions to PC1 and examines the impact of learning rates on Oja's rule convergence relative to traditional PCA.

### Instructions
- Run the exercise by executing `oja.py`.
- To adjust settings like the number of epochs or learning rate, modify the parameters directly within the function definitions.
- To display or hide specific graphs:
   - Simply uncomment or comment the line that calls the corresponding function in `oja.py`.
- Ensure all necessary libraries are installed to avoid any runtime issues.

## Exercise 2

This exercise explores the implementation of a Hopfield Network, which is trained on various letter patterns and tested for pattern recognition accuracy under different conditions, such as increasing levels of noise and repeated trials.

### Instructions

#### 1. Training and Displaying Results with Specific Letters
To train the Hopfield Network with specific letters and display the recognition results:
1. Open **`letters.py`**.
2. Initialize the training set with the letters of your choice. Examples of letter patterns can be found in **`letters.txt`**.
3. Configure parameters like the **number of iterations** and the **noise level** to test different recognition scenarios.
4. Run the program to see the results, with visual outputs showing the network's performance in recognizing noisy patterns.

#### 2. Running Multiple Simulations with Varying Noise Levels
To test the network's accuracy across multiple simulations, with noise levels incrementally increasing from 0 to 0.5:
1. Open **`multiple_simulations.py`**.
2. Set the desired **number of simulations** using the `trials` variable.
3. Run the program to conduct the simulations. After completion, it will output the following averages:
   - **Correct Probability**: Likelihood of correctly identifying a noisy pattern.
   - **Incorrect Probability**: Likelihood of incorrect pattern recognition.
   - **Spurious Probability**: Likelihood of the network entering a spurious state (unintended stable state).

#### 3. Testing Spurious States and Displaying Energy Graphs
To analyze spurious states and visualize the network's energy landscape:
1. Run **`spurious_energy.py`**.
2. Adjust parameters such as the **number of iterations** and the **noise level** to test different scenarios.
3. This script will output an energy graph, allowing you to observe the Hopfield Network's stability as it converges to different states.

---

Each script allows for customization of key parameters, enabling detailed exploration of the network's behavior under various training and testing conditions. Modify the variables as needed to tailor the experiments to specific objectives.

---

### Additional Notes
- **Letters Format**: The letter patterns in `letters.txt` are represented as matrices of `1` (black) and `-1` (white), where each entry corresponds to a pixel in a 5x5 grid.
- **Energy Graphs**: The energy function provides insight into the stability of different states in the Hopfield Network, which is useful for understanding how the network avoids or settles into spurious states.

### Requirements
Ensure you have installed any necessary dependencies (e.g., `numpy`, `matplotlib`) before running the scripts.
