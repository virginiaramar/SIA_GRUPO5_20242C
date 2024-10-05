import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

# Import the class multilayer perceptron
from multilayer_perceptron import multilayer_perceptron 



##### EJERCICIO 3.1 #####

# To evaluate XOR, change these in the config
# "input": [[0, 0], [0, 1], [1, 0], [1, 1]],
# "output": [[0], [1], [1], [0]]
# Once changed, select 1 at the end of this code


def plot_decision_boundary(mlp, X, y):
    # Define plot limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict for each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=ListedColormap(cmap_bold), edgecolor='k', s=100)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary learned by MLP")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

def run_xor_exercise():
    mlp = multilayer_perceptron('config.json')
    # Train the model
    mlp.multilayer_algorithm()

    # Evaluate performance
    mlp.evaluate()

    # Plot decision boundary
    plot_decision_boundary(mlp, mlp.X, mlp.y)







if __name__ == "__main__":
    # Choose which exercise to run
    exercise = 1
    
    if exercise == 1:
        run_xor_exercise()
    elif exercise == 2:
        # Placeholder for Exercise 2
        print("Exercise 2 will be implemented.")
    elif exercise == 3:
        # Placeholder for Exercise 3
        print("Exercise 3 will be implemented.")
    else:
        print("Invalid exercise. Please choose 1, 2, or 3.")

