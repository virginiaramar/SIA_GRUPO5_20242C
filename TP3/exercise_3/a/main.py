import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mlp.multilayer_perceptron import *
import time
from datetime import datetime
import json
import plotly.graph_objects as go

def main():
    with open("config.json") as file:
        config = json.load(file)

    input_data_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [[1, 0], [1, 0], [0, 1], [0, 1]]

    neuronNetwork = MultiPerceptron(config["architecture"]["entry_layer_amount"],
                                    config["architecture"]["hidden_layer_amount"],
                                    config["architecture"]["neurons_per_layer"],
                                    config["architecture"]["output_layer_amount"],
                                    theta_logistic,
                                    theta_logistic_derivative,
                                    config["learning_constant"],
                                    config["activation_function"]["beta"],
                                    )

    start_time = time.time()
    error, w_min, metrics = neuronNetwork.train(
        config["epsilon"],
        config["epoch"],
        config["optimization_method"]["alpha"],
        input_data_xor,
        expected_output,
        collect_metrics,
        config["batch_size"]
    )
    end_time = time.time()
    print(error, end_time - start_time)

    if config["print_final_values"]:
        for input_data, output in zip(input_data_xor, expected_output):
            print(f"result: {neuronNetwork.forward_propagation(input_data)}, expected output: {output}")

    if config["generate_error_graph"]:
        export_metrics(metrics)
        generate_error_scatter()


def collect_metrics(metrics, error, iteration):
    metrics["error"].append(error)
    metrics["iteration"] = iteration


def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"./results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))


def generate_error_scatter(file_name=None):
    if file_name is None:
        path = "./results"
        files = os.listdir(path)
        files.sort()
        file_name = f"{path}/{files[-1]}"

        with open(file_name) as file:
            results = json.load(file)

        errors = results["error"]
        num_iterations = list(range(1, len(errors) + 1))  # Generate a list of iteration numbers

        # Create a scatter plot using Plotly Graph Objects
        fig = go.Figure(data=go.Scatter(x=num_iterations, y=errors, mode='markers'))
        fig.update_layout(title="Error Scatter Plot")
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Error")

        fig.show()

#Agregar el dibujo de plot

main()