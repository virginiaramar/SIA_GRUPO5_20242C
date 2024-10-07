import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from exercise_3.multilayer import *
import time
from datetime import datetime
import json
import plotly.graph_objects as go

def main():
    config_path = "config.json"

    input_data_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [[1, 0], [1, 0], [0, 1], [0, 1]]

    neuronNetwork = MultiPerceptron(config_path)

    start_time = time.time()
    error, metrics = neuronNetwork.train(input_data_xor, expected_output, collect_metrics)
    end_time = time.time()
    print(f"Final error: {error}, Training time: {end_time - start_time} seconds")

    if neuronNetwork.print_final_values:
        for input_data, output in zip(input_data_xor, expected_output):
            result = neuronNetwork.forward_propagation(input_data)
            print(f"Input: {input_data}, Result: {result}, Expected output: {output}")

    if neuronNetwork.generate_error_graph:
        export_metrics(metrics)
        generate_error_scatter()

def collect_metrics(metrics, error, iteration):
    metrics["error"].append(error)
    metrics["iteration"] = iteration

def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/results_{now}.json", mode="w+") as file:
        json.dump(metrics, file, indent=4)

def generate_error_scatter(file_name=None):
    if file_name is None:
        path = "./results"
        files = os.listdir(path)
        files.sort()
        file_name = f"{path}/{files[-1]}"

    with open(file_name) as file:
        results = json.load(file)

    errors = results["error"]
    num_iterations = list(range(1, len(errors) + 1))

    fig = go.Figure(data=go.Scatter(x=num_iterations, y=errors, mode='markers'))
    fig.update_layout(title="Error Scatter Plot")
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Error")

    fig.show()

if __name__ == "__main__":
    main()