import pandas as pd
import os

def get_average_best_fitness_for_last_generation(file_path):
    """Calculates the average best fitness for the last generation of each simulation in a CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        # Group by 'Simulation' and get the best fitness of the last generation for each simulation
        last_generation_best_fitness = df.groupby('Simulation').apply(
            lambda x: x.loc[x['Generation'].idxmax(), 'Best Fitness']
        )
        
        # Calculate the average best fitness across all simulations in this file
        average_best_fitness = last_generation_best_fitness.mean()
        
        return average_best_fitness
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_csv_folder(directory):
    """Processes all CSV files in the specified folder to find the top and bottom 10 files based on average best fitness."""
    results = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            average_fitness = get_average_best_fitness_for_last_generation(file_path)
            if average_fitness is not None:
                results.append((filename, average_fitness))
    
    # Convert results to a DataFrame for easier manipulation
    results_df = pd.DataFrame(results, columns=['File', 'Average Best Fitness'])
    
    # Remove duplicates and find top 10 and bottom 10 unique files
    unique_results_df = results_df.drop_duplicates(subset=['File', 'Average Best Fitness'])

    # Find top 10 files with the highest average fitness
    best_files = unique_results_df.nlargest(10, 'Average Best Fitness')
    
    # Find top 10 files with the lowest average fitness
    worst_files = unique_results_df.nsmallest(10, 'Average Best Fitness')
    
    return best_files, worst_files

def main():
    directory = 'output/hibrido/selection'  # Path to your folder containing CSV files
    best_files, worst_files = process_csv_folder(directory)
    
    print("Top 10 files with the best average fitness:")
    print(best_files)
    
    print("\nTop 10 files with the worst average fitness:")
    print(worst_files)

if __name__ == "__main__":
    main()
