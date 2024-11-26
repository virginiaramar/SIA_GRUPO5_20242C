import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def run_experiment(config_path):
    """Ejecuta un experimento con una configuración específica"""
    # Cargar configuración
    with open(config_path) as f:
        config = json.load(f)
    
    # Crear directorio para resultados
    experiment_name = Path(config_path).stem
    results_dir = f"results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir)
    
    # Guardar configuración
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Ejecutar experimento
    history = vae.train(data)
    
    # Guardar resultados
    results = {
        'config_name': experiment_name,
        'final_loss': history[-1],
        'best_loss': min(history),
        'convergence_epoch': history.index(min(history)),
        'config': config
    }
    
    # Guardar visualizaciones
    plot_all_results(vae, history, results_dir)
    
    return results

def run_all_experiments():
    """Ejecuta todos los experimentos en la carpeta experiments"""
    results = []
    
    # Recorrer todas las carpetas de experimentos
    for exp_dir in sorted(os.listdir('experiments')):
        exp_path = f'experiments/{exp_dir}'
        if not os.path.isdir(exp_path):
            continue
            
        print(f"\nEjecutando experimentos en {exp_dir}")
        
        # Ejecutar cada configuración
        for config_file in os.listdir(exp_path):
            if not config_file.endswith('.json'):
                continue
                
            config_path = f"{exp_path}/{config_file}"
            print(f"\nEjecutando {config_file}...")
            
            try:
                result = run_experiment(config_path)
                results.append(result)
                print(f"Completado: Loss final = {result['final_loss']:.4f}")
            except Exception as e:
                print(f"Error en {config_file}: {str(e)}")
    
    # Guardar resumen de resultados
    df = pd.DataFrame(results)
    df.to_csv('results_summary.csv', index=False)
    print("\nResultados guardados en results_summary.csv")

if __name__ == "__main__":
    run_all_experiments()