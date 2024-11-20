import re
import numpy as np

def read_font_file(file_path):
    """Lee el archivo font.h y convierte los patrones hexadecimales en binarios"""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {file_path}.")
        return []

    pattern = re.compile(r'\{(.*?)\}')
    font_patterns = []
    
    for line in lines:
        match = pattern.search(line)
        if match:
            hex_values = [int(value.strip(), 16) for value in match.group(1).split(',')]
            font_patterns.append(hex_values)
    
    return font_patterns

def preprocess_patterns(patterns):
    """Convierte los patrones hexadecimales en vectores binarios"""
    binary_patterns = []
    for pattern in patterns:
        binary_pattern = [int(bit) for hex_val in pattern for bit in f"{hex_val:05b}"]
        binary_patterns.append(binary_pattern)
    return np.array(binary_patterns, dtype=int)  # Cambiado a int para asegurar que los valores sean 0 o 1

def save_to_txt(data, file_path):
    """Guarda los datos binarios en un archivo .txt"""
    with open(file_path, 'w') as f:
        for line in data:
            f.write(" ".join(map(str, line)) + "\n")

def convert_font_to_bin(input_path, output_path):
    """Convierte el archivo font.h a binario y lo guarda en un archivo .txt"""
    patterns = read_font_file(input_path)
    binary_data = preprocess_patterns(patterns)
    save_to_txt(binary_data, output_path)
    print(f"Datos binarios guardados en {output_path}")


if __name__ == "__main__":
    input_path = 'input/font.h'  # Path al archivo font.h
    output_path = 'input/font_vectors.txt'  # Path donde guardar el archivo .txt
    convert_font_to_bin(input_path, output_path)

