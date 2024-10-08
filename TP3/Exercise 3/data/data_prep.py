import numpy as np

# Función para cargar y procesar los dígitos
def load_and_flatten_digits(input_filename, output_filename):
    # Leer el archivo de entrada
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    digits = []
    current_digit = []

    # Proceso para organizar los dígitos
    for line in lines:
        line = line.strip()
        if line == "":  # Detecta separaciones entre dígitos
            if current_digit:
                digits.append(current_digit)
                current_digit = []
        else:
            current_digit.append([int(char) for char in line if char.isdigit()])
    
    if current_digit:  # Añade el último dígito si no hay separación final
        digits.append(current_digit)
    
    # Aplanar los dígitos
    flattened_digits = [np.array(digit).flatten() for digit in digits]
    
    # Guardar los datos aplanados en un archivo de texto
    with open(output_filename, 'w') as output_file:
        for digit in flattened_digits:
            output_file.write(' '.join(map(str, digit)) + '\n')

# Llamar a la función, especificando los archivos
input_filename = 'data/TP3-ej3-digitos.txt'  # Cambia el nombre si es necesario
output_filename = 'data/digitos_procesados.txt'  # Cambia este nombre para el archivo que desees
load_and_flatten_digits(input_filename, output_filename)
