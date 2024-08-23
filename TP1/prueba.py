def read_file_to_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convertir cada línea del archivo en una fila de la matriz, quitando los caracteres de nueva línea.
    matrix = [list(line.rstrip()) for line in lines]

    return matrix

if __name__ == "__main__":
    file_path = 'BOARDS\LEVELS\easy.txt'  # Cambia esto a la ruta correcta de tu archivo
    matrix = read_file_to_matrix(file_path)
    
    # Imprimir la matriz para verificar
    for row in matrix:
        print(''.join(row))
