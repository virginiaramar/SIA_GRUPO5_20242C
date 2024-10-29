import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca_analysis(file_path):
    # Leer los datos desde el archivo CSV
    europe_data = pd.read_csv(file_path)

    # Extraer datos numéricos y estandarizarlos
    numeric_data = europe_data.drop(columns=['Country'])
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)

    # Calcular la matriz de covarianza para los datos estandarizados
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Calcular la matriz de correlación
    correlation_matrix = np.corrcoef(standardized_data.T)

    # Calcular autovalores y autovectores de la matriz de correlación (PCA manual)
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    # Realizar la PCA manualmente proyectando los datos originales en las componentes principales
    pca_components_manual = np.dot(standardized_data, eigenvectors)

    # Crear un DataFrame para los resultados de PCA manual
    pca_df_manual = pd.DataFrame(pca_components_manual, columns=[f'PC{i+1}' for i in range(pca_components_manual.shape[1])])

    # Realizar PCA usando la librería sklearn para comparar
    pca = PCA()
    pca.fit(standardized_data)
    explained_variance = pca.explained_variance_ratio_

    # Variancia explicada de manera manual
    explained_variance_manual = eigenvalues / np.sum(eigenvalues)

    # Calcular las cargas (loadings) de las componentes principales
    loadings_df = pd.DataFrame(eigenvectors, index=numeric_data.columns, columns=[f'PC{i+1}' for i in range(eigenvectors.shape[1])])

    # Crear DataFrame que combine los países con los valores de las componentes principales calculados manualmente
    countries_pc_df = pd.concat([europe_data['Country'], pca_df_manual], axis=1)

    # Obtener las cargas para PC1
    pc1_loadings = loadings_df['PC1']

    # Devolver los resultados necesarios para análisis posterior
    return {
        "countries_pc": countries_pc_df,
        "explained_variance_sklearn": explained_variance,
        "explained_variance_manual": explained_variance_manual,
        "pc1_loadings": pc1_loadings
    }
