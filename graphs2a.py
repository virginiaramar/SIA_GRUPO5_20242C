import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos (asegúrate de que el archivo CSV está en el mismo directorio o proporciona la ruta completa)
summary_2a = pd.read_csv("capture_summary.csv")


# Filtrar datos para Snorlax
snorlax_data = summary_2a[summary_2a['pokemon'] == 'snorlax']
# Configurar estilo de gráficos
plt.figure(figsize=(14, 8))
sns.barplot(x='pokeball', y='effectiveness', hue='status_effect', data=snorlax_data)
plt.title('Efectividad de Captura de Snorlax con Diferentes Pokébolas y Estados de Salud')
plt.xlabel('Pokébola')
plt.ylabel('Efectividad')
plt.legend(title='Estado de Salud')
plt.show()
