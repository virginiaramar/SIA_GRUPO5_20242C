import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de resumen desde el archivo CSV
summary_2b = pd.read_csv("capture_summary_2b.csv")

# Filtrar por Snorlax, StatusEffect.NONE y Pokéball
snorlax_data = summary_2b[
    (summary_2b['pokemon'] == 'mewtwo') &
    (summary_2b['status_effect'] == 'FREEZE') &
    (summary_2b['pokeball'] == 'heavyball')
]

# Ordenar los datos por los puntos de salud (health_points)
snorlax_data = snorlax_data.sort_values('health_points')

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.errorbar(snorlax_data['health_points'], snorlax_data['average_capture_rate'], 
             yerr=np.sqrt(snorlax_data['variance']), fmt='o-', label='Capture Rate')
plt.fill_between(snorlax_data['health_points'], 
                 snorlax_data['average_capture_rate'] - np.sqrt(snorlax_data['variance']),
                 snorlax_data['average_capture_rate'] + np.sqrt(snorlax_data['variance']),
                 color='b', alpha=0.2, label='Variance')

plt.title('Capture Rate vs Health Points for Snorlax with Pokéball and No Status Effect')
plt.xlabel('Health Points')
plt.ylabel('Average Capture Rate')
plt.grid(True)
plt.legend()
plt.show()
