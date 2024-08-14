import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Upload the data from csv
summary_2b = pd.read_csv("capture_summary_2b_noise.csv")

# Filter the data
pk_data = summary_2b[
    (summary_2b['pokemon'] == 'caterpie') &
    (summary_2b['status_effect'] == 'NONE') &
    (summary_2b['pokeball'] == 'ultraball')
]

# Sort the hp values
pk_data = pk_data.sort_values('health_points')

plt.figure(figsize=(10, 6))
plt.plot(pk_data['health_points'], pk_data['average_capture_rate'], 
         'o-', label='Capture Rate', linewidth=1)

# Shadow for the variance
plt.fill_between(pk_data['health_points'], 
                 pk_data['average_capture_rate'] - np.sqrt(pk_data['variance']),
                 pk_data['average_capture_rate'] + np.sqrt(pk_data['variance']),
                 color='b', alpha=0.2, label='Variance')

plt.title('Caterpie con Ultraball con ruido')
plt.xlabel('Puntos de vida')
plt.ylabel('Probabilidad de captura')
plt.grid(True)
plt.legend()

# To save the images
#plt.savefig('C:/Users/virgi/OneDrive/Escritorio/ITBA/SIA/TP0/IMAGES/2B_noise/cater_ultra_ruido.png', bbox_inches='tight')


plt.show()
