import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
summary_2a = pd.read_csv("capture_summary.csv")

# Normalise the data
normalized_summary = summary_2a.copy()
# Normalise with NONE status
none_data = summary_2a[summary_2a['status_effect'] == 'NONE']

# Join df
normalized_summary = normalized_summary.merge(
    none_data[['pokeball', 'pokemon', 'effectiveness', 'variance']],
    on=['pokeball', 'pokemon'],
    suffixes=('', '_none')
)

# normalise the effectiveness and variance
normalized_summary['normalized_effectiveness'] = (
    normalized_summary['effectiveness'] / normalized_summary['effectiveness_none']
)
normalized_summary['normalized_variance'] = (
    normalized_summary['variance'] / normalized_summary['variance_none']
)

# Eliminate the columns none
normalized_summary = normalized_summary.drop(columns=['effectiveness_none', 'variance_none'])


pokemons = ["jolteon", "caterpie", "snorlax", "onix", "mewtwo"]

# Create the subplots for all the pokemons
fig, axes = plt.subplots(nrows=len(pokemons), ncols=1, figsize=(12, 6 * len(pokemons)))


for i, pokemon in enumerate(pokemons):
    # Filter the data for the each one
    pokemon_data = normalized_summary[normalized_summary['pokemon'] == pokemon]
    
    if pokemon_data.empty:
        continue
    
    # Grapg creation: bar graph in this case
    ax = sns.barplot(
        ax=axes[i],
        x='pokeball',
        y='normalized_effectiveness',
        hue='status_effect',
        data=pokemon_data,
        errorbar=None
    )
    
    # Add the value on top of each bar to better visualization
    for bar in ax.patches:
        # x,y to localise the text
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        
        # Grey color and fontsize
        ax.text(
            x, y + 0.02, f'{y:.2f}', ha='center', va='bottom', 
            color='grey', fontsize=9
        )

    # Titles and labels for the graphs
    axes[i].set_title(f'{pokemon.capitalize()}', fontsize=14, pad=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Efectividad')
    
    # Take out the legend
    axes[i].legend().remove()

# We wanted only one legend for all the subplots, this adds it outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Estado de Salud', loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=10)

# Main title
fig.suptitle('Efectividad de captura normalizada con todas las pokébolas y estados de salud', fontsize=16, y=1)

# Layout for the subplots
plt.tight_layout(rect=[0, 0, 0.85, 0.95], h_pad=4)

# Subplots adjustment
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(bottom=0.2)  
plt.subplots_adjust(hspace=0.8)  


plt.show()






#######################################3


# Filters data only for ONE pokemon + variance
snorlax_data = normalized_summary[normalized_summary['pokemon'] == 'onix']


plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x='pokeball',
    y='normalized_effectiveness',
    hue='status_effect',
    data=snorlax_data,
    ci=None
)

# Adds error bars on top of the data bars
for bar, err in zip(ax.patches, snorlax_data['normalized_variance']):
    x = bar.get_x() + bar.get_width() / 2  
    y = bar.get_height()  
    ax.errorbar(
        x=x, y=y, yerr=err,
        fmt='none', color='grey', capsize=5, capthick=2, elinewidth=2
    )


plt.title('Efectividad de captura normalizada para Onix', fontsize=16, pad=10)
plt.xlabel('Pokébola', fontsize=12)
plt.ylabel('Efectividad Normalizada', fontsize=12)


plt.legend(title='Estado de Salud', loc='upper right', bbox_to_anchor=(1.15, 1))


plt.tight_layout()


plt.show()
