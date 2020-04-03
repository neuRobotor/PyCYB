import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import numpy as np
from matplotlib.gridspec import GridSpec

sns.set_style('white')
model = load_model('Models/best_model_cyb104.h5')
filters1, biases = model.layers[0].get_weights()
filters2, _ = model.layers[2].get_weights()

fig = plt.figure(constrained_layout=False, figsize=(12.8, 7.2))
fig.suptitle("Primary and Secondary Features")
gs = GridSpec(4, 2, figure=fig)
muscle_names = ['L Internal Oblique', 'R Internal Oblique', 'L External Oblique', 'R External Oblique',
                    'L Trapezius', 'R Trapezius', 'Erector Spinae', 'ECG']
for i in range(8):
    print(i)
    inner_grid = gs[i].subgridspec(2, 4)
    ax = fig.add_subplot(gs[i])
    ax.set_title(muscle_names[i])
    plt.axis('off')
    for j in range(8):
        ax = fig.add_subplot(inner_grid[j])
        col_map = 'gray'

        if j < 4:
            delta = np.max(filters1) - np.min(filters1)
            delta = delta*0.05
            sns.heatmap(filters1[:, :, i, j % 4], ax=ax, vmin=np.min(filters1)-delta, vmax=np.max(filters1)+delta,
                        cmap=col_map, cbar=False, xticklabels=False, yticklabels=False)
            if j % 4 == 0:
                ax.set_ylabel("Primary")
        else:
            delta = np.max(filters2) - np.min(filters2)
            delta = delta*0.05
            sns.heatmap(filters2[:, :, i, j % 3], ax=ax, vmin=np.min(filters2)-delta, vmax=np.max(filters2)+delta,
                        cmap=col_map, cbar=False, xticklabels=False, yticklabels=False)
            if j % 4 == 0:
                ax.set_ylabel("Secondary")

plt.subplots_adjust(hspace=0.35, wspace=0.12)
plt.savefig("filters_trans", transparent=True)
