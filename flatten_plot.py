import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import numpy as np
from matplotlib.gridspec import GridSpec

sns.set_style('white')
model = load_model('filtered_w40_s1.h5')
model.summary()
weights, biases = model.layers[5].get_weights()
impact = np.mean(np.abs(weights), axis=1)
impacts = impact.reshape((6, 8, 16))
# impacts = np.rollaxis(impacts, 1, 0)
# impacts = impacts.reshape((8, 96))

fig = plt.figure(constrained_layout=False, figsize=(16, 8))
fig.suptitle("Mean absolute value of weighing into dense layers")
gs = GridSpec(6, 8, figure=fig)
col_map = "Greys"
muscle_names = ['L Internal Oblique', 'R Internal Oblique', 'L External Oblique', 'R External Oblique',
                'Latissimus Dorsi', 'Transverse Trapezius', 'Erector Spinae', 'ECG']
ax = fig.add_subplot(gs[:])
ax.set_ylabel("MaxPooling( pool size 5) of time axis")
ax.xaxis.set_visible(False)
# make spines (the box) invisible
plt.setp(ax.spines.values(), visible=False)
# remove ticks and labels for the left axis
ax.tick_params(left=False, labelleft=False)
#remove background patch (only needed for non-white background)
ax.patch.set_visible(False)
for col in range(8):
    print(col)
    for row in range(6):
        ax = fig.add_subplot(gs[row * 8 + col])
        delta = np.max(impact) - np.min(impact)
        delta = 0.2 * delta
        if row == 0:
            ax.set_title(muscle_names[col])
        if row == 5 and col == 0:
            sns.heatmap(np.expand_dims(impacts[row, col, :], 0), ax=ax,
                        vmin=np.min(impact) - delta, vmax=np.max(impact) + delta, cmap=col_map, cbar=False,
                        yticklabels=False)
            ax.set_xlabel("Features")
        else:
            sns.heatmap(np.expand_dims(impacts[row, col, :], 0), ax=ax, vmin=np.min(impact) - delta,
                        vmax=np.max(impact) + delta,
                        cmap=col_map, cbar=False, xticklabels=False, yticklabels=False)
#plt.show()
plt.savefig("impact", transparent=True)
