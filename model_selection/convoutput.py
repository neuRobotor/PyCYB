from model_selection.model_compare import depthwise_output
import numpy as np
from matplotlib.pyplot import imshow, show, figure

model = depthwise_output()
inp = np.concatenate((np.arange(20).reshape((20, 1)) * -1, np.zeros((20, 1)), np.ones((20, 1))*10, np.zeros((20, 4)),
                      np.arange(20).reshape((20, 1))), axis=1)
inp = np.expand_dims(inp, 0)
inp = np.expand_dims(inp, 0)
print(inp)

p, inter, inter2 = model.predict(inp)
imshow(inp[0, 0, :, :], cmap='gray')
figure()
imshow(inter[0, 0, :, :])
figure()
imshow(inter2[0, 0, :, :])
figure()
imshow(np.tile(p[:, :], (200, 1)))

impact = p
impacts = p.reshape(5, 8, 4)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(constrained_layout=False, figsize=(16, 8))
fig.suptitle("Mean absolute value of weighing into dense layers")
gs = GridSpec(5, 8, figure=fig)
col_map = "gray"
muscle_names = ['L Internal Oblique', 'R Internal Oblique', 'L External Oblique', 'R External Oblique',
                'Latissimus Dorsi', 'Transverse Trapezius', 'Erector Spinae', 'ECG']
ax = fig.add_subplot(gs[:])
ax.set_ylabel("MaxPooling( pool size 5) of time axis")
ax.xaxis.set_visible(False)
# make spines (the box) invisible
plt.setp(ax.spines.values(), visible=False)
# remove ticks and labels for the left axis
ax.tick_params(left=False, labelleft=False)
# remove background patch (only needed for non-white background)
ax.patch.set_visible(False)
for col in range(8):
    print(col)
    for row in range(5):
        ax = fig.add_subplot(gs[row * 8 + col])
        delta = np.max(impact) - np.min(impact)
        delta = 0.0 * delta
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


show()
print(p)
