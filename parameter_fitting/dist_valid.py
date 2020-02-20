import numpy as np
from scipy.stats import norm, chi
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
import seaborn as sns

def make_pdf(dist, params, size=100000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.9999999, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
sns.set()
sns.set_context('paper')
fig, axes = plt.subplots(4, 2)
n_channels = 8
X = np.empty((n_channels, 0))
for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
    with open(data_path + '\\' + file) as json_file:
        dict_data = json.load(json_file)
    emg_data = np.array(dict_data["EMG"])
    X = np.hstack((X, emg_data))
X0 = np.array(X)
X_std = np.std(X, axis=1)
X_mean = np.mean(X, axis=1)
X = np.abs((X - X_mean[:, None]) / X_std[:, None])


with open(data_path+r'\params.csv') as f:
    reader = csv.reader(f)
    params = list(reader)
    params = [[float(el) for el in row] for row in params]
for i, p in enumerate(params):
    pdf = make_pdf(chi, p)
    axes.flatten()[i].plot(pdf, lw=2)
    y, bins, _ = axes.flatten()[i].hist(X[i,:], bins=200, density=True, rwidth=1, edgecolor=sns.color_palette()[1])
    #axes.flatten()[i].set_ylim(top=np.max(y)+0.1)
    #axes.flatten()[i].set_xlim(right=np.max(bins) + 0.1)
    axes.flatten()[i].set_yscale("log")
    axes.flatten()[i].set_xlabel("Normalised voltage")
    axes.flatten()[i].set_ylabel("Probability density")

fig.tight_layout()
from old_net.convmemnet_chi import chi_norm
X1 = chi_norm(X0, params)
fig, axes = plt.subplots(4, 2)
plt.tight_layout()
for i, ecdf in enumerate(params):
    axes.flatten()[i].hist(X1[i, :], bins=100, density=True, rwidth=1, edgecolor=sns.color_palette()[0])
    if i%2 is 0:
        axes.flatten()[i].set_ylabel("Probability density")
    if i>5:
        axes.flatten()[i].set_xlabel("Normalised voltage")
plt.tight_layout()
plt.show()

plt.show()

print(chi.cdf(8.47, loc=params[6][-2], scale=params[6][-1], *params[6][:-2]))
# Display


n_channels = 8
X = np.empty((n_channels, 0))
for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
    with open(data_path + '\\' + file) as json_file:
        dict_data = json.load(json_file)
    emg_data = np.array(dict_data["EMG"])
    X = np.hstack((X, emg_data))
X_std = np.std(X, axis=1)
X_mean = np.mean(X, axis=1)
X = (X - X_mean[:, None]) / X_std[:, None]
params = list()
dist= chi
for chnl in range(len(X)):
    if chnl != 7:
        continue
    print(chnl)
    data = X[chnl, :]
    # data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])
    data = np.abs(data)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        params.append(list(dist.fit(data)))

for p in params:
    arg = p[:-2]
    loc = p[-2]
    scale = p[-1]
    a_max = dist.ppf(0.9999999999999999, loc=loc, scale=scale, *arg)
    a_min = dist.ppf(0.000000000001, loc=loc, scale=scale, *arg)
    transf = np.clip(X[7, :], a_min=a_min, a_max=a_max)
    transf = dist.cdf(transf, loc=loc, scale=scale, *arg)
    transf = norm.ppf(transf)
plt.figure()
plt.hist(transf, bins=100)

plt.show()