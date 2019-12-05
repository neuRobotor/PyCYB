import numpy as np
from scipy.stats import norm, chi
import os
import csv
import json

data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\Data'
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
for chnl in range(len(X)):
    print(chnl)
    data = X[chnl, :]
    # data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])
    data = np.abs(data)
    params.append(list(chi.fit(data)))

with open(data_path + r'\params.csv', "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(params)