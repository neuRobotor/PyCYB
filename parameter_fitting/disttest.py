import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import chi
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set()

# Load data from statsmodels datasets
data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\Data'
n_channels = 8
X = np.empty((n_channels, 0))
for i, file in enumerate(sorted([f for f in os.listdir(data_path) if f.endswith('.json')])):
    with open(data_path+'\\'+file) as json_file:
        dict_data = json.load(json_file)
        X = np.concatenate((X, dict_data["EMG"]), axis=1)
    if i>=9:
        break

print("Loaded")
data = X[0, :]
data = (data - np.mean(data))/np.std(data)
data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])

params = chi.fit(data)

# Separate parts of parameters
arg = params[:-2]
loc = params[-2]
scale = params[-1]

# Calculate fitted PDF and error with fit in distribution
transf = chi.cdf(data, loc=loc, scale=scale, *arg)
transf = norm.ppf(transf)
plt.figure()
plt.hist(transf, bins=50)

plt.show()

